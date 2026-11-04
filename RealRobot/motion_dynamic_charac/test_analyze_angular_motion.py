#!/usr/bin/env python3
"""Synthetic-response checks for analyze_angular_motion.py (standard library only).

    python3 RealRobot/motion_dynamic_charac/test_analyze_angular_motion.py
"""

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_angular_motion as A  # noqa: E402

DT = 0.05          # 20 Hz odometry
PHASE = 0.013      # samples are not aligned with the command time


def sample(w_of_t, t_start, t_stop, yaw0=0.0, pivot_r=0.0):
    """Native-rate samples of an angular-velocity profile. Heading by fine integration; the odometry frame sits
    pivot_r from the centre of rotation (0 = rotating in place). yaw is reported wrapped, like the robot."""
    t, w, yaw, x, y, v = [], [], [], [], [], []
    tk, th, tau, h = t_start + PHASE, yaw0, t_start, 1e-4
    while tk < t_stop:
        while tau < tk:
            step = min(h, tk - tau)
            th += w_of_t(tau + step / 2) * step
            tau += step
        t.append(tk); w.append(w_of_t(tk)); yaw.append(A.wrap(th))
        x.append(pivot_r * math.cos(th)); y.append(pivot_r * math.sin(th)); v.append(abs(w_of_t(tk)) * pivot_r)
        tk += DT
    return {"t": t, "w": w, "yaw_raw": yaw, "yaw": A.unwrap(yaw), "x": x, "y": y, "v": v}


def first_order(w_prev, w_new, t_cmd, delay, tau):
    return lambda t: w_prev if t < t_cmd + delay else w_new + (w_prev - w_new) * math.exp(-(t - t_cmd - delay) / tau)


def second_order(w_prev, w_new, t_cmd, delay, zeta, wn):
    wd = wn * math.sqrt(1 - zeta ** 2)

    def f(t):
        s = t - t_cmd - delay
        if s < 0:
            return w_prev
        r = 1 - math.exp(-zeta * wn * s) * (math.cos(wd * s) + zeta / math.sqrt(1 - zeta ** 2) * math.sin(wd * s))
        return w_prev + (w_new - w_prev) * r
    return f


class Metrics(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(A.classify(0.0, -0.2), "startup")         # CW startup, not a deceleration
        self.assertEqual(A.classify(-0.2, -0.4), "acceleration")
        self.assertEqual(A.classify(-0.6, -0.2), "deceleration")
        self.assertEqual(A.classify(0.4, -0.4), "reversal")
        self.assertEqual(A.classify(-0.4, 0.0), "stop")

    def test_positive_acceleration(self):
        t_cmd, t_end, D, tau = 2.0, 5.0, 0.12, 0.15
        o = sample(first_order(0.2, 0.4, t_cmd, D, tau), 0.0, t_end + 1)
        m, flags, _ = A.transition_metrics(o, t_cmd, t_end, 0.2, 0.4)
        onset = D + tau * -math.log(1 - A.NOISE_RADPS / 0.2)
        self.assertTrue(onset <= m["response_delay_s"] < onset + DT + 1e-9, m["response_delay_s"])
        self.assertAlmostEqual(m["rise_time_s"], tau * math.log(9), delta=0.01)
        self.assertAlmostEqual(m["steady_state_error_radps"], 0.0, places=6)
        self.assertAlmostEqual(m["tracking_ratio"], 1.0, places=6)
        self.assertEqual(m["overshoot_pct"], 0.0)
        self.assertNotIn("translating_while_rotating", flags)

    def test_negative_startup(self):
        # 0 -> -0.4 is a clockwise acceleration: delay/rise along the negative direction, overshoot = beyond -0.4.
        o = sample(second_order(0.0, -0.4, 1.0, 0.15, 0.5, 12.0), 0.0, 5.0)
        m, _, _ = A.transition_metrics(o, 1.0, 4.0, 0.0, -0.4)
        self.assertTrue(0.15 < m["response_delay_s"] < 0.15 + 0.1, m["response_delay_s"])
        self.assertAlmostEqual(m["overshoot_pct"], 100 * math.exp(-math.pi * 0.5 / math.sqrt(0.75)), delta=1.5)
        self.assertLess(m["peak_radps"], -0.4)
        self.assertAlmostEqual(m["tracking_ratio"], 1.0, places=3)

    def test_reversal(self):
        # +0.4 -> -0.4: 0.8 rad/s step; 10/90 % levels are +0.32 and -0.32; overshoot past -0.4.
        zeta, wn = 0.6, 10.0
        o = sample(second_order(0.4, -0.4, 2.0, 0.12, zeta, wn), 0.0, 6.0)
        m, _, _ = A.transition_metrics(o, 2.0, 5.0, 0.4, -0.4)
        f = second_order(0.4, -0.4, 0.0, 0.0, zeta, wn)
        ts = [i * 1e-5 for i in range(100000)]
        rise = next(t for t in ts if f(t) <= -0.32) - next(t for t in ts if f(t) <= 0.32)
        self.assertAlmostEqual(m["rise_time_s"], rise, delta=0.01)
        self.assertAlmostEqual(m["overshoot_pct"], 100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2)),
                               delta=1.5)
        self.assertGreater(m["response_delay_s"], 0.12)

    def test_deceleration_negative(self):
        # -0.6 -> -0.2: change is positive; an undershoot towards 0 (beyond -0.2) is overshoot.
        o = sample(second_order(-0.6, -0.2, 1.0, 0.1, 0.5, 12.0), 0.0, 5.0)
        m, _, _ = A.transition_metrics(o, 1.0, 4.0, -0.6, -0.2)
        self.assertGreater(m["peak_radps"], -0.2)
        self.assertAlmostEqual(m["overshoot_pct"], 16.3, delta=1.5)

    def test_overtracking_sign(self):
        o = sample(first_order(0.0, -0.255, 1.0, 0.1, 0.05), 0.0, 4.0)        # commanded -0.2, turns at -0.255
        m, _, _ = A.transition_metrics(o, 1.0, 3.5, 0.0, -0.2)
        self.assertAlmostEqual(m["steady_state_error_radps"], 0.055, places=4)  # e = w_cmd - w_meas > 0 for CW
        self.assertAlmostEqual(m["tracking_ratio"], 1.275, places=3)

    def test_stop_angle_across_wrap(self):
        # Rotating CCW at 0.4 rad/s through +pi: the raw yaw jumps to -pi, the stopping angle must not.
        t_cmd, D, a = 3.0, 0.1, 2.0
        f = lambda t: 0.4 if t < t_cmd + D else max(0.0, 0.4 - a * (t - t_cmd - D))
        o = sample(f, 0.0, 6.0, yaw0=math.pi - 1.25)                        # crosses pi at about t = 3.1 s
        self.assertTrue(any(abs(b - a_) > math.pi for a_, b in zip(o["yaw_raw"], o["yaw_raw"][1:])))
        m, _, _ = A.transition_metrics(o, t_cmd, math.inf, 0.4, 0.0)
        t_stop = next(t for t, w in zip(o["t"], o["w"]) if t >= t_cmd and abs(w) <= A.STOP_RADPS)
        self.assertAlmostEqual(m["stopping_time_s"], t_stop - t_cmd, places=9)
        s = min(t_stop - t_cmd - D, 0.4 / a)
        expected = 0.4 * D + 0.4 * s - a * s * s / 2                        # 0.04 + 0.04 = 0.08 rad
        self.assertAlmostEqual(m["stopping_angle_rad"], expected, places=5)
        self.assertAlmostEqual(m["stopping_angle_deg"], math.degrees(expected), places=3)
        f_cw = lambda t: -f(t)                                              # same stop, clockwise
        o = sample(f_cw, 0.0, 6.0, yaw0=-math.pi + 1.25)
        m, _, _ = A.transition_metrics(o, t_cmd, math.inf, -0.4, 0.0)
        self.assertAlmostEqual(m["stopping_angle_rad"], -expected, places=5)
        self.assertAlmostEqual(m["abs_stopping_angle_rad"], expected, places=5)

    def test_unwrap(self):
        raw = [3.0, 3.1, -3.1, -3.0, 3.1, 3.0]                              # +pi crossing, then back
        u = A.unwrap(raw)
        self.assertAlmostEqual(u[2], 3.1 + (2 * math.pi - 6.2), places=12)
        self.assertAlmostEqual(u[-1], 3.0, places=12)

    def test_pivot_detection(self):
        f = first_order(0.0, 0.4, 1.0, 0.1, 0.05)
        m, flags, _ = A.transition_metrics(sample(f, 0.0, 5.0, pivot_r=0.117), 1.0, 4.0, 0.0, 0.4)
        self.assertIn("translating_while_rotating", flags)
        self.assertAlmostEqual(m["rotation_center_offset_m"], 0.117, delta=0.002)
        self.assertAlmostEqual(m["max_abs_v_linear_mps"], 0.4 * 0.117, delta=0.002)
        m, flags, _ = A.transition_metrics(sample(f, 0.0, 5.0, pivot_r=0.0), 1.0, 4.0, 0.0, 0.4)
        self.assertNotIn("translating_while_rotating", flags)
        self.assertAlmostEqual(m["path_length_m"], 0.0)

    def test_short_and_missing(self):
        o = sample(first_order(-0.4, -0.6, 1.0, 0.12, 0.05), 0.0, 1.3)       # aborted 0.2 s after the command
        m, flags, notes = A.transition_metrics(o, 1.0, 1.2, -0.4, -0.6)
        self.assertIn("short_window", flags)
        self.assertTrue(math.isnan(m["steady_state_error_radps"]) and math.isnan(m["settling_time_s"]))
        m, flags, _ = A.transition_metrics(o, 5.0, 6.0, -0.4, -0.6)
        self.assertIn("no_data", flags)

    def test_baseline_ignores_previous_transient(self):
        # Previous phase only started moving 0.6 s before this command (zeros in 40 % of the baseline window):
        # median baseline = plateau, no false onset.
        f = lambda t: 0.0 if t < 1.4 else (0.26 if t < 2.0 + 0.6 else 0.4)
        o = sample(f, 0.0, 5.0)
        m, _, _ = A.transition_metrics(o, 2.0, 4.5, 0.2, 0.4)
        self.assertAlmostEqual(m["baseline_radps"], 0.26)
        self.assertTrue(0.6 <= m["response_delay_s"] < 0.6 + DT + 1e-9, m["response_delay_s"])


SEQ = [["BASELINE", 1.0, 0.0], ["A", 2.0, 0.4], ["B", 2.0, -0.4], ["S1", 2.0, 0.0], ["C", 2.0, 0.4],
       ["S2", 2.0, 0.0]]


def write_run(root, name, w_profile, t_stop, outcome, stamp_offset=0.55):
    d = Path(root) / name
    d.mkdir()
    cmds, t = [], 0.0
    for ph, dur, w in SEQ:
        for _ in range(int(round(dur * 5))):
            if t < t_stop:
                cmds.append((round(t, 3), w, ph))
            t += 0.2
    t_last = cmds[-1][0]
    cmds += [(t_last + 0.2 + 0.05 * k, 0.0, "EXIT_STOP") for k in range(10)]
    o = sample(w_profile, 0.0, t_last + 0.7)
    with open(d / "commands.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "ros_time_s", "cmd_linear", "cmd_angular", "phase", "yaw", "rotation_rad", "odom_age_s"])
        for tc, w, ph in cmds:
            wr.writerow([tc, 1000 + tc, 0.0, w, ph, 0.0, 0.0, 0.01])
    with open(d / "odometry.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "ros_time_s", "stamp_s", "x", "y", "yaw", "yaw_unwrapped", "rotation_rad", "v_linear",
                     "v_angular"])
        for i, ts in enumerate(o["t"]):
            arr = ts + 0.02 + 0.1 * (i % 6 == 0)
            wr.writerow([arr, 1000 + arr, ts + stamp_offset, o["x"][i], o["y"][i], o["yaw_raw"][i], o["yaw"][i], 0.0,
                         o["v"][i], o["w"][i]])
    (d / "run_info.json").write_text(json.dumps({"sequence": SEQ, "outcome": outcome,
                                                 "start_pose": {"x": 0.0, "y": 0.0}}))
    return d


def profile(t):
    for t0, w in ((9.0, 0.0), (7.0, 0.4), (5.0, 0.0), (3.0, -0.4), (1.0, 0.4)):
        if t >= t0 + 0.15:
            return w
    return 0.0


class EndToEnd(unittest.TestCase):
    def test_runs_abort_and_summary(self):
        with tempfile.TemporaryDirectory() as root:
            rows1, run1 = A.analyze_run(write_run(root, "angular_run_1", profile, 99, "COMPLETED"))
            self.assertEqual([r["transition_id"] for r in rows1],
                             ["T1_+0.00->+0.40", "T2_+0.40->-0.40", "T3_-0.40->+0.00", "T4_+0.00->+0.40",
                              "T5_+0.40->+0.00"])
            self.assertEqual([r["class"] for r in rows1], ["startup", "reversal", "stop", "startup", "stop"])
            self.assertEqual(rows1[-1]["phase"], "S2+EXIT_STOP")
            for r in rows1:                                     # clock offset and arrival jitter removed
                self.assertTrue(0.17 <= r["response_delay_s"] <= 0.17 + DT + 1e-9, r)
            self.assertAlmostEqual(run1["net_rotation_rad"], 0.4 * 2 - 0.4 * 2 + 0.4 * 2, delta=0.01)
            # An instantaneous reversal inside one 50 ms interval hides up to 2 * 0.4 * 0.05 rad from |dyaw| sums.
            self.assertAlmostEqual(run1["total_rotation_rad"], 0.4 * 6, delta=0.04)
            self.assertEqual(run1["max_abs_cmd_linear_mps"], 0.0)

            # Aborted 0.4 s into phase B (EXIT_STOP is not a transition); later transitions not executed.
            rows2, run2 = A.analyze_run(write_run(root, "angular_run_2", profile, 3.4, "ABORTED: test"))
            self.assertTrue(rows2[1]["executed"])
            self.assertIn("phase_truncated_by_abort", rows2[1]["flags"])
            self.assertAlmostEqual(rows2[1]["t_phase_end_s"], 3.4, places=6)
            self.assertTrue(all(not r["executed"] and "not executed" in r["notes"] for r in rows2[2:]))
            self.assertEqual(run2["n_transitions_executed"], 2)

            summ = A.summarize(rows1 + rows2, [run1, run2])
            # T1 and T4 are the same command pair (0 -> +0.4) but different events: never pooled.
            s1 = next(s for s in summ if s["transition_id"] == "T1_+0.00->+0.40" and s["subset"] == "all"
                      and s["metric"] == "response_delay_s")
            s4 = next(s for s in summ if s["transition_id"] == "T4_+0.00->+0.40" and s["subset"] == "all"
                      and s["metric"] == "response_delay_s")
            self.assertEqual((s1["n_valid_runs"], s1["n_runs"]), (2, 2))
            self.assertEqual((s4["n_valid_runs"], s4["runs"]), (1, "angular_run_1"))
            vals = [rows1[0]["response_delay_s"], rows2[0]["response_delay_s"]]
            self.assertAlmostEqual(s1["mean"], sum(vals) / 2, places=12)
            self.assertAlmostEqual(s1["std"], abs(vals[0] - vals[1]) / math.sqrt(2), places=12)
            self.assertTrue(math.isnan(s4["std"]))
            self.assertFalse(any(s["metric"] == "stopping_time_s" and s["transition_id"].startswith("T2")
                                 for s in summ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
