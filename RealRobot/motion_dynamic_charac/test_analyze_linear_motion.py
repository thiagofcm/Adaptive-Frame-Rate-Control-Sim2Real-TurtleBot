#!/usr/bin/env python3
"""Synthetic-response checks for analyze_linear_motion.py (standard library only).

    python3 RealRobot/motion_dynamic_charac/test_analyze_linear_motion.py
"""

import csv
import json
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_linear_motion as A  # noqa: E402

DT = 0.05          # 20 Hz odometry
PHASE = 0.013      # samples are not aligned with the command time


def sample(v_of_t, t_start, t_stop, x0=0.0):
    """Native-rate samples of a velocity profile; position by exact integration (fine Euler steps)."""
    t, v, x, y = [], [], [], []
    tk, pos, h = t_start + PHASE, x0, 1e-4
    tau = t_start
    while tk < t_stop:
        while tau < tk:                                # integrate position up to tk
            step = min(h, tk - tau)
            pos += v_of_t(tau + step / 2) * step
            tau += step
        t.append(tk); v.append(v_of_t(tk)); x.append(pos); y.append(0.0)
        tk += DT
    return {"t": t, "v": v, "x": x, "y": y}


def first_order(v_prev, v_new, t_cmd, delay, tau):
    return lambda t: v_prev if t < t_cmd + delay else v_new + (v_prev - v_new) * math.exp(-(t - t_cmd - delay) / tau)


def second_order(v_prev, v_new, t_cmd, delay, zeta, wn):
    wd = wn * math.sqrt(1 - zeta ** 2)

    def f(t):
        s = t - t_cmd - delay
        if s < 0:
            return v_prev
        r = 1 - math.exp(-zeta * wn * s) * (math.cos(wd * s) + zeta / math.sqrt(1 - zeta ** 2) * math.sin(wd * s))
        return v_prev + (v_new - v_prev) * r
    return f


class Metrics(unittest.TestCase):
    def test_acceleration_first_order(self):
        t_cmd, t_end, D, tau = 2.0, 6.0, 0.15, 0.2
        o = sample(first_order(0.10, 0.15, t_cmd, D, tau), 0.0, t_end + 1.0)
        m, flags, notes = A.transition_metrics(o, t_cmd, t_end, 0.10, 0.15)
        onset = D + tau * -math.log(1 - 0.01 / 0.05)          # dev reaches the 0.01 m/s detection threshold
        self.assertTrue(onset <= m["response_delay_s"] < onset + DT + 1e-9, m["response_delay_s"])
        self.assertAlmostEqual(m["rise_time_s"], tau * math.log(9), delta=0.01)   # 10-90 % of a first-order lag
        settle = D + tau * math.log(0.05 / A.SETTLE_TOL_MPS)
        self.assertTrue(settle <= m["settling_time_s"] < settle + DT + 1e-9, m["settling_time_s"])
        self.assertAlmostEqual(m["steady_state_error_mps"], 0.0, places=6)
        self.assertEqual(m["overshoot_pct"], 0.0)
        self.assertTrue(math.isnan(m["stopping_time_s"]))

    def test_deceleration_second_order_overshoot(self):
        t_cmd, t_end, zeta, wn = 1.0, 6.0, 0.5, 10.0
        o = sample(second_order(0.20, 0.10, t_cmd, 0.12, zeta, wn), 0.0, t_end + 1.0)
        m, _, _ = A.transition_metrics(o, t_cmd, t_end, 0.20, 0.10)
        expected = 100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))     # 16.3 %
        self.assertAlmostEqual(m["overshoot_pct"], expected, delta=1.5)
        self.assertLess(m["peak_mps"], 0.10)                     # undershoot below the new command
        self.assertTrue(0.12 < m["response_delay_s"] < 0.12 + 0.1)
        self.assertAlmostEqual(m["steady_state_error_mps"], 0.0, places=4)
        # 10-90 % rise of this second-order step, from the analytic curve at fine resolution
        f = second_order(0.20, 0.10, 0.0, 0.0, zeta, wn)
        ts = [i * 1e-5 for i in range(100000)]
        t10 = next(t for t in ts if f(t) <= 0.19)
        t90 = next(t for t in ts if f(t) <= 0.11)
        self.assertAlmostEqual(m["rise_time_s"], t90 - t10, delta=0.01)

    def test_steady_state_bias_and_sign(self):
        o = sample(first_order(0.15, 0.20, 1.0, 0.1, 0.05), 0.0, 6.0)
        o["v"] = [vi - 0.003 if ti > 1.5 else vi for ti, vi in zip(o["t"], o["v"])]     # robot 3 mm/s slow
        m, _, _ = A.transition_metrics(o, 1.0, 5.0, 0.15, 0.20)
        self.assertAlmostEqual(m["steady_state_error_mps"], 0.003, places=6)        # e = v_cmd - v_meas > 0
        self.assertAlmostEqual(m["abs_steady_state_error_mps"], 0.003, places=6)

    def test_stop_time_and_distance(self):
        t_cmd, D, a = 3.0, 0.1, 0.5                              # 0.10 m/s, then constant deceleration
        f = lambda t: 0.10 if t < t_cmd + D else max(0.0, 0.10 - a * (t - t_cmd - D))
        o = sample(f, 0.0, 7.0)
        m, _, _ = A.transition_metrics(o, t_cmd, math.inf, 0.10, 0.0)
        t_stop = next(t for t, v in zip(o["t"], o["v"]) if t >= t_cmd and abs(v) <= A.STOP_MPS)
        self.assertAlmostEqual(m["stopping_time_s"], t_stop - t_cmd, places=9)
        s = min(t_stop - t_cmd - D, 0.10 / a)                    # analytic distance from t_cmd to t_stop
        dist = 0.10 * D + 0.10 * s - a * s * s / 2
        self.assertAlmostEqual(m["stopping_distance_m"], dist, places=6)
        self.assertAlmostEqual(dist, 0.02, delta=0.001)          # nearly the full 0.02 m coast

    def test_stop_never_reached(self):
        f = lambda t: 0.10 if t < 1.1 else max(0.0, 0.10 - 0.05 * (t - 1.1))    # 2 s to stop
        o = sample(f, 0.0, 2.0)                                   # recording ends while still moving
        m, _, notes = A.transition_metrics(o, 1.0, math.inf, 0.10, 0.0)
        self.assertTrue(math.isnan(m["stopping_time_s"]) and math.isnan(m["stopping_distance_m"]))
        self.assertTrue(any(n.startswith("stopping") for n in notes))

    def test_short_phase(self):
        o = sample(first_order(0.10, 0.15, 1.0, 0.15, 0.05), 0.0, 3.0)
        m, _, notes = A.transition_metrics(o, 1.0, 1.6, 0.10, 0.15)    # 0.6 s phase
        self.assertFalse(math.isnan(m["response_delay_s"]))
        self.assertTrue(math.isnan(m["steady_state_error_mps"]))
        self.assertTrue(any("phase too short" in n for n in notes))
        m, _, notes = A.transition_metrics(o, 1.0, 1.25, 0.10, 0.15)   # ends before settling can be held
        self.assertTrue(math.isnan(m["settling_time_s"]))

    def test_next_phase_excluded(self):
        # Response only after the next command: nothing may be detected inside this window.
        o = sample(first_order(0.10, 0.15, 1.0, 0.8, 0.05), 0.0, 4.0)
        m, flags, _ = A.transition_metrics(o, 1.0, 1.5, 0.10, 0.15)
        self.assertIn("no_response", flags)
        self.assertTrue(all(math.isnan(m[k]) for k in ("response_delay_s", "rise_time_s", "overshoot_pct")))

    def test_missing_data(self):
        o = sample(lambda t: 0.10, 0.0, 1.0)
        m, flags, _ = A.transition_metrics(o, 2.0, 3.0, 0.10, 0.15)    # no samples in the window
        self.assertIn("no_data", flags)
        self.assertTrue(all(math.isnan(m[k]) for k, _ in A.TRANSITION_METRICS))

    def test_noise_not_detected(self):
        o = sample(lambda t: 0.10 + 0.008 * math.sin(37 * t), 0.0, 5.0)   # +/-0.008 m/s, below NOISE_MPS
        m, flags, _ = A.transition_metrics(o, 1.0, 4.0, 0.10, 0.15)
        self.assertIn("no_response", flags)


def write_run(root, name, cmd_rows, odom, outcome, stamp_offset, jitter):
    d = Path(root) / name
    d.mkdir()
    with open(d / "commands.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "ros_time_s", "cmd_linear", "cmd_angular", "phase", "forward_m", "odom_age_s"])
        for t, v, ph in cmd_rows:
            w.writerow([t, 1000 + t, v, 0.0, ph, 0.0, 0.01])
    with open(d / "odometry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "ros_time_s", "stamp_s", "x", "y", "yaw", "v_linear", "v_angular", "forward_m"])
        for i, (t, v, x) in enumerate(zip(odom["t"], odom["v"], odom["x"])):
            arrival = t + 0.02 + jitter(i)                      # min latency 0.02 s + jitter >= 0
            w.writerow([arrival, 1000 + arrival, t + stamp_offset, x, 0.0, 0.0, v, 0.0, x])
    sched = [["A", 0.0, 0.10], ["B", 0.4, 0.15], ["STOP", 0.8, 0.0]]
    (d / "run_info.json").write_text(json.dumps({"schedule": sched, "outcome": outcome,
                                                 "start_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}}))
    return d


class EndToEnd(unittest.TestCase):
    def test_alignment_sequence_abort_and_summary(self):
        def profile(t):
            if t < 1.0 + 0.3:
                return 0.0
            if t < 5.0 + 0.15:
                return 0.10
            if t < 8.0 + 0.15:
                return 0.15
            return 0.0
        cmds = [(round(0.2 * k, 3), 0.0, "PRE_ROLL") for k in range(5)]
        cmds += [(round(0.2 * k, 3), 0.10, "A") for k in range(5, 25)]
        cmds += [(round(0.2 * k, 3), 0.15, "B") for k in range(25, 40)]
        cmds += [(8.0, 0.0, "STOP")] + [(round(0.2 * k, 3), 0.0, "SETTLE") for k in range(41, 55)]
        cmds += [(11.0 + 0.05 * k, 0.0, "EXIT_STOP") for k in range(10)]
        odom = sample(profile, 0.0, 11.5)
        jitter = lambda i: 0.15 * ((i * 7) % 5 == 0)            # bursts of late arrivals
        with tempfile.TemporaryDirectory() as root:
            r1 = write_run(root, "run_1", cmds, odom, "COMPLETED", 0.55, jitter)
            rows, run_row = A.analyze_run(r1)
            self.assertEqual([r["transition_id"] for r in rows], ["T1_0.00->0.10", "T2_0.10->0.15", "T3_0.15->0.00"])
            self.assertEqual(rows[2]["phase"], "STOP+SETTLE+EXIT_STOP")
            # The robot clock (+0.55 s) and arrival jitter are removed: delays match the profile's dead time
            # plus the 0.02 s minimum latency, to within one sample.
            for r, dead in zip(rows, (0.3, 0.15, 0.15)):
                self.assertTrue(dead + 0.02 <= r["response_delay_s"] <= dead + 0.02 + DT + 1e-9,
                                (r["transition_id"], r["response_delay_s"]))
            self.assertAlmostEqual(run_row["clock_offset_s"], 0.55 - 0.02, places=9)
            self.assertAlmostEqual(run_row["net_displacement_m"], odom["x"][-1], places=9)
            self.assertEqual(run_row["n_valid_transitions"], 3)

            # Aborted during phase B: T2 truncated at EXIT_STOP, T3 reported as not executed.
            ab = [c for c in cmds if c[0] < 6.0] + [(6.0 + 0.05 * k, 0.0, "EXIT_STOP") for k in range(10)]
            ab_odom = {k: [v for t, v in zip(odom["t"], odom[k]) if t < 6.6] for k in odom}
            r2 = write_run(root, "run_2", ab, ab_odom, "ABORTED: test", 0.55, jitter)
            rows2, run_row2 = A.analyze_run(r2)
            self.assertIn("phase_truncated_by_abort", rows2[1]["flags"])
            self.assertAlmostEqual(rows2[1]["t_phase_end_s"], 6.0)
            self.assertFalse(rows2[2]["executed"])
            self.assertTrue(math.isnan(rows2[2]["response_delay_s"]))
            self.assertIn("not executed", rows2[2]["notes"])
            self.assertFalse(run_row2["completed"])

            # Aggregation: per-run values, sample sd, NaN excluded and counted.
            summ = A.summarize(rows + rows2, [run_row, run_row2])
            s = next(x for x in summ if x["transition_id"] == "T2_0.10->0.15" and x["metric"] == "response_delay_s")
            vals = [rows[1]["response_delay_s"], rows2[1]["response_delay_s"]]
            self.assertAlmostEqual(s["mean"], sum(vals) / 2, places=12)
            self.assertAlmostEqual(s["std"], abs(vals[0] - vals[1]) / math.sqrt(2), places=12)
            s3 = next(x for x in summ if x["transition_id"] == "T3_0.15->0.00" and x["metric"] == "stopping_time_s")
            self.assertEqual((s3["n_valid_runs"], s3["n_runs"]), (1, 2))
            self.assertTrue(math.isnan(s3["std"]))                # sd undefined for a single run
            self.assertFalse(any(x["metric"].startswith("stopping") and x["transition_id"].startswith("T1")
                                 for x in summ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
