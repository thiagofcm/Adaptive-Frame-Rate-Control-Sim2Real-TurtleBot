#!/usr/bin/env python3
"""Synthetic-response checks for analyze_combined_motion.py (standard library only).

    python3 RealRobot/motion_dynamic_charac/test_analyze_combined_motion.py
"""

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_combined_motion as A  # noqa: E402

DT = 0.05
PHASE = 0.013


def first_order(prev, new, t_cmd, delay, tau):
    return lambda t: prev if t < t_cmd + delay else new + (prev - new) * math.exp(-(t - t_cmd - delay) / tau)


def sample(v_of_t, w_of_t, t_stop):
    """Native-rate unicycle samples: velocities and the integrated pose (fine steps)."""
    o = {k: [] for k in ("t", "v", "w", "x", "y", "yaw")}
    tk, tau, x, y, th, h = PHASE, 0.0, 0.0, 0.0, 0.0, 1e-4
    while tk < t_stop:
        while tau < tk:
            s = min(h, tk - tau)
            m = tau + s / 2
            x += v_of_t(m) * math.cos(th) * s
            y += v_of_t(m) * math.sin(th) * s
            th += w_of_t(m) * s
            tau += s
        for k, val in zip(("t", "v", "w", "x", "y", "yaw"), (tk, v_of_t(tk), w_of_t(tk), x, y, th)):
            o[k].append(val)
        tk += DT
    return o


class Metrics(unittest.TestCase):
    def test_components_use_their_own_signal_and_delta_tau(self):
        # v reacts after 0.15 s (tau 0.15), w after 0.25 s (tau 0.25): metrics must not be swapped.
        o = sample(first_order(0.0, 0.15, 1.0, 0.15, 0.15), first_order(0.0, 0.4, 1.0, 0.25, 0.25), 6.0)
        m, _, _ = A.transition_metrics(o, 1.0, 5.0, 0.0, 0.15, 0.0, 0.4)
        on_v = 0.15 + 0.15 * -math.log(1 - 0.015 / 0.15)       # 10 % of 0.15 = 0.015 > linear noise 0.01
        on_w = 0.25 + 0.25 * -math.log(1 - 0.05 / 0.4)         # angular noise 0.05 > 10 % of 0.4
        self.assertTrue(on_v <= m["lin_response_delay_s"] < on_v + DT + 1e-9, m["lin_response_delay_s"])
        self.assertTrue(on_w <= m["ang_response_delay_s"] < on_w + DT + 1e-9, m["ang_response_delay_s"])
        self.assertAlmostEqual(m["delta_tau_s"], m["ang_response_delay_s"] - m["lin_response_delay_s"])
        self.assertGreater(m["delta_tau_s"], 0)
        self.assertAlmostEqual(m["lin_rise_time_s"], 0.15 * math.log(9), delta=0.01)
        self.assertAlmostEqual(m["ang_rise_time_s"], 0.25 * math.log(9), delta=0.01)
        self.assertAlmostEqual(m["lin_tracking_ratio"], 1.0, places=6)
        self.assertAlmostEqual(m["ang_tracking_ratio"], 1.0, places=4)          # exp. tail ~4e-6
        self.assertAlmostEqual(m["ang_heading_change_rad"], 0.4 * (4.0 - 0.25 - 0.25), delta=0.01)
        self.assertAlmostEqual(m["ang_cmd_heading_change_rad"], 1.6)

    def test_unchanged_component(self):
        # v stays at 0.15 while w steps 0.3 -> 0.5: no linear delay/rise/overshoot, but linear tracking is kept.
        o = sample(lambda t: 0.15 * 1.02, first_order(0.3, 0.5, 1.0, 0.15, 0.05), 5.0)
        m, flags, _ = A.transition_metrics(o, 1.0, 4.0, 0.15, 0.15, 0.3, 0.5)
        self.assertIn("lin:unchanged_command", flags)
        for k in ("lin_response_delay_s", "lin_rise_time_s", "lin_settling_time_s", "lin_overshoot_pct", "delta_tau_s"):
            self.assertTrue(math.isnan(m[k]), k)
        self.assertAlmostEqual(m["lin_tracking_ratio"], 1.02, places=6)
        self.assertFalse(math.isnan(m["ang_response_delay_s"]))

    def test_stop_distance_and_angle(self):
        t_cmd, D, a_v, a_w = 3.0, 0.1, 0.75, 2.5           # both stop 0.2 s after the onset
        v = lambda t: 0.15 if t < t_cmd + D else max(0.0, 0.15 - a_v * (t - t_cmd - D))
        w = lambda t: -0.5 if t < t_cmd + D else min(0.0, -0.5 + a_w * (t - t_cmd - D))
        o = sample(v, w, 7.0)
        m, _, _ = A.transition_metrics(o, t_cmd, math.inf, 0.15, 0.0, -0.5, 0.0)
        tv = next(t for t, x in zip(o["t"], o["v"]) if t >= t_cmd and abs(x) <= 0.01)
        tw = next(t for t, x in zip(o["t"], o["w"]) if t >= t_cmd and abs(x) <= 0.02)
        self.assertAlmostEqual(m["lin_stopping_time_s"], tv - t_cmd, places=9)
        self.assertAlmostEqual(m["ang_stopping_time_s"], tw - t_cmd, places=9)
        s = min(tw - t_cmd - D, 0.2)
        self.assertAlmostEqual(m["ang_stopping_angle_rad"], -(0.5 * D + 0.5 * s - a_w * s * s / 2), places=5)
        s = min(tv - t_cmd - D, 0.2)
        # path length of the arc = distance along it (path_length uses chords between 50 ms samples)
        self.assertAlmostEqual(m["lin_stopping_distance_m"], 0.15 * D + 0.15 * s - a_v * s * s / 2, delta=2e-5)
        self.assertTrue(math.isnan(m["lin_tracking_ratio"]))


SEQ = [["C1", 2.0, 0.0, 0.0], ["C2", 2.0, 0.15, 0.3], ["C3", 2.0, 0.0, 0.0], ["C4", 2.0, 0.15, -0.3],
       ["C5", 2.0, 0.0, 0.0]]


def write_run(root, name, t_stop, outcome):
    d = Path(root) / name
    d.mkdir()
    cmds, t = [], 0.0
    for ph, dur, v, w in SEQ:
        for _ in range(int(round(dur * 5))):
            if t < t_stop:
                cmds.append((round(t, 3), v, w, ph))
            t += 0.2
    cmds += [(cmds[-1][0] + 0.2 + 0.05 * k, 0.0, 0.0, "EXIT_STOP") for k in range(10)]

    def prof(i):
        def f(t):
            for t0, val in ((8.0, 0.0), (6.0, SEQ[3][i]), (4.0, 0.0), (2.0, SEQ[1][i])):
                if t >= t0 + 0.2:
                    return val
            return 0.0
        return f
    o = sample(prof(2), prof(3), cmds[-1][0] + 0.5)
    with open(d / "commands.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "ros_time_s", "cmd_linear", "cmd_angular", "phase"])
        for tc, v, w, ph in cmds:
            wr.writerow([tc, 1000 + tc, v, w, ph])
    with open(d / "odometry.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "ros_time_s", "stamp_s", "x", "y", "yaw", "v_linear", "v_angular"])
        for i, ts in enumerate(o["t"]):
            arr = ts + 0.02 + 0.1 * (i % 7 == 0)
            wr.writerow([arr, 1000 + arr, ts + 0.55, o["x"][i], o["y"][i], A.wrap(o["yaw"][i]), o["v"][i], o["w"][i]])
    (d / "run_info.json").write_text(json.dumps({
        "mode": "test", "sequence": SEQ, "outcome": outcome, "start_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "ideal_phase_ends_start_frame": [{"x": 0.0, "y": 0.0, "theta_rad": 0.0}]}))
    return d


class EndToEnd(unittest.TestCase):
    def test_runs_abort_and_summary(self):
        with tempfile.TemporaryDirectory() as root:
            rows1, run1, _ = A.analyze_run(write_run(root, "combined_run_1", 99, "COMPLETED"))
            self.assertEqual([r["transition_id"] for r in rows1],
                             ["T1_(0.00,+0.00)->(0.15,+0.30)", "T2_(0.15,+0.30)->(0.00,+0.00)",
                              "T3_(0.00,+0.00)->(0.15,-0.30)", "T4_(0.15,-0.30)->(0.00,+0.00)"])
            for r in rows1:                                     # same 0.2 s dead time on both: delta_tau = 0
                self.assertTrue(0.22 <= r["lin_response_delay_s"] <= 0.22 + DT + 1e-9, r["lin_response_delay_s"])
                self.assertAlmostEqual(r["delta_tau_s"], 0.0)
            self.assertEqual(rows1[-1]["phase"], "C5+EXIT_STOP")
            self.assertAlmostEqual(run1["final_heading_change_deg"], 0.0, delta=0.5)

            rows2, run2, _ = A.analyze_run(write_run(root, "combined_run_2", 4.6, "ABORTED: test"))
            self.assertIn("phase_truncated_by_abort", rows2[1]["flags"])
            self.assertFalse(rows2[2]["executed"] or rows2[3]["executed"])
            self.assertTrue(math.isnan(run2["final_position_error_m"]))   # no ideal comparison when aborted

            summ = A.summarize(rows1 + rows2, [run1, run2])
            g = lambda tid, comp, met: next(s for s in summ if s["transition_id"].startswith(tid)
                                            and s["component"] == comp and s["metric"] == met)
            s = g("T1", "linear", "response_delay_s")
            self.assertEqual((s["n_valid_runs"], s["n_runs"]), (2, 2))
            self.assertEqual(g("T3", "pair", "delta_tau_s")["n_valid_runs"], 1)
            self.assertEqual((g("T2", "angular", "stopping_time_s")["cmd_angular_radps"]), 0.0)
            self.assertFalse(any(x["metric"] == "stopping_time_s" and x["transition_id"].startswith("T1")
                                 for x in summ))
            self.assertFalse(any(x["metric"] == "tracking_ratio" and x["transition_id"].startswith("T2")
                                 for x in summ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
