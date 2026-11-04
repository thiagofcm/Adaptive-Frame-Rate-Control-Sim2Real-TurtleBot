#!/usr/bin/env python3
"""Synthetic checks for analyze_td3_navigation.py: known dynamics seen through a recorder like the navigation one
(20 Hz odometry with transport latency, 10 Hz rows holding the latest pose). Standard library only.

    python3 RealRobot/motion_dynamic_charac/test_analyze_td3_navigation.py
"""

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_td3_navigation as A  # noqa: E402


def synthetic_episode(theta, tau, k_w, wheel_limit=None, latency=(0.01, 0.03), duration=60.0, seed=1):
    """TD3-like random commands every 0.2 s; true response = wheel limit (scale) -> dead time -> first-order lag
    -> gain k_w on w; unicycle pose integrated at 1 kHz; 20 Hz odometry delivered after a random latency; rows
    every 0.1 s hold the latest delivered pose."""
    rng = random.Random(seed)
    ticks = [0.2 * k for k in range(int(duration / 0.2))]
    cmd_v = [rng.uniform(0.1, 0.22) for _ in ticks]
    cmd_w = [rng.uniform(-1.6, 1.6) for _ in ticks]
    ep = {"ticks": ticks, "cmd_v": cmd_v, "cmd_w": cmd_w}
    dt = 0.001
    x = y = th = fv = fw = 0.0
    a = 1 - math.exp(-dt / tau) if tau > 0 else 1.0
    odom = []                                     # (arrival time, x, y, yaw)
    for n in range(int(duration / dt)):
        t = n * dt
        v, w = A.command_at(ep, "v", t - theta), A.command_at(ep, "w", t - theta)
        if wheel_limit:
            v, w = A.saturate(v, w, "scale", wheel_limit)
        fv += (v - fv) * a
        fw += (w - fw) * a
        x += fv * math.cos(th) * dt
        y += fv * math.sin(th) * dt
        th += k_w * fw * dt
        if n % 50 == 0:                           # 20 Hz odometry
            odom.append((t + rng.uniform(*latency), x, y, A.wrap(th)))
    odom.sort()
    T, X, Y, YAW, j = [], [], [], [], -1
    for i in range(1, int(duration / 0.1) - 5):
        t = 0.1 * i + 0.0003
        while j + 1 < len(odom) and odom[j + 1][0] <= t:
            j += 1
        T.append(t); X.append(odom[j][1]); Y.append(odom[j][2]); YAW.append(odom[j][3])
    ep.update({"name": "synthetic", "kind": "td3", "T": T, "x": X, "y": Y, "yaw": A.unwrap(YAW), "t_end": T[-1]})
    return ep


class Estimators(unittest.TestCase):
    def test_fopdt_fit_recovers_dynamics(self):
        theta, tau = 0.15, 0.05
        ep = synthetic_episode(theta, tau, k_w=1.0)
        incs = A.increments(ep)
        for comp in ("v", "w"):
            f = A.fit_fopdt(ep, comp, incs)
            # the recorder adds the mean pose age (~25 ms) + mean latency (20 ms) to the apparent dead time
            self.assertAlmostEqual(f["theta_s"] + f["tau_s"], theta + tau + 0.045, delta=0.03, msg=comp)
            self.assertAlmostEqual(f["tau_s"], tau, delta=0.03, msg=comp)
            self.assertAlmostEqual(f["K"], 1.0, delta=0.03, msg=comp)

    def test_gain_detected(self):
        ep = synthetic_episode(0.15, 0.03, k_w=1.25, seed=2)
        f = A.fit_fopdt(ep, "w", A.increments(ep))
        self.assertAlmostEqual(f["K"], 1.25, delta=0.04)

    def test_wheel_limit_recovered(self):
        ep = synthetic_episode(0.15, 0.03, k_w=1.0, wheel_limit=0.25, seed=3)
        incs = A.increments(ep)
        unsat = [d for d in incs if A.demand(ep, d[0] - 0.8, d[1]) <= 0.24]
        fits = {c: A.fit_fopdt(ep, c, unsat) for c in ("v", "w")}
        A.SAT_GRID = [0.21 + 0.01 * i for i in range(9)]
        scores = A.fit_saturation(ep, incs, fits)
        (model, wmax), _ = min(scores.items(), key=lambda kv: kv[1])
        self.assertEqual(model, "scale")
        self.assertAlmostEqual(wmax, 0.25, delta=0.011)
        self.assertLess(scores[(model, wmax)], 0.5 * scores[("none", next(k[1] for k in scores if k[0] == "none"))])


if __name__ == "__main__":
    unittest.main(verbosity=2)
