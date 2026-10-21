import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import moving_obstacle as mo

# Reference values printed by the Gazebo header itself (random_ping_pong.hh compiled
# with g++, OB1 configuration, seed 1000): s0, direction, then the first 9 targets.
GAZEBO_SEED_1000 = {
    1: (0.32038657214493127, 1, [0.80241147710885985, 0.22040729256432265, 0.98972579118075332,
                                 0.07899370527965538, 0.80928333244972506, 0.10890929576658726,
                                 0.99445405621630911, 0.0069726218176750787, 0.89336866922377234]),
    2: (0.47950208661134808, 1, [0.93458068569341779, 0.16298588859595517, 0.83415646670577703,
                                 0.022324298067347176, 0.88028522279735244, 0.045602974798740009,
                                 0.94643095981669279, 0.152543548144465, 0.87845996856754982]),
    3: (0.80104683342720517, 1, [0.96689134836412249, 0.19510017750396233, 0.80248469243675058,
                                 0.18277575234948268, 0.90856668259104167, 0.040797857075518751,
                                 0.95281054825577272, 0.12428437419822205, 0.76297749536796222]),
}


class SimRobot:
    """Unicycle with first-order velocity lag, a heading bias and 1-tick odometry delay."""

    def __init__(self, x, y, yaw, tau=0.15, yaw_bias=0.0, turn_gain=1.0):
        self.x, self.y, self.yaw = x, y, yaw
        self.v = self.w = 0.0
        self.tau, self.yaw_bias, self.turn_gain = tau, yaw_bias, turn_gain
        self.delayed = (x, y, yaw, 0.0, 0.0)

    def advance(self, v_cmd, w_cmd, dt, substeps=10):
        h = dt / substeps
        for _ in range(substeps):
            self.v += (v_cmd - self.v) * h / self.tau
            self.w += (self.turn_gain * w_cmd - self.w) * h / self.tau
            self.yaw = mo.wrap_angle(self.yaw + (self.w + self.yaw_bias * abs(self.v) / 0.06) * h)
            self.x += self.v * math.cos(self.yaw) * h
            self.y += self.v * math.sin(self.yaw) * h

    def odom(self):
        current, self.delayed = self.delayed, (self.x, self.y, self.yaw, self.v, self.w)
        return current


def simulate(episode, yaw0, duration=70.0, seed=1000, **robot_kwargs):
    plan = mo.GazeboReversalPlan(seed, episode)
    frame = mo.TrackFrame(0.0, 0.0, yaw0, plan.s0, plan.initial_direction)
    controller = mo.ObstacleController(plan, frame, max_duration_s=duration)
    robot = SimRobot(0.0, 0.0, yaw0, **robot_kwargs)
    dt = 1.0 / mo.CONTROL_RATE_HZ
    ticks = []
    t = 0.0
    while controller.state != mo.STOPPED:
        x, y, yaw, v, w = robot.odom()
        v_cmd, w_cmd = controller.step(t, x, y, yaw, v, w)
        ticks.append((t, controller.state, v_cmd, w_cmd, controller.last_track_position["s"],
                      controller.last_track_position["lateral"]))
        robot.advance(v_cmd, w_cmd, dt)
        t += dt
        if t > duration + 5:
            raise AssertionError("controller never stopped")
    return plan, controller, ticks


class GeneratorTest(unittest.TestCase):
    def test_matches_gazebo_header(self):
        for episode, (s0, direction, targets) in GAZEBO_SEED_1000.items():
            plan = mo.GazeboReversalPlan(1000, episode)
            self.assertEqual(plan.s0, s0)
            self.assertEqual(plan.initial_direction, direction)
            self.assertEqual(plan.targets(9), targets)

    def test_targets_alternate_regions_and_do_not_mutate(self):
        plan = mo.GazeboReversalPlan(1000, 7)
        before = plan.targets(10)
        self.assertEqual(plan.targets(10), before)
        self.assertEqual(plan.first_target, before[0])
        direction = plan.initial_direction
        for target in before:
            if direction > 0:
                self.assertTrue(0.75 <= target <= 1.0)
            else:
                self.assertTrue(0.0 <= target <= 0.25)
            direction = -direction

    def test_dry_run_output(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mo.main(["--seed", "1000", "--episode", "1", "--dry-run"])
        text = out.getvalue()
        self.assertIn("s0 = 0.320 m from S (toward E), facing E", text)
        self.assertIn("first reversal target 0.802 m", text)


class TrackFrameTest(unittest.TestCase):
    def test_frame_independent_of_initial_facing(self):
        for facing, direction in (("E", 1), ("S", -1)):
            frame = mo.TrackFrame(1.0, 2.0, 0.3, 0.5, direction)
            # Driving 0.1 m forward along the initial heading moves toward the facing end.
            s, lateral = frame.project(1.0 + 0.1 * math.cos(0.3), 2.0 + 0.1 * math.sin(0.3))
            self.assertAlmostEqual(s, 0.5 + 0.1 * direction, places=12, msg=facing)
            self.assertAlmostEqual(lateral, 0.0, places=12)
            self.assertAlmostEqual(mo.wrap_angle(frame.theta_s - frame.theta_e), math.pi, places=12)
        # Lateral is positive to the left of the S->E direction.
        frame = mo.TrackFrame(0.0, 0.0, 0.0, 0.0, 1)
        self.assertGreater(frame.project(0.0, 0.05)[1], 0.0)

    def test_wraparound_headings(self):
        frame = mo.TrackFrame(0.0, 0.0, math.pi - 0.01, 0.5, -1)   # facing S, heading near +pi
        self.assertAlmostEqual(frame.theta_e, -0.01, places=12)
        self.assertAlmostEqual(frame.theta_s, math.pi - 0.01, places=12)


class ControllerTest(unittest.TestCase):
    def check_run(self, episode, yaw0, overshoot_limit=0.01, max_turn_s=8.0, **robot_kwargs):
        plan, controller, ticks = simulate(episode, yaw0, **robot_kwargs)
        self.assertEqual(controller.end_reason, "max_duration")
        done = [r for r in controller.reversals if "aborted" not in r]
        self.assertGreaterEqual(len(done), 3)
        # Same reversal targets as Gazebo, in order.
        self.assertEqual([r["planned_target_s"] for r in done], plan.targets(len(done)))
        for r in done:
            self.assertLess(abs(r["overshoot_m"]), overshoot_limit)
            self.assertLessEqual(abs(r["final_heading_error_deg"]), math.degrees(mo.SETTLE_TOLERANCE))
            self.assertTrue(2.5 < r["rotation_duration_s"] < max_turn_s, r["rotation_duration_s"])
        # Never translate while stopping, turning or settling; straight-leg corrections stay gentle.
        for _, state, v_cmd, w_cmd, s, lateral in ticks:
            if state in (mo.STOPPING, mo.ROTATE_180, mo.SETTLE, mo.STOPPED):
                self.assertEqual(v_cmd, 0.0)
            if state == mo.MOVE_FORWARD:
                self.assertLessEqual(abs(w_cmd), mo.STRAIGHT_TURN_RATE_MAX + 1e-12)
            self.assertLessEqual(abs(w_cmd), mo.TURN_RATE_MAX + 1e-12)
            self.assertTrue(-0.02 < s < 1.02)
            self.assertLess(abs(lateral), 0.03)
        return controller

    def test_episodes_both_facings_and_wraparound(self):
        for episode, yaw0 in ((1, 0.4), (4, math.pi - 0.02), (5, -math.pi + 0.01), (6, -1.2)):
            with self.subTest(episode=episode, yaw0=yaw0):
                self.check_run(episode, yaw0)

    def test_heading_bias_is_corrected_in_both_directions(self):
        controller = self.check_run(2, 0.0, yaw_bias=0.02)   # ~1.1 deg/s of drift while driving
        self.assertGreaterEqual(controller.reversal_count, 3)

    def test_turn_overshoot_settles_with_retries(self):
        # A sluggish, over-responding drive: turns overshoot and need fine re-turns.
        # Its 0.4 s lag also coasts ~0.06 m/s * 0.4 s = 2.4 cm after a stop command.
        controller = self.check_run(3, 0.0, overshoot_limit=0.03, max_turn_s=12.0,
                                    tau=0.4, turn_gain=1.3)
        self.assertTrue(any(r["turn_retries"] > 0 for r in controller.reversals))

    def test_first_target_already_reached(self):
        plan = mo.GazeboReversalPlan(1000, 4)                 # s0 0.056, first target 0.052
        _, controller, ticks = simulate(4, 0.0, duration=10.0)
        self.assertEqual(ticks[0][1], mo.STOPPING)
        self.assertEqual(controller.reversals[0]["planned_target_s"], plan.first_target)

    def test_turn_timeout(self):
        _, controller, ticks = simulate(3, 0.0, turn_gain=0.0)   # wheels cannot turn
        self.assertEqual(controller.end_reason, "turn_timeout")
        self.assertEqual(controller.reversals[-1]["aborted"], "turn_timeout")
        self.assertEqual(ticks[-1][2:4], (0.0, 0.0))

    def test_safety_limits(self):
        for name, (x, y), reason in (("track", (0.8, 0.0), "track_limit"),
                                     ("lateral", (0.0, 0.11), "lateral_limit")):
            with self.subTest(name):
                plan = mo.GazeboReversalPlan(1000, 1)
                frame = mo.TrackFrame(0.0, 0.0, 0.0, plan.s0, plan.initial_direction)
                controller = mo.ObstacleController(plan, frame)
                self.assertEqual(controller.step(0.1, x, y, 0.0, 0.0, 0.0), (0.0, 0.0))
                self.assertEqual(controller.end_reason, reason)

    def test_max_duration(self):
        _, controller, ticks = simulate(1, 0.0, duration=5.0)
        self.assertEqual(controller.end_reason, "max_duration")
        self.assertAlmostEqual(ticks[-1][0], 5.0, delta=0.06)


class LoggerTest(unittest.TestCase):
    def test_files(self):
        with tempfile.TemporaryDirectory() as root:
            logger = mo.RunLogger(root, 1000, 1)
            plan, controller, _ = simulate(1, 0.0, duration=20.0)
            logger.write_config({"realization": mo.realization_summary(plan),
                                 "parameters": mo.controller_parameters()})
            logger.odom_row({name: 0 for name in mo.ODOM_FIELDS})
            logger.reversals(controller.reversals)
            logger.close()
            run = Path(logger.dir)
            self.assertEqual(run.name[:16], "seed1000_ep0001_")
            config = json.loads((run / "config.json").read_text())
            self.assertEqual(config["realization"]["initial_facing"], "E")
            with (run / "reversals.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(controller.reversals))
            self.assertEqual(float(rows[0]["planned_target_s"]), plan.first_target)
            with (run / "odom.csv").open() as handle:
                self.assertEqual(next(csv.reader(handle)), mo.ODOM_FIELDS)


if __name__ == "__main__":
    unittest.main()
