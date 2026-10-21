#!/usr/bin/env python3
"""Drive a second TurtleBot4 (Robot 2) as the Stage 14 moving obstacle OB1.

Gazebo OB1 (worlds/turtlebot3_drl_stage14/burger.model, plugin
corridor_random_obstacle.cc + random_ping_pong.hh) moves at 0.06 m/s along a
1.0 m straight track, reversing at random targets drawn from the start region
[0, 0.25] m and the end region [0.75, 1.0] m. Its start position s0, first
direction and every reversal target come from a std::mt19937_64 seeded with
std::seed_seq{seed, episode, obstacle_index}. This script reproduces that
generator bit-for-bit, so `--seed 1000 --episode k` replays the OB1 of Gazebo
evaluation episode k (seed 1000 is the held-out Stage 14 evaluation seed).

Physical differences from Gazebo (logged, not compensated):
  * Robot 2 drives forward only and turns 180 degrees in place at each
    reversal point (Gazebo reverses instantly), so it falls behind Gazebo's
    schedule by roughly one turn duration per reversal.
  * Position comes from wheel odometry; reversals have a small overshoot.

Track frame: S is the start point (s = 0), E the end point (s = 1.0 m).
Place Robot 2 on the taped S-E line at s0 (printed by --dry-run), facing E if
its first move is toward E, otherwise facing S. The frame is fixed from the
first odometry pose and the known initial facing, so s always increases
toward E.

Usage:
    python3 moving_obstacle.py --seed 1000 --episode 1 --dry-run
    python3 moving_obstacle.py --namespace robot2 --seed 1000 --episode 1

No synchronization with Robot 1: the motion starts as soon as the reference
pose is captured.
"""

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Stage 14 OB1 parameters (worlds/turtlebot3_drl_stage14/burger.model)
# ---------------------------------------------------------------------------
TRACK_LENGTH_M = 1.0          # |end - start| = |(2.4, 2.0) - (1.4, 2.0)|
SPEED_MPS = 0.06
START_REGION_FRACTION = (0.0, 0.25)
END_REGION_FRACTION = (0.75, 1.0)
OBSTACLE_INDEX = 0            # OB1's <obstacle_index>
DEFAULT_SEED = 1000           # held-out Stage 14 evaluation seed
DEFAULT_MAX_DURATION_S = 70.0  # Stage 14 EPISODE_TIMEOUT_SECONDS

# ---------------------------------------------------------------------------
# Controller parameters (initial values, see the plan)
# ---------------------------------------------------------------------------
CONTROL_RATE_HZ = 20.0
TARGET_TOLERANCE_M = 0.005            # reversal triggered this close to (or past) the target
STOP_SPEED_MPS = 0.01                 # STOPPING ends below this measured speed
STOP_MAX_S = 1.0                      # ... or after this long
TURN_RATE_MAX = 0.8                   # rad/s
TURN_RATE_MIN = 0.15                  # rad/s, overcomes the drive's dead band near the end of a turn
TURN_ACCEL = 1.5                      # rad/s^2, ramp limit on the commanded turn rate
TURN_P_GAIN = 2.0                     # rad/s per rad, within TURN_FINE_ZONE
TURN_FINE_ZONE = 0.4                  # rad, below this the turn uses proportional control
TURN_DONE_TOLERANCE = math.radians(2.0)    # ROTATE_180 -> SETTLE
SETTLE_TOLERANCE = math.radians(3.0)       # heading accepted after settling
SETTLE_RATE = 0.05                    # rad/s, measured turn rate considered settled
SETTLE_TICKS = 3                      # consecutive settled ticks required
SETTLE_MAX_S = 1.5                    # settle check is taken after this long at the latest
MAX_TURN_RETRIES = 3                  # fine re-turns allowed per reversal
TURN_TIMEOUT_S = 15.0                 # about 3x the expected turn time
HEADING_GAIN = 1.0                    # rad/s per rad of heading error (straight legs)
LATERAL_GAIN = 1.0                    # rad of heading offset per m of lateral error
LATERAL_HEADING_MAX = math.radians(5.0)    # cap on that heading offset
STRAIGHT_TURN_RATE_MAX = 0.2          # rad/s, cap on straight-leg corrections

# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------
TRACK_MARGIN_M = 0.05                 # stop if s leaves [-margin, length + margin]
LATERAL_LIMIT_M = 0.10                # stop if |lateral| exceeds this
ODOM_TIMEOUT_S = 0.3                  # stop if odometry is older than this
ODOM_WAIT_S = 5.0                     # wait this long for the first odometry
STOP_REPEATS = 5                      # zero commands published on exit

MOVE_FORWARD, STOPPING, ROTATE_180, SETTLE, STOPPED = (
    "MOVE_FORWARD", "STOPPING", "ROTATE_180", "SETTLE", "STOPPED")


def wrap_angle(angle):
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# ===========================================================================
# Gazebo-compatible generator: std::seed_seq + std::mt19937_64 + RandomPingPong
# (random_ping_pong.hh). Verified bit-identical to the C++ header.
# ===========================================================================
_M32 = 0xFFFFFFFF
_M64 = 0xFFFFFFFFFFFFFFFF


def _seed_seq_generate(values, n):
    """std::seed_seq::generate for 32-bit input values (C++11 [rand.util.seedseq])."""
    s = len(values)
    t = 11 if n >= 623 else 7 if n >= 68 else 5 if n >= 39 else 3 if n >= 7 else (n - 1) // 2
    p = (n - t) // 2
    q = p + t
    m = max(s + 1, n)
    b = [0x8B8B8B8B] * n

    def tmix(x):
        return (x ^ (x >> 27)) & _M32

    for k in range(m):
        r1 = (1664525 * tmix(b[k % n] ^ b[(k + p) % n] ^ b[(k - 1) % n])) & _M32
        if k == 0:
            r2 = (r1 + s) & _M32
        elif k <= s:
            r2 = (r1 + (k % n) + values[k - 1]) & _M32
        else:
            r2 = (r1 + (k % n)) & _M32
        b[(k + p) % n] = (b[(k + p) % n] + r1) & _M32
        b[(k + q) % n] = (b[(k + q) % n] + r2) & _M32
        b[k % n] = r2
    for k in range(m, m + n):
        r3 = (1566083941 * tmix((b[k % n] + b[(k + p) % n] + b[(k - 1) % n]) & _M32)) & _M32
        r4 = (r3 - (k % n)) & _M32
        b[(k + p) % n] ^= r3
        b[(k + q) % n] ^= r4
        b[k % n] = r4
    return b


class MersenneTwister64:
    """std::mt19937_64 seeded from a std::seed_seq of 32-bit values."""

    def __init__(self, seed_values):
        words = _seed_seq_generate(seed_values, 624)
        self.state = [(words[2 * i] | (words[2 * i + 1] << 32)) & _M64 for i in range(312)]
        self.index = 312

    def __call__(self):
        if self.index >= 312:
            mt = self.state
            for k in range(312):
                y = (mt[k] & 0xFFFFFFFF80000000) | (mt[(k + 1) % 312] & 0x7FFFFFFF)
                mt[k] = mt[(k + 156) % 312] ^ (y >> 1) ^ (0xB5026F5AA96619E9 if y & 1 else 0)
            self.index = 0
        y = self.state[self.index]
        self.index += 1
        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71D67FFFEDA60000
        y ^= (y << 37) & 0xFFF7EEE000000000
        y ^= y >> 43
        return y & _M64


class GazeboReversalPlan:
    """OB1's realization for (seed, episode, obstacle_index), as random_ping_pong.hh draws it."""

    def __init__(self, seed, episode, obstacle_index=OBSTACLE_INDEX, length=TRACK_LENGTH_M,
                 start_region=START_REGION_FRACTION, end_region=END_REGION_FRACTION):
        self.seed, self.episode, self.obstacle_index = int(seed), int(episode), int(obstacle_index)
        self.length = float(length)
        self.start_lo, self.start_hi = start_region[0] * length, start_region[1] * length
        self.end_lo, self.end_hi = end_region[0] * length, end_region[1] * length

        def lo(v):
            return v & _M32

        def hi(v):
            return (v >> 32) & _M32

        self._rng = MersenneTwister64([lo(self.seed), hi(self.seed), lo(self.episode),
                                       hi(self.episode), lo(self.obstacle_index),
                                       hi(self.obstacle_index)])
        self.s0 = self._uniform(0.0, self.length)
        self.initial_direction = 1 if self._unit() < 0.5 else -1
        self._s, self._direction = self.s0, self.initial_direction
        self.first_target = self._sample_target()
        self._target = self.first_target

    def _unit(self):
        return (self._rng() >> 11) * (1.0 / 9007199254740992.0)

    def _uniform(self, lo, hi):
        return lo + (hi - lo) * self._unit()

    def _sample_target(self):
        if self._direction > 0:
            return self._uniform(max(self.end_lo, self._s), self.end_hi)
        return self._uniform(self.start_lo, min(self.start_hi, self._s))

    def next_target(self):
        """Reverse at the current target and draw the next one, exactly as Advance() does.
        The draw depends only on the planned target, never on the measured position."""
        self._s = self._target
        self._direction = -self._direction
        self._target = self._sample_target()
        return self._target

    def targets(self, count):
        """The first `count` reversal targets, without touching this plan's state."""
        copy = GazeboReversalPlan(self.seed, self.episode, self.obstacle_index, self.length,
                                  (self.start_lo / self.length, self.start_hi / self.length),
                                  (self.end_lo / self.length, self.end_hi / self.length))
        return [copy.first_target] + [copy.next_target() for _ in range(count - 1)]


# ===========================================================================
# Track frame
# ===========================================================================
class TrackFrame:
    """Fixed track coordinates from the first odometry pose and the initial facing.

    s increases toward E whatever the initial facing; lateral is positive to the
    left of the S->E direction. theta_e / theta_s are the two travel headings.
    """

    def __init__(self, x0, y0, yaw0, s0, initial_direction):
        self.x0, self.y0, self.s0 = float(x0), float(y0), float(s0)
        self.theta_e = wrap_angle(yaw0 if initial_direction > 0 else yaw0 + math.pi)
        self.theta_s = wrap_angle(self.theta_e + math.pi)
        self.ux, self.uy = math.cos(self.theta_e), math.sin(self.theta_e)

    def heading_for(self, direction):
        return self.theta_e if direction > 0 else self.theta_s

    def project(self, x, y):
        dx, dy = x - self.x0, y - self.y0
        return self.s0 + dx * self.ux + dy * self.uy, -dx * self.uy + dy * self.ux


# ===========================================================================
# Controller (no ROS): one step per control tick
# ===========================================================================
class ObstacleController:
    """MOVE_FORWARD -> STOPPING -> ROTATE_180 -> SETTLE -> MOVE_FORWARD ... -> STOPPED."""

    def __init__(self, plan, frame, max_duration_s=DEFAULT_MAX_DURATION_S, dt=1.0 / CONTROL_RATE_HZ):
        self.plan, self.frame, self.dt = plan, frame, dt
        self.max_duration_s = float(max_duration_s)
        self.state = MOVE_FORWARD
        self.direction = plan.initial_direction
        self.target = plan.first_target
        self.target_heading = frame.heading_for(self.direction)
        self.turn_sign = 1                  # first turn counter-clockwise, then alternate
        self.end_reason = None
        self.reversal_count = 0
        self.reversal = None                # record of the reversal in progress
        self.reversals = []                 # completed reversal records
        self._state_since = 0.0
        self._turn_cmd = 0.0
        self._settled_ticks = 0
        self._turn_retries = 0
        self._last = {}

    # -- helpers ---------------------------------------------------------
    def _enter(self, state, t):
        self.state, self._state_since = state, t

    def _stop(self, reason, t):
        if self.state != STOPPED:
            self.end_reason = reason
            if self.reversal is not None:       # keep the interrupted reversal in the log
                self.reversal["aborted"] = reason
                self.reversals.append(self.reversal)
                self.reversal = None
            self._enter(STOPPED, t)

    def heading_error(self, yaw):
        return wrap_angle(self.target_heading - yaw)

    def _straight_command(self, yaw, lateral):
        # Heading first: steer toward the travel heading, offset by at most
        # LATERAL_HEADING_MAX to drift back onto the axis. Lateral is measured
        # to the left of the S->E direction, so flip it for travel toward S.
        lateral_travel = self.direction * lateral
        offset = max(-LATERAL_HEADING_MAX, min(LATERAL_HEADING_MAX, LATERAL_GAIN * lateral_travel))
        error = wrap_angle(self.target_heading - offset - yaw)
        return max(-STRAIGHT_TURN_RATE_MAX, min(STRAIGHT_TURN_RATE_MAX, HEADING_GAIN * error))

    def _turn_command(self, yaw):
        error = self.heading_error(yaw)
        if abs(error) > TURN_FINE_ZONE:
            # Far from the target heading the sign of a wrapped error near +-pi is
            # ambiguous, so keep the direction chosen at the start of the turn.
            sign, magnitude = self.turn_sign, TURN_RATE_MAX
        else:
            sign = 1 if error > 0 else -1
            magnitude = max(TURN_RATE_MIN, min(TURN_RATE_MAX, TURN_P_GAIN * abs(error)))
        # Ramp limit on the commanded rate (also limits sign changes).
        desired = sign * magnitude
        step = TURN_ACCEL * self.dt
        self._turn_cmd = max(self._turn_cmd - step, min(self._turn_cmd + step, desired))
        return self._turn_cmd

    # -- one tick --------------------------------------------------------
    def step(self, t, x, y, yaw, v_meas, w_meas):
        """Advance one control tick. t is seconds since start. Returns (v_cmd, w_cmd)."""
        s, lateral = self.frame.project(x, y)
        self._last = {"s": s, "lateral": lateral}

        if self.state != STOPPED:
            if t >= self.max_duration_s:
                self._stop("max_duration", t)
            elif not -TRACK_MARGIN_M <= s <= self.plan.length + TRACK_MARGIN_M:
                self._stop("track_limit", t)
            elif abs(lateral) > LATERAL_LIMIT_M:
                self._stop("lateral_limit", t)

        v_cmd = w_cmd = 0.0
        if self.state == MOVE_FORWARD:
            if self.direction * (self.target - s) <= TARGET_TOLERANCE_M:
                self.reversal = {"index": self.reversal_count + 1, "planned_target_s": self.target,
                                 "direction_before": self.direction, "trigger_time_s": t,
                                 "trigger_s": s}
                self._enter(STOPPING, t)
            else:
                v_cmd, w_cmd = SPEED_MPS, self._straight_command(yaw, lateral)
        elif self.state == STOPPING:
            if abs(v_meas) < STOP_SPEED_MPS or t - self._state_since >= STOP_MAX_S:
                self.reversal.update({"stopped_time_s": t, "stopped_s": s,
                                      "overshoot_m": self.direction * (s - self.target),
                                      "rotation_start_s": t, "turn_sign": self.turn_sign})
                self.direction = -self.direction
                self.target_heading = self.frame.heading_for(self.direction)
                self._turn_cmd, self._turn_retries = 0.0, 0
                self._enter(ROTATE_180, t)
        if self.state == ROTATE_180:
            if t - self.reversal["rotation_start_s"] > TURN_TIMEOUT_S:
                self._stop("turn_timeout", t)
            elif abs(self.heading_error(yaw)) < TURN_DONE_TOLERANCE:
                self._turn_cmd, self._settled_ticks = 0.0, 0
                self._enter(SETTLE, t)
            else:
                w_cmd = self._turn_command(yaw)
        elif self.state == SETTLE:
            # Zero commands, then verify the final heading before translating.
            self._settled_ticks = self._settled_ticks + 1 if abs(w_meas) < SETTLE_RATE else 0
            if self._settled_ticks >= SETTLE_TICKS or t - self._state_since >= SETTLE_MAX_S:
                error = self.heading_error(yaw)
                if abs(error) <= SETTLE_TOLERANCE:
                    self._finish_reversal(t, error)
                elif self._turn_retries >= MAX_TURN_RETRIES:
                    self._stop("turn_not_settled", t)
                else:
                    self._turn_retries += 1
                    self._turn_cmd = 0.0
                    self._enter(ROTATE_180, t)
        return v_cmd, w_cmd

    def _finish_reversal(self, t, heading_error):
        self.reversal.update({"rotation_end_s": t,
                              "rotation_duration_s": t - self.reversal["rotation_start_s"],
                              "final_heading_error_deg": math.degrees(heading_error),
                              "turn_retries": self._turn_retries,
                              "next_target_s": None})
        self.target = self.plan.next_target()
        self.reversal["next_target_s"] = self.target
        self.reversals.append(self.reversal)
        self.reversal = None
        self.reversal_count += 1
        self.turn_sign = -self.turn_sign
        self._enter(MOVE_FORWARD, t)

    @property
    def last_track_position(self):
        return self._last


# ===========================================================================
# Logging
# ===========================================================================
ODOM_FIELDS = ["t_s", "state", "x", "y", "yaw", "s", "lateral", "target_s", "direction",
               "target_heading", "heading_error", "cmd_linear", "cmd_angular",
               "meas_linear", "meas_angular"]
REVERSAL_FIELDS = ["index", "planned_target_s", "direction_before", "trigger_time_s", "trigger_s",
                   "stopped_time_s", "stopped_s", "overshoot_m", "turn_sign", "rotation_start_s",
                   "rotation_end_s", "rotation_duration_s", "final_heading_error_deg",
                   "turn_retries", "next_target_s", "aborted"]


class RunLogger:
    def __init__(self, root, seed, episode):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"seed{seed}_ep{episode:04d}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self._odom_file = (self.dir / "odom.csv").open("w", newline="")
        self._odom = csv.DictWriter(self._odom_file, fieldnames=ODOM_FIELDS)
        self._odom.writeheader()
        self._rev_file = (self.dir / "reversals.csv").open("w", newline="")
        self._rev = csv.DictWriter(self._rev_file, fieldnames=REVERSAL_FIELDS)
        self._rev.writeheader()
        self._written_reversals = 0
        self.config = {}

    def write_config(self, config):
        self.config = config
        tmp = self.dir / "config.json.tmp"
        tmp.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.dir / "config.json")

    def odom_row(self, row):
        self._odom.writerow(row)
        self._odom_file.flush()

    def reversals(self, records):
        for record in records[self._written_reversals:]:
            self._rev.writerow(record)
        self._written_reversals = len(records)
        self._rev_file.flush()

    def close(self):
        self._odom_file.close()
        self._rev_file.close()


def realization_summary(plan, n_targets=8):
    targets = plan.targets(n_targets)
    return {
        "seed": plan.seed, "episode": plan.episode, "obstacle_index": plan.obstacle_index,
        "s0_m": plan.s0, "initial_direction": plan.initial_direction,
        "initial_facing": "E" if plan.initial_direction > 0 else "S",
        "planned_targets_m": targets,
        "gazebo_first_reversal_time_s": abs(plan.first_target - plan.s0) / SPEED_MPS,
    }


def controller_parameters():
    names = ["TRACK_LENGTH_M", "SPEED_MPS", "START_REGION_FRACTION", "END_REGION_FRACTION",
             "CONTROL_RATE_HZ", "TARGET_TOLERANCE_M", "STOP_SPEED_MPS", "STOP_MAX_S",
             "TURN_RATE_MAX", "TURN_RATE_MIN", "TURN_ACCEL", "TURN_P_GAIN", "TURN_FINE_ZONE",
             "TURN_DONE_TOLERANCE", "SETTLE_TOLERANCE", "SETTLE_RATE", "SETTLE_TICKS",
             "SETTLE_MAX_S", "MAX_TURN_RETRIES", "TURN_TIMEOUT_S", "HEADING_GAIN", "LATERAL_GAIN",
             "LATERAL_HEADING_MAX", "STRAIGHT_TURN_RATE_MAX", "TRACK_MARGIN_M", "LATERAL_LIMIT_M",
             "ODOM_TIMEOUT_S"]
    return {name: globals()[name] for name in names}


def print_dry_run(plan, n_targets=8):
    info = realization_summary(plan, n_targets)
    print(f"OB1 realization: seed {plan.seed}, episode {plan.episode}, obstacle index {plan.obstacle_index}")
    print(f"  place Robot 2 at s0 = {plan.s0:.3f} m from S (toward E), facing {info['initial_facing']}")
    print(f"  Gazebo position: x = {1.4 + plan.s0:.3f}, y = 2.000")
    print(f"  first reversal target {plan.first_target:.3f} m "
          f"(Gazebo reaches it after {info['gazebo_first_reversal_time_s']:.1f} s)")
    print("  reversal targets (m): " + ", ".join(f"{v:.3f}" for v in info["planned_targets_m"]))


# ===========================================================================
# ROS 2 node
# ===========================================================================
def run_robot(args):
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, qos_profile_sensor_data

    def yaw_of(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    plan = GazeboReversalPlan(args.seed, args.episode)
    ns = "/" + args.namespace.strip("/")
    cmd_topic = args.cmd_vel_topic or f"{ns}/cmd_vel_unstamped"
    odom_topic = args.odom_topic or f"{ns}/odom"

    # Keep the ROS context alive on Ctrl+C: rclpy's own SIGINT handler shuts the
    # context down first, after which the final zero-velocity commands cannot be
    # published. With it disabled, Ctrl+C (and SIGTERM, below) raise
    # KeyboardInterrupt and the finally block still stops the robot.
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)    # even if launched with SIGINT ignored
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = Node("moving_obstacle")
    cmd_pub = node.create_publisher(Twist, cmd_topic, QoSProfile(depth=10))
    latest = {}

    def on_odom(msg):
        latest["msg"] = msg
        latest["received"] = time.monotonic()

    node.create_subscription(Odometry, odom_topic, on_odom, qos_profile_sensor_data)

    def publish(v, w):
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        cmd_pub.publish(cmd)

    def safe_stop():
        for _ in range(STOP_REPEATS):
            publish(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.02)

    logger = None
    controller = None
    try:
        print_dry_run(plan)
        print(f"Velocity topic {cmd_topic}, odometry topic {odom_topic}")
        deadline = time.monotonic() + ODOM_WAIT_S
        while "msg" not in latest and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if "msg" not in latest:
            raise RuntimeError(f"no odometry on {odom_topic} within {ODOM_WAIT_S:.0f} s")

        first = latest["msg"].pose.pose
        frame = TrackFrame(first.position.x, first.position.y, yaw_of(first.orientation),
                           plan.s0, plan.initial_direction)
        controller = ObstacleController(plan, frame, args.max_duration)
        logger = RunLogger(args.output_root, args.seed, args.episode)
        config = {
            "realization": realization_summary(plan),
            "reference_pose": {"x": frame.x0, "y": frame.y0, "yaw": yaw_of(first.orientation),
                               "theta_e": frame.theta_e, "theta_s": frame.theta_s},
            "topics": {"cmd_vel": cmd_topic, "odom": odom_topic},
            "max_duration_s": args.max_duration,
            "parameters": controller_parameters(),
            "start_wall_time": datetime.now().isoformat(timespec="seconds"),
            "end_reason": None,
        }
        logger.write_config(config)
        print(f"Logging to {logger.dir}")

        start = time.monotonic()
        period = 1.0 / CONTROL_RATE_HZ
        next_tick = start
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=max(0.0, next_tick - time.monotonic()))
            now = time.monotonic()
            if now < next_tick:
                continue
            next_tick += period
            t = now - start
            msg = latest["msg"]
            if now - latest["received"] > ODOM_TIMEOUT_S:
                controller._stop("odom_stale", t)
            pose, twist = msg.pose.pose, msg.twist.twist
            yaw = yaw_of(pose.orientation)
            v_cmd, w_cmd = controller.step(t, pose.position.x, pose.position.y, yaw,
                                           twist.linear.x, twist.angular.z)
            if controller.state == STOPPED:
                v_cmd = w_cmd = 0.0
            publish(v_cmd, w_cmd)
            track = controller.last_track_position
            logger.odom_row({"t_s": t, "state": controller.state, "x": pose.position.x,
                             "y": pose.position.y, "yaw": yaw, "s": track["s"],
                             "lateral": track["lateral"], "target_s": controller.target,
                             "direction": controller.direction,
                             "target_heading": controller.target_heading,
                             "heading_error": controller.heading_error(yaw),
                             "cmd_linear": v_cmd, "cmd_angular": w_cmd,
                             "meas_linear": twist.linear.x, "meas_angular": twist.angular.z})
            logger.reversals(controller.reversals)
            if controller.state == STOPPED:
                break
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping Robot 2.")
        if controller is not None:
            controller._stop("ctrl_c", float("nan"))
    except Exception as exc:
        print(f"Error: {exc}")
        if controller is not None:
            controller._stop(f"error: {exc}", float("nan"))
        raise
    finally:
        try:
            safe_stop()
        except Exception as exc:
            print(f"WARNING: final stop failed: {exc}")
        if logger is not None:
            logger.reversals(controller.reversals)
            logger.config["end_reason"] = controller.end_reason
            logger.config["completed_reversals"] = controller.reversal_count
            logger.config["end_wall_time"] = datetime.now().isoformat(timespec="seconds")
            logger.write_config(logger.config)
            logger.close()
            print(f"Stopped ({controller.end_reason}) after {controller.reversal_count} reversals; "
                  f"log: {logger.dir}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Robot 2 as the Stage 14 moving obstacle OB1.")
    parser.add_argument("--namespace", help="Robot 2's ROS namespace, e.g. robot2 (required unless --dry-run)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Obstacle seed (default: %(default)s)")
    parser.add_argument("--episode", type=int, required=True,
                        help="Gazebo evaluation episode whose OB1 realization to replay (1, 2, ...)")
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S,
                        help="Stop after this many seconds (default: %(default)s, the Stage 14 timeout)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the start position, facing and reversal targets; do not move")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent / "obstacle_runs"))
    parser.add_argument("--cmd-vel-topic", default=None, help="Override <namespace>/cmd_vel_unstamped")
    parser.add_argument("--odom-topic", default=None, help="Override <namespace>/odom")
    args = parser.parse_args(argv)
    if args.episode < 0:
        parser.error("--episode must be >= 0")
    if args.max_duration <= 0:
        parser.error("--max-duration must be > 0")

    if args.dry_run:
        print_dry_run(GazeboReversalPlan(args.seed, args.episode))
        return 0
    if not args.namespace:
        parser.error("--namespace is required to drive the robot")
    run_robot(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
