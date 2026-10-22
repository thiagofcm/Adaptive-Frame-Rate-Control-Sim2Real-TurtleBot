#!/usr/bin/env python3
"""Hardware check, stage 3: deterministic back-and-forth on a fixed 1 m track.

Sequence per cycle:
    MOVE_TO_END -> STOP -> ROTATE_180 -> STOP -> MOVE_TO_START -> STOP -> ROTATE_180 -> STOP

Track: the odometry pose at start defines S = (x0, y0) and theta0; E is 1 m ahead
along theta0. Both stay fixed for the whole run.
    s       = (x - x0) cos(theta0) + (y - y0) sin(theta0)       (progress along S->E)
    lateral = -(x - x0) sin(theta0) + (y - y0) cos(theta0)       (left of S->E is positive)

Straight legs use v = 0.10 m/s and w = 0 with NO heading or lateral correction (a
baseline for measuring natural drift); a leg ends when s reaches the endpoint within
--tolerance. Turns aim at the fixed headings theta0 + pi (at E) and theta0 (at S),
always counter-clockwise, tracked on accumulated yaw so +-180 degrees never jumps.
After every turn the robot must be confirmed still (distinct odometry samples) and its
heading must be within 3 degrees, fixed if needed by up to 3 short corrective pulses;
otherwise the run aborts instead of starting the next leg.

Publishes geometry_msgs/msg/Twist on /<namespace>/cmd_vel_unstamped with the same QoS
as Robot 1's working velocity publisher (best effort, keep last 10, volatile).

Safety: motion starts only after a matched velocity subscriber and odometry are found;
any odometry older than 0.5 s, a leg longer than 35 s, a turn longer than 25 s, s outside
[-0.15, 1.15] m or |lateral| above 0.25 m aborts the run; every exit (finish, abort,
Ctrl+C, SIGTERM) publishes zero velocity.

Logs: RealRobot/obstacle_tests/back_and_forth_<time>/trajectory.csv and endpoints.csv.

Usage:
    python3 test_straight_motion.py --namespace robot3 --cycles 1
"""

import argparse
import csv
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions

LINEAR_SPEED_MPS = 0.10     # test speed (Gazebo Stage 14 OB1 itself moves at 0.06 m/s)
TRACK_LENGTH_M = 1.0
RATE_HZ = 20.0
DISCOVERY_TIMEOUT_S = 15.0
ODOM_WAIT_S = 3.0           # odometry must be available before any motion
STOP_REPEATS = 10           # zero commands at the end (0.5 s at 20 Hz)
ODOM_TIMEOUT_S = 0.5        # abort if odometry is older than this during the run

# STOP phase: zero commands until SETTLE_SAMPLES distinct odometry samples show the robot
# still; abort if that does not happen within SETTLE_MAX_S.
SETTLE_MAX_S = 3.0          # odometry kept reporting rotation 1.45 s after the zero command
SETTLE_LINEAR = 0.01        # m/s
SETTLE_ANGULAR = 0.05       # rad/s
SETTLE_SAMPLES = 3

# Straight legs.
DEFAULT_TOLERANCE_M = 0.02
LEG_TIMEOUT_S = 35.0        # a 1 m leg takes 10 s at 0.10 m/s
TRACK_ABORT_MARGIN_M = 0.15
LATERAL_ABORT_M = 0.25      # generous: natural drift is being measured, runaways are not allowed

# ROTATE_180 (unchanged controller).
TURN_GAIN = 1.0             # rad/s per rad of remaining error
TURN_RATE_MAX = 0.3         # rad/s
TURN_RATE_MIN = 0.05        # rad/s, keeps the robot from stalling just short of the target
TURN_TOLERANCE = math.radians(3.0)
TURN_TIMEOUT_S = 25.0       # about 2x the expected turn time at 0.3 rad/s

# Heading check after a turn: open-loop corrective pulses, each followed by settling.
CORRECTION_RATE = 0.10      # rad/s
MAX_CORRECTIONS = 3
CORRECTION_MAX_PULSE_S = 6.0

TRAJECTORY_FIELDS = ["t_s", "state", "cycle", "x", "y", "yaw", "s", "lateral",
                     "heading_error", "cmd_linear", "cmd_angular", "odom_age_s", "odom_seq"]
ENDPOINT_FIELDS = ["cycle", "leg", "target_s", "stop_s", "along_track_error_m", "lateral_m",
                   "heading_error_deg", "leg_duration_s", "turn_rotation_deg", "turn_error_after_settle_deg",
                   "turn_corrections", "turn_final_error_deg", "turn_duration_s"]


class Abort(Exception):
    pass


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    parser = argparse.ArgumentParser(description="Stage 3: back and forth on a fixed 1 m odometry track.")
    parser.add_argument("--namespace", default="robot3", help="Robot namespace (default: %(default)s)")
    parser.add_argument("--reliability", choices=["best_effort", "reliable"], default="best_effort",
                        help="Velocity publisher reliability (default: best_effort, as Robot 1 uses)")
    parser.add_argument("--cycles", type=int, default=1,
                        help="Round trips S->E->S, each with two 180 degree turns (default: %(default)s)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_M,
                        help="Endpoint tolerance along the track in m (default: %(default)s)")
    parser.add_argument("--log-root",default=str(Path(__file__).resolve().parent / "obstacle_tests"), help="Where the run folder is written (default: obstacle_tests/ next to the script)")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be >= 1")
    if not 0.0 < args.tolerance < TRACK_LENGTH_M / 2:
        parser.error("--tolerance must be between 0 and half the track length")

    ns = "/" + args.namespace.strip("/")
    cmd_topic, odom_topic = f"{ns}/cmd_vel_unstamped", f"{ns}/odom"
    reliability = (QoSReliabilityPolicy.BEST_EFFORT if args.reliability == "best_effort"
                   else QoSReliabilityPolicy.RELIABLE)

    # Keep the ROS context alive on Ctrl+C / SIGTERM so the final zero commands can be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = Node("test_straight_motion")
    # One executor for the whole run (the global executor used by the rclpy helper
    # would add and remove the node on every call).
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    pub = node.create_publisher(Twist, cmd_topic, QoSProfile(depth=10, reliability=reliability))

    odom = {"x": None, "y": None, "count": 0, "received": None,
            "yaw": None, "unwrapped": None, "v": 0.0, "w": 0.0, "max_gap": 0.0, "track_gap": False}

    def on_odom(msg):
        now = time.monotonic()
        if odom["track_gap"] and odom["received"] is not None:
            odom["max_gap"] = max(odom["max_gap"], now - odom["received"])
        odom["received"] = now
        p = msg.pose.pose.position
        odom["x"], odom["y"] = p.x, p.y
        odom["count"] += 1
        odom["v"], odom["w"] = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        yaw = yaw_of(msg.pose.pose.orientation)
        # Unwrapped yaw: add each step's wrapped change, so crossing +-pi never jumps.
        if odom["yaw"] is None:
            odom["unwrapped"] = yaw
        else:
            odom["unwrapped"] += wrap_angle(yaw - odom["yaw"])
        odom["yaw"] = yaw

    node.create_subscription(Odometry, odom_topic, on_odom, qos_profile_sensor_data)

    def send(v, w=0.0):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        pub.publish(cmd)

    def stop():
        for _ in range(STOP_REPEATS):
            send(0.0)
            executor.spin_once(timeout_sec=1.0 / RATE_HZ)

    period = 1.0 / RATE_HZ
    timing = {"late_ticks": 0, "max_late": 0.0}

    def drain_callbacks():
        """Run every odometry callback already waiting (bounded by the queue depth)."""
        for _ in range(10):
            before = odom["count"]
            executor.spin_once(timeout_sec=0.0)
            if odom["count"] == before:
                return

    def wait_tick(t_next):
        """End of one control tick: process callbacks, wait until t_next, return the next tick time.

        Callbacks are processed even when the loop is already late, so odometry stays fresh
        after a slow tick. A late loop skips the missed ticks instead of running them back to
        back without spinning.
        """
        drain_callbacks()
        while True:
            remaining = t_next - time.monotonic()
            if remaining <= 0:
                break
            executor.spin_once(timeout_sec=remaining)
        now = time.monotonic()
        late = now - t_next
        if late > 0.5 * period:
            timing["late_ticks"] += 1
            timing["max_late"] = max(timing["max_late"], late)
        return t_next + period if now < t_next + period else now + period

    print(f"Velocity topic {cmd_topic} (geometry_msgs/msg/Twist, {args.reliability}, depth 10)")
    track = {}          # x0, y0, theta0 once motion starts
    run = {"t0": None, "files": []}
    traj_writer = end_writer = None

    def project():
        dx, dy = odom["x"] - track["x0"], odom["y"] - track["y0"]
        c, s = math.cos(track["theta0"]), math.sin(track["theta0"])
        return dx * c + dy * s, -dx * s + dy * c

    def check_odom():
        age = time.monotonic() - odom["received"]
        if age > ODOM_TIMEOUT_S:
            raise Abort(f"odometry is {age:.2f} s old (limit {ODOM_TIMEOUT_S} s)")
        return age

    def log_tick(state, cycle, heading_ref, v, w):
        age = check_odom()
        s, lateral = project()
        traj_writer.writerow({"t_s": time.monotonic() - run["t0"], "state": state, "cycle": cycle,
                              "x": odom["x"], "y": odom["y"], "yaw": odom["yaw"], "s": s,
                              "lateral": lateral, "heading_error": wrap_angle(heading_ref - odom["yaw"]),
                              "cmd_linear": v, "cmd_angular": w, "odom_age_s": age,
                              "odom_seq": odom["count"]})
        run["files"][0].flush()
        return s, lateral

    def settle(cycle, heading_ref, label):
        """Zero velocity until SETTLE_SAMPLES distinct odometry samples show the robot still."""
        t0, still, last_seq = time.monotonic(), 0, odom["count"]
        t_next = t0 + period
        while True:
            send(0.0)
            log_tick("STOP", cycle, heading_ref, 0.0, 0.0)
            t_next = wait_tick(t_next)
            if odom["count"] != last_seq:            # judge only new samples, never a repeated one
                last_seq = odom["count"]
                still = still + 1 if abs(odom["v"]) < SETTLE_LINEAR and abs(odom["w"]) < SETTLE_ANGULAR else 0
                if still >= SETTLE_SAMPLES:
                    return
            if time.monotonic() - t0 > SETTLE_MAX_S:
                raise Abort(f"robot not confirmed still within {SETTLE_MAX_S:.0f} s {label} "
                            f"(measured v={odom['v']:.3f} m/s, w={odom['w']:.3f} rad/s)")

    def move_leg(cycle, state, target_s, direction, heading_ref):
        """Drive straight (v = 0.10, w = 0, no correction) until s reaches target_s within tolerance."""
        print(f"[{state}] cycle {cycle}: toward s = {target_s:.2f} m")
        t0 = time.monotonic()
        t_next = t0 + period
        while True:
            s, lateral = log_tick(state, cycle, heading_ref, LINEAR_SPEED_MPS, 0.0)
            if direction * (target_s - s) <= args.tolerance:
                break
            if not -TRACK_ABORT_MARGIN_M <= s <= TRACK_LENGTH_M + TRACK_ABORT_MARGIN_M:
                raise Abort(f"s = {s:.3f} m left the track during {state}")
            if abs(lateral) > LATERAL_ABORT_M:
                raise Abort(f"lateral deviation {lateral:.3f} m exceeded {LATERAL_ABORT_M} m during {state}")
            if time.monotonic() - t0 > LEG_TIMEOUT_S:
                raise Abort(f"{state} did not reach s = {target_s} within {LEG_TIMEOUT_S:.0f} s")
            send(LINEAR_SPEED_MPS, 0.0)
            t_next = wait_tick(t_next)
        duration = time.monotonic() - t0
        settle(cycle, heading_ref, f"after {state}")
        s, lateral = project()
        heading_error = math.degrees(wrap_angle(heading_ref - odom["yaw"]))
        print(f"[{state}] stopped at s = {s:.3f} m (error {s - target_s:+.3f} m), lateral {lateral:+.3f} m, "
              f"heading error {heading_error:+.1f} deg, {duration:.1f} s")
        return {"cycle": cycle, "leg": state, "target_s": target_s, "stop_s": s,
                "along_track_error_m": s - target_s, "lateral_m": lateral,
                "heading_error_deg": heading_error, "leg_duration_s": duration}

    def rotate_to(cycle, target_heading):
        """Turn counter-clockwise to the fixed heading target_heading (unchanged P controller)."""
        # Counter-clockwise angle still needed to reach the fixed heading; tracked on accumulated yaw.
        needed = (target_heading - odom["yaw"]) % (2 * math.pi)
        if needed > 1.5 * math.pi:           # already slightly past the target: small clockwise fix
            needed -= 2 * math.pi
        start_unwrapped = odom["unwrapped"]
        print(f"[ROTATE_180] cycle {cycle}: yaw {math.degrees(odom['yaw']):.1f} deg -> "
              f"{math.degrees(wrap_angle(target_heading)):.1f} deg, rotation needed {math.degrees(needed):+.1f} deg")
        t0 = time.monotonic()
        t_next = t0 + period
        while True:
            rotated = odom["unwrapped"] - start_unwrapped
            error = needed - rotated
            if abs(error) < TURN_TOLERANCE:
                break
            if time.monotonic() - t0 > TURN_TIMEOUT_S:
                raise Abort(f"turn not finished within {TURN_TIMEOUT_S:.0f} s")
            w = math.copysign(max(TURN_RATE_MIN, min(TURN_RATE_MAX, TURN_GAIN * abs(error))), error)
            log_tick("ROTATE_180", cycle, target_heading, 0.0, w)
            send(0.0, w)
            t_next = wait_tick(t_next)
        print(f"[ROTATE_180] loop ended at heading {math.degrees(odom['yaw']):.1f} deg after "
              f"{time.monotonic() - t0:.1f} s")
        settle(cycle, target_heading, "after turn")
        first_error = math.degrees(wrap_angle(target_heading - odom["yaw"]))
        corrections = 0
        while True:
            error = wrap_angle(target_heading - odom["yaw"])
            print(f"[HEADING] after settling ({corrections} correction(s)): heading "
                  f"{math.degrees(odom['yaw']):.1f} deg, error {math.degrees(error):+.1f} deg")
            if abs(error) <= TURN_TOLERANCE:
                break
            if corrections >= MAX_CORRECTIONS:
                raise Abort(f"heading error {math.degrees(error):+.1f} deg after {corrections} corrective "
                            "turns; not starting the next leg")
            corrections += 1
            # Open-loop pulse sized from the settled error; the next settle measures the result.
            w = math.copysign(CORRECTION_RATE, error)
            pulse = min(abs(error) / CORRECTION_RATE, CORRECTION_MAX_PULSE_S)
            print(f"[CORRECT_TURN] {corrections}/{MAX_CORRECTIONS}: {w:+.2f} rad/s for {pulse:.2f} s")
            p0 = time.monotonic()
            t_next = p0 + period
            while time.monotonic() - p0 < pulse:
                log_tick("CORRECT_TURN", cycle, target_heading, 0.0, w)
                send(0.0, w)
                t_next = wait_tick(t_next)
            send(0.0)
            settle(cycle, target_heading, "after corrective turn")
        duration = time.monotonic() - t0
        rotated = odom["unwrapped"] - start_unwrapped
        final_error = math.degrees(error)
        print(f"[ROTATE_180] rotated {math.degrees(rotated):.1f} deg in {duration:.1f} s, error after first "
              f"settling {first_error:+.1f} deg, final error {final_error:+.1f} deg, {corrections} correction(s)")
        return {"turn_rotation_deg": math.degrees(rotated), "turn_error_after_settle_deg": first_error,
                "turn_corrections": corrections, "turn_final_error_deg": final_error,
                "turn_duration_s": duration}

    try:
        # 1. Discovery: wait for a matched (QoS-compatible) velocity subscriber.
        print(f"Waiting up to {DISCOVERY_TIMEOUT_S:.0f} s for a velocity subscriber...")
        start_wait = time.monotonic()
        while pub.get_subscription_count() < 1:
            if time.monotonic() - start_wait > DISCOVERY_TIMEOUT_S:
                seen = node.get_subscriptions_info_by_topic(cmd_topic)
                if seen:
                    print("Subscribers were discovered but none matched this publisher's QoS:")
                    for info in seen:
                        print(f"  node {info.node_namespace}/{info.node_name}, type {info.topic_type}, "
                              f"reliability {info.qos_profile.reliability.name}")
                    print("Try --reliability reliable." if args.reliability == "best_effort"
                          else "Try --reliability best_effort.")
                else:
                    print(f"No subscriber on {cmd_topic}. Check ROS_DOMAIN_ID, the namespace and the robot.")
                return 1
            executor.spin_once(timeout_sec=0.1)
        print(f"Matched {pub.get_subscription_count()} subscriber(s) after "
              f"{time.monotonic() - start_wait:.1f} s:")
        settle_end = time.monotonic() + 0.5          # let graph info and odometry arrive
        while time.monotonic() < settle_end:
            executor.spin_once(timeout_sec=0.05)
        for info in node.get_subscriptions_info_by_topic(cmd_topic):
            print(f"  node {info.node_namespace}/{info.node_name}, type {info.topic_type}, "
                  f"reliability {info.qos_profile.reliability.name}")

        # Fresh odometry is required before any motion.
        odom_deadline = time.monotonic() + ODOM_WAIT_S
        while (odom["yaw"] is None or time.monotonic() - odom["received"] > ODOM_TIMEOUT_S) \
                and time.monotonic() < odom_deadline:
            executor.spin_once(timeout_sec=0.05)
        if odom["yaw"] is None or time.monotonic() - odom["received"] > ODOM_TIMEOUT_S:
            print(f"No fresh odometry on {odom_topic} within {ODOM_WAIT_S:.0f} s; not moving.")
            return 1

        # 2. Fixed track from the current pose.
        track.update({"x0": odom["x"], "y0": odom["y"], "theta0": odom["yaw"]})
        theta0 = track["theta0"]
        print(f"Track: S = ({track['x0']:.3f}, {track['y0']:.3f}), theta0 = {math.degrees(theta0):.1f} deg, "
              f"E = ({track['x0'] + math.cos(theta0):.3f}, {track['y0'] + math.sin(theta0):.3f}), "
              f"{args.cycles} cycle(s), tolerance {args.tolerance} m")
        run_dir = Path(args.log_root) / f"back_and_forth_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=False)
        traj_file = (run_dir / "trajectory.csv").open("w", newline="")
        end_file = (run_dir / "endpoints.csv").open("w", newline="")
        run["files"] = [traj_file, end_file]
        traj_writer = csv.DictWriter(traj_file, fieldnames=TRAJECTORY_FIELDS)
        end_writer = csv.DictWriter(end_file, fieldnames=ENDPOINT_FIELDS)
        traj_writer.writeheader()
        end_writer.writeheader()
        print(f"Logging to {run_dir}")

        # 3. Cycles: S -> E, turn to theta0 + pi, E -> S, turn to theta0.
        run["t0"] = time.monotonic()
        odom["max_gap"], odom["track_gap"] = 0.0, True
        heading_e, heading_s = theta0, wrap_angle(theta0 + math.pi)
        for cycle in range(1, args.cycles + 1):
            leg = move_leg(cycle, "MOVE_TO_END", TRACK_LENGTH_M, +1, heading_e)
            leg.update(rotate_to(cycle, heading_s))
            end_writer.writerow(leg)
            end_file.flush()
            leg = move_leg(cycle, "MOVE_TO_START", 0.0, -1, heading_s)
            leg.update(rotate_to(cycle, heading_e))
            end_writer.writerow(leg)
            end_file.flush()
        print(f"[DONE] {args.cycles} cycle(s) in {time.monotonic() - run['t0']:.1f} s, "
              f"largest odometry gap {odom['max_gap']:.3f} s")
        return 0
    except Abort as exc:
        print(f"ABORTED: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        try:
            stop()
            print(f"Sent {STOP_REPEATS} zero-velocity commands.")
        except Exception as exc:
            print(f"WARNING: zero-velocity commands failed: {exc}")
        if run["t0"] is not None:
            print(f"Loop timing: {timing['late_ticks']} late tick(s), worst lateness "
                  f"{timing['max_late'] * 1000:.0f} ms (period {period * 1000:.0f} ms)")
        for f in run["files"]:
            f.close()
        executor.shutdown(timeout_sec=0.1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
