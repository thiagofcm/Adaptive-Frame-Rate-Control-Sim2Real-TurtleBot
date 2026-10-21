#!/usr/bin/env python3
"""Hardware check, stage 2: drive a TurtleBot4 straight, then turn 180 degrees in place.

Sequence: MOVE_FORWARD (0.06 m/s for 5 s, about 0.30 m) -> STOP -> ROTATE_180 -> STOP.

Publishes geometry_msgs/msg/Twist on /<namespace>/cmd_vel_unstamped with the same
QoS as Robot 1's working velocity publisher (drl_environment_real.py: best effort,
keep last 10, volatile).

The turn uses /<namespace>/odom: yaw from the orientation quaternion, accumulated
("unwrapped") across messages, so passing +-180 degrees never makes the error jump.
The turn is always counter-clockwise toward +180 degrees of accumulated rotation,
with a proportional command capped at 0.3 rad/s, and ends below a 3 degree error.

Safety:
  * Motion starts only after a matched velocity subscriber and odometry are found.
  * The turn stops if odometry is older than 0.5 s or the turn exceeds 25 s.
  * Every exit (finish, abort, Ctrl+C, SIGTERM) publishes zero velocity.
  * If the script dies without cleanup (e.g. kill -9), the robot keeps moving only
    until its own command timeout stops it; no command lasts longer than one 0.05 s tick.

Usage:
    python3 test_straight_motion.py                    # robot3
    python3 test_straight_motion.py --namespace robot1
"""

import argparse
import math
import signal
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions

LINEAR_SPEED_MPS = 0.06
DURATION_S = 5.0
RATE_HZ = 20.0
DISCOVERY_TIMEOUT_S = 15.0
ODOM_WAIT_S = 3.0           # odometry must be available before any motion
STOP_REPEATS = 10           # zero commands at the end (0.5 s at 20 Hz)

# STOP phase: zero commands until the robot is still, at most SETTLE_MAX_S.
SETTLE_MAX_S = 1.5
SETTLE_LINEAR = 0.01        # m/s
SETTLE_ANGULAR = 0.05       # rad/s
SETTLE_TICKS = 3

# ROTATE_180 phase.
TARGET_ROTATION = math.pi   # rad, counter-clockwise
TURN_GAIN = 1.0             # rad/s per rad of remaining error
TURN_RATE_MAX = 0.3         # rad/s
TURN_RATE_MIN = 0.05        # rad/s, keeps the robot from stalling just short of the target
TURN_TOLERANCE = math.radians(3.0)
ODOM_TIMEOUT_S = 0.5
TURN_TIMEOUT_S = 25.0       # about 2x the expected turn time at 0.3 rad/s


class Abort(Exception):
    pass


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    parser = argparse.ArgumentParser(description="Stage 2: straight motion then a 180 degree turn.")
    parser.add_argument("--namespace", default="robot3", help="Robot namespace (default: %(default)s)")
    parser.add_argument("--reliability", choices=["best_effort", "reliable"], default="best_effort",
                        help="Velocity publisher reliability (default: best_effort, as Robot 1 uses)")
    args = parser.parse_args()

    ns = "/" + args.namespace.strip("/")
    cmd_topic, odom_topic = f"{ns}/cmd_vel_unstamped", f"{ns}/odom"
    reliability = (QoSReliabilityPolicy.BEST_EFFORT if args.reliability == "best_effort"
                   else QoSReliabilityPolicy.RELIABLE)

    # Keep the ROS context alive on Ctrl+C / SIGTERM so the final zero commands can be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = Node("test_straight_motion")
    pub = node.create_publisher(Twist, cmd_topic, QoSProfile(depth=10, reliability=reliability))

    odom = {"first": None, "last": None, "count": 0, "max_speed": 0.0, "received": None,
            "yaw": None, "unwrapped": None, "v": 0.0, "w": 0.0, "max_gap": 0.0, "track_gap": False}

    def on_odom(msg):
        now = time.monotonic()
        if odom["track_gap"] and odom["received"] is not None:
            odom["max_gap"] = max(odom["max_gap"], now - odom["received"])
        odom["received"] = now
        p = msg.pose.pose.position
        odom["last"] = (p.x, p.y)
        odom["count"] += 1
        if odom["first"] is None:
            odom["first"] = (p.x, p.y)
        odom["v"], odom["w"] = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        odom["max_speed"] = max(odom["max_speed"], abs(odom["v"]))
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
            rclpy.spin_once(node, timeout_sec=1.0 / RATE_HZ)

    def wait_until(t):
        while True:
            remaining = t - time.monotonic()
            if remaining <= 0:
                return
            rclpy.spin_once(node, timeout_sec=remaining)

    def settle(label):
        print(f"[STOP] {label}: zero velocity until still (at most {SETTLE_MAX_S} s)")
        t0, still, k = time.monotonic(), 0, 0
        while time.monotonic() - t0 < SETTLE_MAX_S:
            send(0.0)
            k += 1
            wait_until(t0 + k / RATE_HZ)
            still = still + 1 if abs(odom["v"]) < SETTLE_LINEAR and abs(odom["w"]) < SETTLE_ANGULAR else 0
            if still >= SETTLE_TICKS:
                break
        print(f"[STOP] settled after {time.monotonic() - t0:.2f} s "
              f"(measured v={odom['v']:.3f} m/s, w={odom['w']:.3f} rad/s)")

    print(f"Velocity topic {cmd_topic} (geometry_msgs/msg/Twist, {args.reliability}, depth 10)")
    sent = 0
    moving = False
    forward_end = None
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
            rclpy.spin_once(node, timeout_sec=0.1)
        print(f"Matched {pub.get_subscription_count()} subscriber(s) after "
              f"{time.monotonic() - start_wait:.1f} s:")
        settle_end = time.monotonic() + 0.5          # let graph info and odometry arrive
        while time.monotonic() < settle_end:
            rclpy.spin_once(node, timeout_sec=0.05)
        for info in node.get_subscriptions_info_by_topic(cmd_topic):
            print(f"  node {info.node_namespace}/{info.node_name}, type {info.topic_type}, "
                  f"reliability {info.qos_profile.reliability.name}")

        # Odometry is required for the turn, so check it before any motion.
        odom_deadline = time.monotonic() + ODOM_WAIT_S
        while odom["yaw"] is None and time.monotonic() < odom_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if odom["yaw"] is None:
            print(f"No odometry on {odom_topic} within {ODOM_WAIT_S:.0f} s; not moving.")
            return 1
        print(f"Odometry on {odom_topic}: received, yaw {math.degrees(odom['yaw']):.1f} deg")

        # 2. MOVE_FORWARD: 0.06 m/s, 20 Hz, 5 s.
        n_ticks = int(round(DURATION_S * RATE_HZ))
        print(f"[MOVE_FORWARD] {LINEAR_SPEED_MPS} m/s for {DURATION_S:.0f} s ({n_ticks} commands)")
        moving = True
        t0 = time.monotonic()
        for k in range(n_ticks):
            send(LINEAR_SPEED_MPS)
            sent += 1
            wait_until(t0 + (k + 1) / RATE_HZ)
        elapsed = time.monotonic() - t0
        print(f"[MOVE_FORWARD] sent {sent} commands in {elapsed:.2f} s ({sent / elapsed:.1f} Hz)")

        # 3. STOP.
        settle("after forward motion")
        forward_end = odom["last"]

        # 4. ROTATE_180: counter-clockwise, on accumulated yaw.
        start_unwrapped, start_yaw = odom["unwrapped"], odom["yaw"]
        print(f"[ROTATE_180] initial yaw {math.degrees(start_yaw):.1f} deg, target rotation "
              f"+{math.degrees(TARGET_ROTATION):.0f} deg (counter-clockwise), max {TURN_RATE_MAX} rad/s")
        odom["max_gap"], odom["track_gap"] = 0.0, True
        t0, k, next_print = time.monotonic(), 0, 1.0
        while True:
            now = time.monotonic()
            if now - odom["received"] > ODOM_TIMEOUT_S:
                raise Abort(f"odometry older than {ODOM_TIMEOUT_S} s during the turn")
            if now - t0 > TURN_TIMEOUT_S:
                raise Abort(f"turn not finished within {TURN_TIMEOUT_S:.0f} s")
            rotated = odom["unwrapped"] - start_unwrapped
            error = TARGET_ROTATION - rotated
            if abs(error) < TURN_TOLERANCE:
                break
            magnitude = max(TURN_RATE_MIN, min(TURN_RATE_MAX, TURN_GAIN * abs(error)))
            send(0.0, math.copysign(magnitude, error))
            if now - t0 >= next_print:
                print(f"[ROTATE_180] t={now - t0:4.1f} s  rotated {math.degrees(rotated):6.1f} deg  "
                      f"error {math.degrees(error):6.1f} deg  w_cmd {math.copysign(magnitude, error):.2f}")
                next_print += 1.0
            k += 1
            wait_until(t0 + k / RATE_HZ)
        print(f"[ROTATE_180] reached tolerance after {time.monotonic() - t0:.1f} s, "
              f"rotated {math.degrees(odom['unwrapped'] - start_unwrapped):.1f} deg")

        # 5. STOP, then report the settled result.
        settle("after rotation")
        rotated = odom["unwrapped"] - start_unwrapped
        print(f"[RESULT] accumulated rotation {math.degrees(rotated):.1f} deg, final error "
              f"{math.degrees(TARGET_ROTATION - rotated):+.1f} deg, final yaw {math.degrees(odom['yaw']):.1f} deg, "
              f"largest odometry gap during the turn {odom['max_gap']:.3f} s")
        return 0
    except Abort as exc:
        print(f"ABORTED: {exc}")
        return 2
    except KeyboardInterrupt:
        print(f"\nInterrupted after {sent} forward commands.")
        return 130
    finally:
        try:
            stop()
            print(f"Sent {STOP_REPEATS} zero-velocity commands.")
        except Exception as exc:
            print(f"WARNING: zero-velocity commands failed: {exc}")
        end = forward_end or odom["last"]
        if moving and odom["first"] is not None and end is not None:
            dx, dy = end[0] - odom["first"][0], end[1] - odom["first"][1]
            print(f"Odometry: forward displacement {math.hypot(dx, dy):.3f} m "
                  f"(expected about {LINEAR_SPEED_MPS * DURATION_S:.2f} m), "
                  f"max measured speed {odom['max_speed']:.3f} m/s, {odom['count']} messages")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
