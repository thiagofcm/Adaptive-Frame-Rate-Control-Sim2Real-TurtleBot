#!/usr/bin/env python3
"""Open-loop linear-velocity step response of the TurtleBot4, for actuator-dynamics modelling.

The robot drives about 2 m straight ahead while the commanded linear velocity steps through a fixed,
distance-scheduled sequence (angular velocity is always 0):

    forward displacement   command
    pre-roll (1 s)         0.00 m/s   robot still: odometry noise baseline
    0.00 - 0.40 m          0.10 m/s   accelerate from rest
    0.40 - 0.80 m          0.15 m/s
    0.80 - 1.40 m          0.20 m/s
    1.40 - 1.95 m          0.10 m/s   decelerate
    from 1.95 m            0.00 m/s   stop; coasts to about 2 m. Zero commands continue for 3 s (settling)

Forward displacement is measured along the starting heading:
    s = (x - x0) cos(yaw0) + (y - y0) sin(yaw0)
Phase transitions are the only use of odometry: there is no heading or speed feedback.

Commands are published at 5 Hz on a fixed grid (the TD3 navigation rate), as geometry_msgs/msg/Twist on
/<namespace>/cmd_vel_unstamped, best effort, depth 10: the same topic and QoS as real_environment and
AdaptiveFPS/scripts/test_straight_motion.py. Odometry callbacks run between ticks, so every odometry
message is recorded at its native rate.

Timing: t_s in both CSV files is time.monotonic() minus the same origin (start of recording), taken
right after publish() for commands and on callback entry for odometry. ros_time_s is the PC's ROS clock at
the same instants; odometry rows also keep the message's header stamp (the robot's clock).

Safety: motion starts only after a matched velocity subscriber and fresh odometry; commands are clamped to
MAX_LINEAR_MPS; the run aborts on odometry older than 0.5 s, more than --max-duration seconds of motion, or
more than --max-distance metres from the start; every exit (finish, abort, exception, Ctrl+C, SIGTERM)
publishes zero velocity. Other publishers on the velocity topic are reported, not prevented. No reverse
motion and no return to start.

Output: RealRobot/motion_dynamic_charac/runs/run_<N>/commands.csv, odometry.csv, run_info.json, where N is one
more than the highest existing run_<N> (1 for the first run)

Usage (inside the tb3_drl container, ROS sourced, same ROS_DOMAIN_ID as the robot):
    python3 RealRobot/motion_dynamic_charac/actuator_step_response.py --namespace robot1
"""

import argparse
import csv
import json
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

# (phase name, start of the phase in metres of forward displacement, commanded linear velocity m/s)
SCHEDULE = [
    ("V010_ACCEL", 0.00, 0.10),
    ("V015", 0.40, 0.15),
    ("V020", 0.80, 0.20),
    ("V010_DECEL", 1.40, 0.10),
    ("STOP", 1.95, 0.00),
]
COMMAND_RATE_HZ = 5.0         # TD3 navigation rate
MAX_LINEAR_MPS = 0.20         # hard clamp on every command
PRE_ROLL_S = 1.0              # zero commands before motion
SETTLE_S = 3.0                # zero commands after STOP, odometry still recorded
STOP_REPEATS = 10             # zero commands on exit
DISCOVERY_TIMEOUT_S = 15.0
ODOM_WAIT_S = 3.0
ODOM_TIMEOUT_S = 0.5

COMMAND_FIELDS = ["t_s", "ros_time_s", "cmd_linear", "cmd_angular", "phase", "forward_m", "odom_age_s"]
ODOM_FIELDS = ["t_s", "ros_time_s", "stamp_s", "x", "y", "yaw", "v_linear", "v_angular", "forward_m"]


class Abort(Exception):
    pass


def next_run_dir(root):
    """root/run_<N>, N = highest existing run_<N> + 1, or 1 if there is none."""
    root = Path(root)
    indices = [int(d.name[4:]) for d in root.glob("run_*") if d.is_dir() and d.name[4:].isdigit()]
    return root / f"run_{max(indices, default=0) + 1}"


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    parser = argparse.ArgumentParser(description="Open-loop linear-velocity step response over about 2 m.")
    parser.add_argument("--namespace", default="robot1", help="Robot namespace (default: %(default)s)")
    parser.add_argument("--max-distance", type=float, default=2.3,
                        help="Abort beyond this distance from the start, m (default: %(default)s)")
    parser.add_argument("--max-duration", type=float, default=40.0,
                        help="Abort if motion lasts longer than this, s (default: %(default)s)")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent / "runs"),
                        help="Run folders are created here (default: runs/ next to this script)")
    args = parser.parse_args()
    if args.max_distance <= SCHEDULE[-1][1]:
        parser.error(f"--max-distance must exceed the stop point {SCHEDULE[-1][1]} m")

    ns = "/" + args.namespace.strip("/")
    cmd_topic, odom_topic = f"{ns}/cmd_vel_unstamped", f"{ns}/odom"

    # Keep the ROS context alive on Ctrl+C / SIGTERM so the final zero commands can still be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = Node("actuator_step_response")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    pub = node.create_publisher(Twist, cmd_topic, QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT))

    state = {"t0": None, "start": None, "odom": None, "received": None, "count": 0,
             "cmd_writer": None, "odom_writer": None, "files": []}

    def ros_now():
        return node.get_clock().now().nanoseconds * 1e-9

    def forward(x, y):
        if state["start"] is None:
            return 0.0
        x0, y0, yaw0 = state["start"]
        return (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)

    def on_odom(msg):
        now = time.monotonic()
        p, t = msg.pose.pose.position, msg.twist.twist
        o = {"x": p.x, "y": p.y, "yaw": yaw_of(msg.pose.pose.orientation),
             "v": t.linear.x, "w": t.angular.z,
             "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9}
        state["odom"], state["received"] = o, now
        state["count"] += 1
        if state["odom_writer"] is not None:
            state["odom_writer"].writerow({
                "t_s": now - state["t0"], "ros_time_s": ros_now(), "stamp_s": o["stamp"],
                "x": o["x"], "y": o["y"], "yaw": o["yaw"], "v_linear": o["v"], "v_angular": o["w"],
                "forward_m": forward(o["x"], o["y"])})

    node.create_subscription(Odometry, odom_topic, on_odom, qos_profile_sensor_data)

    def publish(v, phase):
        v = max(-MAX_LINEAR_MPS, min(MAX_LINEAR_MPS, float(v)))
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = 0.0
        pub.publish(cmd)
        now = time.monotonic()
        if state["cmd_writer"] is not None:
            o = state["odom"]
            state["cmd_writer"].writerow({
                "t_s": now - state["t0"], "ros_time_s": ros_now(), "cmd_linear": v, "cmd_angular": 0.0,
                "phase": phase, "forward_m": forward(o["x"], o["y"]) if o else "",
                "odom_age_s": now - state["received"] if state["received"] else ""})

    def spin_until(deadline):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            executor.spin_once(timeout_sec=remaining)

    period = 1.0 / COMMAND_RATE_HZ
    timing = {"ticks": 0, "late_ticks": 0, "max_late_s": 0.0}

    def tick(t_next):
        """Wait for the next 5 Hz grid point with callbacks running; return the grid point after it."""
        spin_until(t_next)
        late = time.monotonic() - t_next
        timing["ticks"] += 1
        timing["max_late_s"] = max(timing["max_late_s"], late)
        if late > 0.5 * period:
            timing["late_ticks"] += 1
        t_next += period
        while t_next <= time.monotonic():     # a late tick skips missed grid points instead of bursting
            t_next += period
        return t_next

    def check_safety(motion_start):
        age = time.monotonic() - state["received"]
        if age > ODOM_TIMEOUT_S:
            raise Abort(f"odometry is {age:.2f} s old")
        if motion_start is not None and time.monotonic() - motion_start > args.max_duration:
            raise Abort(f"motion exceeded {args.max_duration:.0f} s")
        o, (x0, y0, _) = state["odom"], state["start"]
        if math.hypot(o["x"] - x0, o["y"] - y0) > args.max_distance:
            raise Abort(f"robot is more than {args.max_distance} m from the start")

    info = {"namespace": args.namespace, "cmd_topic": cmd_topic, "odom_topic": odom_topic,
            "cmd_qos": "best_effort, keep last 10", "odom_qos": "sensor_data",
            "command_rate_hz": COMMAND_RATE_HZ, "schedule": SCHEDULE, "max_linear_mps": MAX_LINEAR_MPS,
            "pre_roll_s": PRE_ROLL_S, "settle_s": SETTLE_S, "max_distance_m": args.max_distance,
            "max_duration_s": args.max_duration, "clock": "t_s = time.monotonic() - start of recording",
            "outcome": "NOT_STARTED"}
    run_dir = None
    try:
        # 1. A matched velocity subscriber (the robot) and fresh odometry before anything moves.
        print(f"Velocity topic {cmd_topic} (Twist, best effort, depth 10); odometry {odom_topic}")
        wait_start = time.monotonic()
        while pub.get_subscription_count() < 1:
            if time.monotonic() - wait_start > DISCOVERY_TIMEOUT_S:
                print(f"No matched subscriber on {cmd_topic}. Check ROS_DOMAIN_ID, the namespace and the robot.")
                return 1
            executor.spin_once(timeout_sec=0.1)
        spin_until(time.monotonic() + 0.5)            # let graph info and odometry arrive
        others = [f"{i.node_namespace.rstrip('/')}/{i.node_name}" for i in node.get_publishers_info_by_topic(cmd_topic)
                  if i.node_name != node.get_name()]
        if others:
            print(f"WARNING: other publishers on {cmd_topic}: {others}. Stop them (real_environment, evaluators,"
                  f" teleop), or their commands will mix with this experiment's.")
        info["other_cmd_publishers"] = others

        deadline = time.monotonic() + ODOM_WAIT_S
        while (state["received"] is None or time.monotonic() - state["received"] > ODOM_TIMEOUT_S) \
                and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        if state["received"] is None or time.monotonic() - state["received"] > ODOM_TIMEOUT_S:
            print(f"No fresh odometry on {odom_topic} within {ODOM_WAIT_S:.0f} s; not moving.")
            return 1

        # 2. Recording starts here: one clock origin for both files.
        o = state["odom"]
        state["start"] = (o["x"], o["y"], o["yaw"])
        info["start_pose"] = {"x": o["x"], "y": o["y"], "yaw": o["yaw"]}
        run_dir = next_run_dir(args.output_root)
        run_dir.mkdir(parents=True, exist_ok=False)
        info["run"] = run_dir.name
        info["started"] = datetime.now().isoformat(timespec="seconds")
        for name, fields, key in (("commands.csv", COMMAND_FIELDS, "cmd_writer"),
                                  ("odometry.csv", ODOM_FIELDS, "odom_writer")):
            f = (run_dir / name).open("w", newline="", buffering=1)
            state["files"].append(f)
            state[key] = csv.DictWriter(f, fieldnames=fields)
            state[key].writeheader()
        state["t0"] = time.monotonic()
        info["t0_ros_time_s"] = ros_now()
        print(f"Start pose ({o['x']:.3f}, {o['y']:.3f}), yaw {math.degrees(o['yaw']):.1f} deg. Logging to {run_dir}")

        # 3. Pre-roll: zero commands, robot still.
        t_next = state["t0"]
        while time.monotonic() - state["t0"] < PRE_ROLL_S:
            t_next = tick(t_next)
            check_safety(None)
            publish(0.0, "PRE_ROLL")

        # 4. Distance-scheduled steps until the STOP phase begins.
        motion_start, phase = time.monotonic(), None
        while True:
            t_next = tick(t_next)
            check_safety(motion_start)
            s = forward(state["odom"]["x"], state["odom"]["y"])
            name, _, v = [p for p in SCHEDULE if s >= p[1]][-1] if s >= 0 else SCHEDULE[0]
            if name != phase:
                print(f"[{time.monotonic() - state['t0']:6.2f} s] s = {s:.3f} m -> {name} ({v:.2f} m/s)")
                phase = name
            publish(v, name)
            if name == "STOP":
                break

        # 5. Settling: zero commands, keep recording the deceleration.
        settle_end = time.monotonic() + SETTLE_S
        while time.monotonic() < settle_end:
            t_next = tick(t_next)
            check_safety(None)
            publish(0.0, "SETTLE")
        o = state["odom"]
        info["outcome"] = "COMPLETED"
        info["final_forward_m"] = forward(o["x"], o["y"])
        info["final_lateral_m"] = (-(o["x"] - state["start"][0]) * math.sin(state["start"][2])
                                   + (o["y"] - state["start"][1]) * math.cos(state["start"][2]))
        print(f"[DONE] final forward displacement {info['final_forward_m']:.3f} m, "
              f"lateral {info['final_lateral_m']:+.3f} m")
        return 0
    except Abort as exc:
        info["outcome"] = f"ABORTED: {exc}"
        print(f"ABORTED: {exc}")
        return 2
    except KeyboardInterrupt:
        info["outcome"] = "INTERRUPTED"
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        info["outcome"] = f"ERROR: {exc!r}"
        raise
    finally:
        try:
            for _ in range(STOP_REPEATS):
                publish(0.0, "EXIT_STOP")
                executor.spin_once(timeout_sec=0.05)
            print(f"Sent {STOP_REPEATS} zero-velocity commands.")
        except Exception as exc:
            print(f"WARNING: zero-velocity commands failed: {exc}")
        info.update({"odometry_messages": state["count"], "command_ticks": timing["ticks"],
                     "late_ticks": timing["late_ticks"], "max_tick_lateness_s": timing["max_late_s"]})
        if state["t0"] is not None:
            print(f"Timing: {timing['late_ticks']} late tick(s) of {timing['ticks']}, worst lateness "
                  f"{timing['max_late_s'] * 1000:.0f} ms")
        state["cmd_writer"] = state["odom_writer"] = None
        for f in state["files"]:
            f.close()
        if run_dir is not None:
            (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        executor.shutdown(timeout_sec=0.1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
