#!/usr/bin/env python3
"""Open-loop angular-velocity step response of the TurtleBot4, for actuator-dynamics modelling.

Companion of actuator_step_response.py (linear). The robot rotates in place while the commanded angular
velocity steps through a fixed, TIME-scheduled sequence (linear velocity is always 0):

    phase              duration   angular velocity
    BASELINE           2 s         0.0 rad/s   robot still: odometry noise baseline
    POS_LOW            3 s        +0.2
    POS_MED            3 s        +0.4
    POS_HIGH           3 s        +0.6
    POS_DECEL          3 s        +0.2
    STOP_1             2 s         0.0
    NEG_LOW            3 s        -0.2
    NEG_MED            3 s        -0.4
    NEG_HIGH           3 s        -0.6
    NEG_DECEL          3 s        -0.2
    STOP_2             2 s         0.0
    REV_POS            2 s        +0.4
    REV_NEG            2 s        -0.4         immediate reversal from +0.4
    FINAL_STOP         2 s         0.0         settling
                       36 s total; positive is counter-clockwise (ROS convention)

Edit SEQUENCE below to change it. Each phase lasts round(duration * 5) commands on the 5 Hz grid, so phase
boundaries fall exactly on command ticks. Odometry is only used for safety checks: there is no heading,
speed or position feedback, no correction of drift and no return to the starting heading.

Same ROS setup as the linear experiment: geometry_msgs/msg/Twist on /<namespace>/cmd_vel_unstamped (best
effort, depth 10), one command every 0.2 s on a fixed grid even when the value does not change; every
nav_msgs/msg/Odometry message on /<namespace>/odom (sensor-data QoS) is recorded at its native rate.

Timing: t_s in both CSV files is time.monotonic() minus the same origin (start of recording), taken right
after publish() for commands and on callback entry for odometry. ros_time_s is the PC's ROS clock at the
same instants; odometry rows also keep the message's header stamp (the robot's clock).

Yaw: yaw is the raw odometry yaw in (-pi, pi]; yaw_unwrapped adds each sample's wrapped change, so it is
continuous over full turns; rotation_rad = yaw_unwrapped - its value at the start of recording.

Safety: motion starts only after a matched velocity subscriber and fresh odometry; linear commands are
always 0 and angular commands are clamped to --max-angular; the run aborts on odometry older than 0.5 s, a
run longer than --max-duration, |rotation_rad| above --max-rotation, or a translation above
--max-translation from the start; every exit (finish, abort, exception, Ctrl+C, SIGTERM) publishes zero
velocity. Other publishers on the velocity topic are reported, not prevented.

Output: RealRobot/motion_dynamic_charac/runs/angular_run_<N>/commands.csv, odometry.csv, run_info.json, where
N is one more than the highest existing angular_run_<N> (1 for the first run)

Usage (inside the tb3_drl container, ROS sourced, same ROS_DOMAIN_ID as the robot):
    python3 RealRobot/motion_dynamic_charac/angular_step_response.py --namespace robot1
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

# (phase name, duration s, commanded angular velocity rad/s); positive = counter-clockwise
SEQUENCE = [
    ("BASELINE", 2.0, 0.0),
    ("POS_LOW", 3.0, 0.2),
    ("POS_MED", 3.0, 0.4),
    ("POS_HIGH", 3.0, 0.6),
    ("POS_DECEL", 3.0, 0.2),
    ("STOP_1", 2.0, 0.0),
    ("NEG_LOW", 3.0, -0.2),
    ("NEG_MED", 3.0, -0.4),
    ("NEG_HIGH", 3.0, -0.6),
    ("NEG_DECEL", 3.0, -0.2),
    ("STOP_2", 2.0, 0.0),
    ("REV_POS", 2.0, 0.4),
    ("REV_NEG", 2.0, -0.4),
    ("FINAL_STOP", 2.0, 0.0),
]
COMMAND_RATE_HZ = 5.0         # TD3 navigation rate
STOP_REPEATS = 10             # zero commands on exit
DISCOVERY_TIMEOUT_S = 15.0
ODOM_WAIT_S = 3.0
ODOM_TIMEOUT_S = 0.5

COMMAND_FIELDS = ["t_s", "ros_time_s", "cmd_linear", "cmd_angular", "phase", "yaw", "rotation_rad", "odom_age_s"]
ODOM_FIELDS = ["t_s", "ros_time_s", "stamp_s", "x", "y", "yaw", "yaw_unwrapped", "rotation_rad",
               "v_linear", "v_angular"]


class Abort(Exception):
    pass


def next_run_dir(root, prefix="angular_run_"):
    """root/<prefix><N>, N = highest existing <prefix><N> + 1, or 1 if there is none."""
    root = Path(root)
    n = len(prefix)
    indices = [int(d.name[n:]) for d in root.glob(prefix + "*") if d.is_dir() and d.name[n:].isdigit()]
    return root / f"{prefix}{max(indices, default=0) + 1}"


def phase_ticks(sequence, rate_hz=COMMAND_RATE_HZ):
    """[(name, number of commands, angular velocity)]: each phase lasts round(duration * rate) ticks."""
    return [(name, int(round(duration * rate_hz)), w) for name, duration, w in sequence]


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def main():
    parser = argparse.ArgumentParser(description="Open-loop angular-velocity step response, rotating in place.")
    parser.add_argument("--namespace", default="robot1", help="Robot namespace (default: %(default)s)")
    parser.add_argument("--max-angular", type=float, default=0.6,
                        help="Clamp on every angular command, rad/s (default: %(default)s)")
    parser.add_argument("--max-rotation", type=float, default=2 * math.pi,
                        help="Abort if |net rotation from the start| exceeds this, rad (default: 2 pi; the "
                             "planned sequence peaks near 4.2 rad)")
    parser.add_argument("--max-translation", type=float, default=0.25,
                        help="Abort if the robot moves more than this from the start, m (default: %(default)s)")
    parser.add_argument("--max-duration", type=float, default=60.0,
                        help="Abort if the run lasts longer than this, s (default: %(default)s)")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent / "runs"),
                        help="Run folders are created here (default: runs/ next to this script)")
    args = parser.parse_args()

    phases = phase_ticks(SEQUENCE)
    total_ticks = sum(n for _, n, _ in phases)
    planned_s = total_ticks / COMMAND_RATE_HZ
    if any(n <= 0 for _, n, _ in phases):
        parser.error("every phase must last at least one command period (0.2 s)")
    if args.max_duration <= planned_s:
        parser.error(f"--max-duration must exceed the planned {planned_s:.1f} s")
    clamped = [name for name, _, w in phases if abs(w) > args.max_angular]
    if clamped:
        print(f"WARNING: phases {clamped} exceed --max-angular {args.max_angular} rad/s and will be clamped.")

    ns = "/" + args.namespace.strip("/")
    cmd_topic, odom_topic = f"{ns}/cmd_vel_unstamped", f"{ns}/odom"

    # Keep the ROS context alive on Ctrl+C / SIGTERM so the final zero commands can still be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = Node("angular_step_response")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    pub = node.create_publisher(Twist, cmd_topic, QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT))

    state = {"t0": None, "start": None, "odom": None, "received": None, "count": 0, "path_rotation": 0.0,
             "cmd_writer": None, "odom_writer": None, "files": []}

    def ros_now():
        return node.get_clock().now().nanoseconds * 1e-9

    def rotation(o):
        return o["yaw_unwrapped"] - state["start"][3] if state["start"] is not None else 0.0

    def on_odom(msg):
        now = time.monotonic()
        p, t = msg.pose.pose.position, msg.twist.twist
        yaw = yaw_of(msg.pose.pose.orientation)
        prev = state["odom"]
        step = wrap_angle(yaw - prev["yaw"]) if prev else 0.0
        o = {"x": p.x, "y": p.y, "yaw": yaw, "yaw_unwrapped": (prev["yaw_unwrapped"] + step) if prev else yaw,
             "v": t.linear.x, "w": t.angular.z, "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9}
        state["odom"], state["received"] = o, now
        state["count"] += 1
        if state["odom_writer"] is not None:
            state["path_rotation"] += abs(step)
            state["odom_writer"].writerow({
                "t_s": now - state["t0"], "ros_time_s": ros_now(), "stamp_s": o["stamp"],
                "x": o["x"], "y": o["y"], "yaw": o["yaw"], "yaw_unwrapped": o["yaw_unwrapped"],
                "rotation_rad": rotation(o), "v_linear": o["v"], "v_angular": o["w"]})

    node.create_subscription(Odometry, odom_topic, on_odom, qos_profile_sensor_data)

    def publish(w, phase):
        w = max(-args.max_angular, min(args.max_angular, float(w)))
        cmd = Twist()
        cmd.linear.x = 0.0                    # never any linear motion
        cmd.angular.z = w
        pub.publish(cmd)
        now = time.monotonic()
        if state["cmd_writer"] is not None:
            o = state["odom"]
            state["cmd_writer"].writerow({
                "t_s": now - state["t0"], "ros_time_s": ros_now(), "cmd_linear": 0.0, "cmd_angular": w,
                "phase": phase, "yaw": o["yaw"] if o else "", "rotation_rad": rotation(o) if o else "",
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

    def check_safety():
        age = time.monotonic() - state["received"]
        if age > ODOM_TIMEOUT_S:
            raise Abort(f"odometry is {age:.2f} s old")
        if time.monotonic() - state["t0"] > args.max_duration:
            raise Abort(f"run exceeded {args.max_duration:.0f} s")
        o, (x0, y0, _, _) = state["odom"], state["start"]
        if abs(rotation(o)) > args.max_rotation:
            raise Abort(f"net rotation {rotation(o):+.2f} rad exceeds {args.max_rotation:.2f} rad")
        if math.hypot(o["x"] - x0, o["y"] - y0) > args.max_translation:
            raise Abort(f"robot moved more than {args.max_translation} m from the start")

    def pose(o):
        return {"x": o["x"], "y": o["y"], "yaw": o["yaw"], "yaw_unwrapped": o["yaw_unwrapped"]}

    info = {"experiment": "angular_step_response", "namespace": args.namespace, "cmd_topic": cmd_topic,
            "odom_topic": odom_topic, "cmd_qos": "best_effort, keep last 10", "odom_qos": "sensor_data",
            "command_rate_hz": COMMAND_RATE_HZ, "sequence": SEQUENCE,
            "phase_ticks": [[name, n, w] for name, n, w in phases], "planned_duration_s": planned_s,
            "cmd_linear_mps": 0.0, "max_angular_radps": args.max_angular, "max_rotation_rad": args.max_rotation,
            "max_translation_m": args.max_translation, "max_duration_s": args.max_duration,
            "clock": "t_s = time.monotonic() - start of recording", "outcome": "NOT_STARTED"}
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
        state["start"] = (o["x"], o["y"], o["yaw"], o["yaw_unwrapped"])
        info["start_pose"] = pose(o)
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
        print(f"Start pose ({o['x']:.3f}, {o['y']:.3f}), yaw {math.degrees(o['yaw']):.1f} deg. "
              f"Planned {planned_s:.1f} s. Logging to {run_dir}")

        # 3. Time-scheduled sequence: phase k covers ticks [sum of earlier phases, + its own count). A late
        # tick skips grid points, so the phase is taken from the grid index, keeping phases on schedule.
        boundaries, total = [], 0
        for name, n, w in phases:
            total += n
            boundaries.append((total, name, w))
        t_next, phase = state["t0"], None
        while True:
            grid_index = int(round((t_next - state["t0"]) * COMMAND_RATE_HZ))
            if grid_index >= total_ticks:
                break
            t_next = tick(t_next)
            check_safety()
            _, name, w = next(b for b in boundaries if grid_index < b[0])
            if name != phase:
                print(f"[{time.monotonic() - state['t0']:6.2f} s] {name} ({w:+.2f} rad/s), "
                      f"rotation so far {rotation(state['odom']):+.2f} rad")
                phase = name
            publish(w, name)

        o = state["odom"]
        info["outcome"] = "COMPLETED"
        print(f"[DONE] net rotation {rotation(o):+.3f} rad, translation "
              f"{math.hypot(o['x'] - state['start'][0], o['y'] - state['start'][1]):.3f} m")
        return 0
    except Abort as exc:
        info["outcome"] = f"ABORTED: {exc}"
        info["abort_reason"] = str(exc)
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
        if state["start"] is not None and state["odom"] is not None:
            o = state["odom"]
            info["final_pose"] = pose(o)
            info["net_rotation_rad"] = rotation(o)
            info["total_rotation_traveled_rad"] = state["path_rotation"]
            info["final_displacement_m"] = math.hypot(o["x"] - state["start"][0], o["y"] - state["start"][1])
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
