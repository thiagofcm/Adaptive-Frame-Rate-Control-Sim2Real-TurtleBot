#!/usr/bin/env python3
"""Open-loop combined linear + angular step response of the TurtleBot4 (Experiment 3).

Companion of actuator_step_response.py (linear) and angular_step_response.py (angular): same ROS setup,
5 Hz command grid, native-rate odometry, CSV layout and safety handling. Linear and angular velocity are
commanded together, on a TIME-scheduled sequence chosen with --mode:

  constant_curvature (default), 27 s, 135 commands:
    phase            duration  v (m/s)  w (rad/s)  ideal radius
    C1_BASELINE        2 s      0.00     0.0
    C2_LEFT_R050       4 s      0.15    +0.3       0.50 m
    C3_STOP            2 s      0.00     0.0
    C4_RIGHT_R050      4 s      0.15    -0.3       0.50 m
    C5_STOP            2 s      0.00     0.0
    C6_LEFT_R030       4 s      0.15    +0.5       0.30 m
    C7_STOP            2 s      0.00     0.0
    C8_RIGHT_R030      4 s      0.15    -0.5       0.30 m
    C9_FINAL_STOP      3 s      0.00     0.0
  Ideal unicycle path from (0, 0, 0): each left/right pair cancels the heading but not the position, so
  the robot ends at about (1.48, 1.49) m, heading 0, 2.10 m from the start (forward and to the LEFT).

  dynamic, 12 s, 60 commands (a 2 s still baseline is added before the requested sequence):
    D0_BASELINE 2 s 0/0; D1 2 s 0.10/+0.2; D2 2 s 0.15/+0.4; D3 2 s 0.20/-0.4; D4 2 s 0.10/-0.2; D5_STOP 2 s 0/0
  Ideal end about (0.86, 0.57) m, heading 0, 1.03 m from the start.

Edit SEQUENCES below to change them. Each phase lasts round(duration * 5) commands on the 5 Hz grid.
Odometry is only used for safety checks: no heading, speed or position feedback, no drift correction.

Start frame: x_s is forward and y_s is LEFT of the starting pose, rotation_rad is the unwrapped heading
change. The ideal trajectory of the selected sequence is computed at start-up (exact unicycle arcs) and
gives the safe region: its bounding box in the start frame grown by --margin on every side, and a circle
of radius (largest ideal distance from the start + --margin). The robot must stay inside both.

Timing: t_s in both CSV files is time.monotonic() minus the same origin (start of recording), taken right
after publish() for commands and on callback entry for odometry. ros_time_s is the PC's ROS clock at the
same instants; odometry rows also keep the message's header stamp (the robot's clock).

Safety: motion starts only after a matched velocity subscriber and fresh odometry; commands are clamped to
--max-linear and --max-angular; the run aborts on odometry older than 0.5 s, a run longer than the plan +
--duration-margin, leaving the safe region (box and radius), or |rotation| above the largest ideal heading
change + --rotation-margin; every exit (finish, abort, exception, Ctrl+C, SIGTERM) publishes zero velocity.
Other publishers on the velocity topic are reported, not prevented.

Output: RealRobot/motion_dynamic_charac/runs/combined_run_<N>/commands.csv, odometry.csv, run_info.json

Usage (inside the tb3_drl container, ROS sourced, ROS_DOMAIN_ID=5 as the robot):
    python3 RealRobot/motion_dynamic_charac/combined_motion_response.py --namespace robot1
    python3 RealRobot/motion_dynamic_charac/combined_motion_response.py --namespace robot1 --mode dynamic
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

# (phase name, duration s, linear velocity m/s, angular velocity rad/s); positive w = counter-clockwise
SEQUENCES = {
    "constant_curvature": [
        ("C1_BASELINE", 2.0, 0.00, 0.0),
        ("C2_LEFT_R050", 4.0, 0.15, 0.3),
        ("C3_STOP", 2.0, 0.00, 0.0),
        ("C4_RIGHT_R050", 4.0, 0.15, -0.3),
        ("C5_STOP", 2.0, 0.00, 0.0),
        ("C6_LEFT_R030", 4.0, 0.15, 0.5),
        ("C7_STOP", 2.0, 0.00, 0.0),
        ("C8_RIGHT_R030", 4.0, 0.15, -0.5),
        ("C9_FINAL_STOP", 3.0, 0.00, 0.0),
    ],
    "dynamic": [
        ("D0_BASELINE", 2.0, 0.00, 0.0),
        ("D1_V010_W+02", 2.0, 0.10, 0.2),
        ("D2_V015_W+04", 2.0, 0.15, 0.4),
        ("D3_V020_W-04", 2.0, 0.20, -0.4),
        ("D4_V010_W-02", 2.0, 0.10, -0.2),
        ("D5_STOP", 2.0, 0.00, 0.0),
    ],
}
COMMAND_RATE_HZ = 5.0         # TD3 navigation rate
STOP_REPEATS = 10             # zero commands on exit
DISCOVERY_TIMEOUT_S = 15.0
ODOM_WAIT_S = 3.0
ODOM_TIMEOUT_S = 0.5

COMMAND_FIELDS = ["t_s", "ros_time_s", "cmd_linear", "cmd_angular", "phase", "v_linear", "v_angular",
                  "x", "y", "yaw", "x_s", "y_s", "rotation_rad", "odom_age_s"]
ODOM_FIELDS = ["t_s", "ros_time_s", "stamp_s", "x", "y", "yaw", "yaw_unwrapped", "x_s", "y_s", "rotation_rad",
               "v_linear", "v_angular"]


class Abort(Exception):
    pass


def next_run_dir(root, prefix="combined_run_"):
    """root/<prefix><N>, N = highest existing <prefix><N> + 1, or 1 if there is none."""
    root = Path(root)
    n = len(prefix)
    indices = [int(d.name[n:]) for d in root.glob(prefix + "*") if d.is_dir() and d.name[n:].isdigit()]
    return root / f"{prefix}{max(indices, default=0) + 1}"


def phase_ticks(sequence, rate_hz=COMMAND_RATE_HZ):
    """[(name, number of commands, v, w)]: each phase lasts round(duration * rate) ticks."""
    return [(name, int(round(duration * rate_hz)), v, w) for name, duration, v, w in sequence]


def ideal_trajectory(phases, rate_hz=COMMAND_RATE_HZ, substeps=20):
    """Ideal unicycle from (0, 0, 0) under the commanded (v, w), each held for whole command periods.

    Exact arcs, sampled `substeps` times per period. Returns (phase-end poses, max distance from the start,
    bounding box (xmin, xmax, ymin, ymax), max |heading|), all in the start frame.
    """
    x = y = th = 0.0
    xmin = xmax = ymin = ymax = rmax = thmax = 0.0
    ends = []
    dt = 1.0 / (rate_hz * substeps)
    for name, n, v, w in phases:
        for _ in range(n * substeps):
            if abs(w) > 1e-12:
                x += v / w * (math.sin(th + w * dt) - math.sin(th))
                y -= v / w * (math.cos(th + w * dt) - math.cos(th))
            else:
                x += v * math.cos(th) * dt
                y += v * math.sin(th) * dt
            th += w * dt
            xmin, xmax, ymin, ymax = min(xmin, x), max(xmax, x), min(ymin, y), max(ymax, y)
            rmax, thmax = max(rmax, math.hypot(x, y)), max(thmax, abs(th))
        ends.append({"phase": name, "x": x, "y": y, "theta_rad": th, "theta_deg": math.degrees(th),
                     "distance_m": math.hypot(x, y)})
    return ends, rmax, (xmin, xmax, ymin, ymax), thmax


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def main():
    parser = argparse.ArgumentParser(description="Open-loop combined linear and angular step response.")
    parser.add_argument("--namespace", default="robot1", help="Robot namespace (default: %(default)s)")
    parser.add_argument("--mode", choices=sorted(SEQUENCES), default="constant_curvature",
                        help="Command sequence (default: %(default)s)")
    parser.add_argument("--max-linear", type=float, default=0.20,
                        help="Clamp on every linear command, m/s (default: %(default)s)")
    parser.add_argument("--max-angular", type=float, default=0.6,
                        help="Clamp on every angular command, rad/s (default: %(default)s)")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Safe region = ideal trajectory's bounding box and radius grown by this, m "
                             "(default: %(default)s)")
    parser.add_argument("--max-radius", type=float, default=None,
                        help="Override the safe radius from the start, m (default: ideal max distance + margin)")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"), default=None,
                        help="Override the safe box in the start frame (x forward, y left), m "
                             "(default: ideal bounding box grown by --margin)")
    parser.add_argument("--rotation-margin", type=float, default=1.0,
                        help="Abort if |rotation| exceeds the ideal maximum heading change + this, rad "
                             "(default: %(default)s)")
    parser.add_argument("--duration-margin", type=float, default=10.0,
                        help="Abort if the run lasts longer than the plan + this, s (default: %(default)s)")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent / "runs"),
                        help="Run folders are created here (default: runs/ next to this script)")
    args = parser.parse_args()

    sequence = SEQUENCES[args.mode]
    phases = phase_ticks(sequence)
    total_ticks = sum(n for _, n, _, _ in phases)
    planned_s = total_ticks / COMMAND_RATE_HZ
    if any(n <= 0 for _, n, _, _ in phases):
        parser.error("every phase must last at least one command period (0.2 s)")
    clamped = [name for name, _, v, w in phases if abs(v) > args.max_linear or abs(w) > args.max_angular]
    if clamped:
        print(f"WARNING: phases {clamped} exceed --max-linear/--max-angular and will be clamped.")

    # Ideal trajectory of the clamped commands -> safe region in the start frame.
    clamp_v = lambda v: max(-args.max_linear, min(args.max_linear, float(v)))       # noqa: E731
    clamp_w = lambda w: max(-args.max_angular, min(args.max_angular, float(w)))     # noqa: E731
    ideal_ends, ideal_rmax, ideal_box, ideal_thmax = ideal_trajectory(
        [(name, n, clamp_v(v), clamp_w(w)) for name, n, v, w in phases])
    safe_radius = args.max_radius if args.max_radius is not None else ideal_rmax + args.margin
    safe_box = tuple(args.bbox) if args.bbox is not None else (
        ideal_box[0] - args.margin, ideal_box[1] + args.margin, ideal_box[2] - args.margin, ideal_box[3] + args.margin)
    max_rotation = ideal_thmax + args.rotation_margin
    max_duration = planned_s + args.duration_margin
    if safe_radius <= ideal_rmax or not (safe_box[0] < ideal_box[0] and safe_box[1] > ideal_box[1]
                                         and safe_box[2] < ideal_box[2] and safe_box[3] > ideal_box[3]):
        parser.error(f"the safe region must contain the ideal trajectory: radius > {ideal_rmax:.2f} m and "
                     f"box beyond x [{ideal_box[0]:.2f}, {ideal_box[1]:.2f}], y [{ideal_box[2]:.2f}, {ideal_box[3]:.2f}]")

    print(f"Mode {args.mode}: {len(phases)} phases, {total_ticks} commands, {planned_s:.1f} s planned.")
    print("Ideal path (start frame, x forward, y left):")
    for e in ideal_ends:
        print(f"  end of {e['phase']:14s} x {e['x']:+.3f}  y {e['y']:+.3f}  heading {e['theta_deg']:+7.1f} deg"
              f"  distance {e['distance_m']:.3f} m")
    print(f"Safe region: box x [{safe_box[0]:.2f}, {safe_box[1]:.2f}] y [{safe_box[2]:.2f}, {safe_box[3]:.2f}] m, "
          f"radius {safe_radius:.2f} m, |rotation| <= {math.degrees(max_rotation):.0f} deg, run <= {max_duration:.0f} s")

    ns = "/" + args.namespace.strip("/")
    cmd_topic, odom_topic = f"{ns}/cmd_vel_unstamped", f"{ns}/odom"

    # Keep the ROS context alive on Ctrl+C / SIGTERM so the final zero commands can still be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = Node("combined_motion_response")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    pub = node.create_publisher(Twist, cmd_topic, QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT))

    state = {"t0": None, "start": None, "odom": None, "received": None, "count": 0, "max_distance": 0.0,
             "cmd_writer": None, "odom_writer": None, "files": []}

    def ros_now():
        return node.get_clock().now().nanoseconds * 1e-9

    def start_frame(o):
        """(x_s forward, y_s left, rotation) relative to the starting pose; zeros before the start."""
        if state["start"] is None:
            return 0.0, 0.0, 0.0
        x0, y0, yaw0, unwrapped0 = state["start"]
        dx, dy = o["x"] - x0, o["y"] - y0
        return (dx * math.cos(yaw0) + dy * math.sin(yaw0), -dx * math.sin(yaw0) + dy * math.cos(yaw0),
                o["yaw_unwrapped"] - unwrapped0)

    def on_odom(msg):
        now = time.monotonic()
        p, t = msg.pose.pose.position, msg.twist.twist
        yaw = yaw_of(msg.pose.pose.orientation)
        prev = state["odom"]
        o = {"x": p.x, "y": p.y, "yaw": yaw,
             "yaw_unwrapped": (prev["yaw_unwrapped"] + wrap_angle(yaw - prev["yaw"])) if prev else yaw,
             "v": t.linear.x, "w": t.angular.z, "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9}
        state["odom"], state["received"] = o, now
        state["count"] += 1
        if state["odom_writer"] is not None:
            xs, ys, rot = start_frame(o)
            state["max_distance"] = max(state["max_distance"], math.hypot(xs, ys))
            state["odom_writer"].writerow({
                "t_s": now - state["t0"], "ros_time_s": ros_now(), "stamp_s": o["stamp"],
                "x": o["x"], "y": o["y"], "yaw": o["yaw"], "yaw_unwrapped": o["yaw_unwrapped"],
                "x_s": xs, "y_s": ys, "rotation_rad": rot, "v_linear": o["v"], "v_angular": o["w"]})

    node.create_subscription(Odometry, odom_topic, on_odom, qos_profile_sensor_data)

    def publish(v, w, phase):
        v, w = clamp_v(v), clamp_w(w)
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        pub.publish(cmd)
        now = time.monotonic()
        if state["cmd_writer"] is not None:
            o = state["odom"]
            row = {"t_s": now - state["t0"], "ros_time_s": ros_now(), "cmd_linear": v, "cmd_angular": w,
                   "phase": phase, "odom_age_s": now - state["received"] if state["received"] else ""}
            if o:
                xs, ys, rot = start_frame(o)
                row.update({"v_linear": o["v"], "v_angular": o["w"], "x": o["x"], "y": o["y"], "yaw": o["yaw"],
                            "x_s": xs, "y_s": ys, "rotation_rad": rot})
            state["cmd_writer"].writerow(row)

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
        if time.monotonic() - state["t0"] > max_duration:
            raise Abort(f"run exceeded {max_duration:.0f} s")
        xs, ys, rot = start_frame(state["odom"])
        if abs(rot) > max_rotation:
            raise Abort(f"rotation {math.degrees(rot):+.0f} deg exceeds {math.degrees(max_rotation):.0f} deg")
        if math.hypot(xs, ys) > safe_radius:
            raise Abort(f"robot is {math.hypot(xs, ys):.2f} m from the start, beyond the safe radius {safe_radius:.2f} m")
        if not (safe_box[0] <= xs <= safe_box[1] and safe_box[2] <= ys <= safe_box[3]):
            raise Abort(f"robot left the safe box at x_s {xs:+.2f} m, y_s {ys:+.2f} m")

    def pose(o):
        xs, ys, rot = start_frame(o)
        return {"x": o["x"], "y": o["y"], "yaw": o["yaw"], "yaw_unwrapped": o["yaw_unwrapped"],
                "x_s": xs, "y_s": ys, "rotation_rad": rot}

    info = {"experiment": "combined_motion_response", "mode": args.mode, "namespace": args.namespace,
            "cmd_topic": cmd_topic, "odom_topic": odom_topic, "cmd_qos": "best_effort, keep last 10",
            "odom_qos": "sensor_data", "command_rate_hz": COMMAND_RATE_HZ, "sequence": sequence,
            "phase_ticks": [[name, n, v, w] for name, n, v, w in phases], "planned_commands": total_ticks,
            "planned_duration_s": planned_s, "max_linear_mps": args.max_linear, "max_angular_radps": args.max_angular,
            "ideal_phase_ends_start_frame": ideal_ends, "ideal_max_distance_m": ideal_rmax,
            "ideal_bbox_start_frame": ideal_box, "ideal_max_rotation_rad": ideal_thmax,
            "safe_box_start_frame": safe_box, "safe_radius_m": safe_radius, "max_rotation_rad": max_rotation,
            "max_duration_s": max_duration, "margin_m": args.margin,
            "start_frame": "x_s forward and y_s left of the starting pose; rotation_rad = unwrapped yaw change",
            "clock": "t_s = time.monotonic() - start of recording", "completed": False, "outcome": "NOT_STARTED",
            "abort_reason": None}
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
        print(f"Start pose ({o['x']:.3f}, {o['y']:.3f}), yaw {math.degrees(o['yaw']):.1f} deg. Logging to {run_dir}")

        # 3. Time-scheduled sequence: phase k covers ticks [sum of earlier phases, + its own count). A late
        # tick skips grid points, so the phase is taken from the grid index, keeping phases on schedule.
        boundaries, total = [], 0
        for name, n, v, w in phases:
            total += n
            boundaries.append((total, name, v, w))
        t_next, phase = state["t0"], None
        while True:
            grid_index = int(round((t_next - state["t0"]) * COMMAND_RATE_HZ))
            if grid_index >= total_ticks:
                break
            t_next = tick(t_next)
            check_safety()
            _, name, v, w = next(b for b in boundaries if grid_index < b[0])
            if name != phase:
                xs, ys, rot = start_frame(state["odom"])
                print(f"[{time.monotonic() - state['t0']:6.2f} s] {name} (v {v:.2f} m/s, w {w:+.2f} rad/s) at "
                      f"x_s {xs:+.3f} y_s {ys:+.3f} m, heading {math.degrees(rot):+.1f} deg")
                phase = name
            publish(v, w, name)

        info["outcome"], info["completed"] = "COMPLETED", True
        xs, ys, rot = start_frame(state["odom"])
        ideal = ideal_ends[-1]
        print(f"[DONE] end at x_s {xs:+.3f} y_s {ys:+.3f} m, heading {math.degrees(rot):+.1f} deg "
              f"(ideal {ideal['x']:+.3f}, {ideal['y']:+.3f} m, {ideal['theta_deg']:+.1f} deg)")
        return 0
    except Abort as exc:
        info["outcome"], info["abort_reason"] = f"ABORTED: {exc}", str(exc)
        print(f"ABORTED: {exc}")
        return 2
    except KeyboardInterrupt:
        info["outcome"], info["abort_reason"] = "INTERRUPTED", "Ctrl+C or SIGTERM"
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        info["outcome"], info["abort_reason"] = f"ERROR: {exc!r}", repr(exc)
        raise
    finally:
        try:
            for _ in range(STOP_REPEATS):
                publish(0.0, 0.0, "EXIT_STOP")
                executor.spin_once(timeout_sec=0.05)
            print(f"Sent {STOP_REPEATS} zero-velocity commands.")
        except Exception as exc:
            print(f"WARNING: zero-velocity commands failed: {exc}")
        info.update({"odometry_messages": state["count"], "command_ticks": timing["ticks"],
                     "late_ticks": timing["late_ticks"], "max_tick_lateness_s": timing["max_late_s"]})
        if state["t0"] is not None:
            info["duration_s"] = time.monotonic() - state["t0"]
            o = state["odom"]
            info["final_pose"] = pose(o)
            info["final_displacement_m"] = math.hypot(info["final_pose"]["x_s"], info["final_pose"]["y_s"])
            info["final_heading_change_rad"] = info["final_pose"]["rotation_rad"]
            info["max_distance_from_start_m"] = state["max_distance"]
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
