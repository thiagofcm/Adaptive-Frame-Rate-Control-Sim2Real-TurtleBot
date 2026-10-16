#!/usr/bin/env python3
"""Fixed sensing-rate real-robot evaluation with passive episode recording."""

import argparse
import copy
import math
import os
import time
from pathlib import Path

import numpy as np

import rclpy
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.srv import DrlStep, Goal

# The upstream model loader expects this variable during module import. Keep
# direct script execution self-contained when the Docker shell did not set it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DRLNAV_BASE_PATH", str(REPOSITORY_ROOT))

from turtlebot3_drl.common import utilities as util
from turtlebot3_drl.common.settings import (
    ENABLE_BACKWARD,
    ENABLE_STACKING,
    EPISODE_TIMEOUT_SECONDS,
    REAL_N_SCAN_SAMPLES,
    REAL_SPEED_ANGULAR_MAX,
    REAL_SPEED_LINEAR_MAX,
    REAL_THRESHOLD_COLLISION,
    REAL_THRESHOLD_GOAL,
    REAL_TOPIC_ODOM,
    REAL_TOPIC_VELO,
    TIMEOUT,
    UNKNOWN,
)
from turtlebot3_drl.common.storagemanager import StorageManager
from turtlebot3_drl.drl_agent.td3 import TD3
from turtlebot3_drl.drl_environment.drl_environment_real import (
    real_scan_beam_indices,
)

from real_episode_recorder import RealEpisodeRecorder


RAW_SCAN_TOPIC = "/robot1/scan"
GOAL_TOPIC = "/goal_pose"
RECORD_INTERVAL_S = 0.1


def absolute_topic(name):
    return "/" + name.lstrip("/")


def default_output_root():
    return str(Path(os.environ["DRLNAV_BASE_PATH"]) / "RealRobot" / "eval")


class RealFixedEvaluator(Node):
    """Own frozen TD3 inference and a deadline-based fixed-rate scan gate."""

    def __init__(
        self,
        fixed_fps,
        policy_name,
        output_root,
        timeout_seconds,
        nav_model_run,
        nav_episode,
        raw_scan_topic,
        gated_scan_topic,
        odom_topic,
        goal_topic,
        cmd_vel_topic,
        step_service,
        goal_service,
    ):
        super().__init__("real_fixed_fps_evaluator")

        if fixed_fps <= 0.0:
            raise ValueError("fixed_fps must be greater than zero")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be greater than zero")
        if raw_scan_topic == gated_scan_topic:
            raise ValueError("raw and gated scan topics must be different")
        if ENABLE_STACKING:
            raise RuntimeError(
                "This evaluator assumes ENABLE_STACKING=False, matching the "
                "current real evaluation setup."
            )

        self.fixed_fps = float(fixed_fps)
        self.current_fps = self.fixed_fps
        self.period = 1.0 / self.current_fps
        self.timeout_seconds = float(timeout_seconds)
        self.raw_scan_topic = absolute_topic(raw_scan_topic)
        self.gated_scan_topic = absolute_topic(gated_scan_topic)
        self.odom_topic = absolute_topic(odom_topic)
        self.goal_topic = absolute_topic(goal_topic)
        self.cmd_vel_topic = absolute_topic(cmd_vel_topic)

        self.next_sample_time = None
        self.last_forward_time = None
        self.force_next_fresh = True
        self.raw_scan_count = 0
        self.gated_scan_count = 0
        self.latest_goal = None
        self.goal_message_count = 0
        self.consumed_goal_message_count = 0
        self.record_cmd_linear = 0.0
        self.record_cmd_angular = 0.0
        self.record_outcome = UNKNOWN
        self.record_done = False

        configuration = {
            "goal_threshold": REAL_THRESHOLD_GOAL,
            "collision_threshold": REAL_THRESHOLD_COLLISION,
            "timeout_seconds": self.timeout_seconds,
            "odom_topic": self.odom_topic,
            "goal_topic": self.goal_topic,
            "cmd_vel_topic": self.cmd_vel_topic,
            "raw_scan_topic": self.raw_scan_topic,
            "gated_scan_topic": self.gated_scan_topic,
            "fixed_requested_fps": self.fixed_fps,
            "navigation_model_run": nav_model_run,
            "navigation_model_episode": int(nav_episode),
            "record_interval_s": RECORD_INTERVAL_S,
        }
        self.recorder = RealEpisodeRecorder(
            output_root=output_root,
            policy_name=policy_name,
            policy_type="fixed",
            configuration=configuration,
        )

        self.step_comm_client = self.create_client(DrlStep, step_service)
        self.goal_comm_client = self.create_client(Goal, goal_service)
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            QoSProfile(depth=10),
        )
        self.gated_pub = self.create_publisher(
            LaserScan,
            self.gated_scan_topic,
            QoSProfile(depth=10),
        )
        self.raw_scan_sub = self.create_subscription(
            LaserScan,
            self.raw_scan_topic,
            self._on_raw_scan_recorded,
            qos_profile_sensor_data,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._on_odom,
            QoSProfile(depth=10),
        )
        self.goal_sub = self.create_subscription(
            Pose,
            self.goal_topic,
            self._on_goal,
            QoSProfile(depth=10),
        )
        self.record_timer = self.create_timer(
            RECORD_INTERVAL_S,
            self._record_timer_callback,
        )
        self.record_timer.cancel()

        self.device = util.check_gpu()
        self.sm = StorageManager(
            "td3",
            nav_model_run,
            nav_episode,
            self.device,
            util.stage,
        )
        self.nav_model = self.sm.load_model()
        self.nav_model.device = self.device
        self.sm.load_weights(self.nav_model.networks)

        if not isinstance(self.nav_model, TD3):
            raise TypeError(f"Expected TD3 navigation model, got {type(self.nav_model)}")
        if self.nav_model.state_size != REAL_N_SCAN_SAMPLES + 4:
            raise ValueError(
                "Navigation state-size mismatch: "
                f"model={self.nav_model.state_size}, "
                f"real environment expected={REAL_N_SCAN_SAMPLES + 4}"
            )

        self.get_logger().info(
            f"Fixed evaluator ready: requested_fps={self.fixed_fps:g}, "
            f"timeout={self.timeout_seconds:g}s, output={self.recorder.policy_dir}"
        )

    # ------------------------------------------------------------------
    # Passive recording inputs
    # ------------------------------------------------------------------

    def _on_odom(self, msg):
        _, _, yaw = util.euler_from_quaternion(msg.pose.pose.orientation)
        stamp = msg.header.stamp
        sample_id = (int(stamp.sec), int(stamp.nanosec))
        self.recorder.update_pose(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
            sample_id=sample_id,
        )

    def _on_goal(self, msg):
        self.latest_goal = (float(msg.position.x), float(msg.position.y))
        self.goal_message_count += 1
        self.recorder.update_goal(*self.latest_goal)

    def _set_record_state(self, cmd_linear, cmd_angular, outcome, done):
        self.record_cmd_linear = float(cmd_linear)
        self.record_cmd_angular = float(cmd_angular)
        self.record_outcome = int(outcome)
        self.record_done = bool(done)

    def _record_current_sample(self):
        if not self.recorder.active:
            return
        self.recorder.log_sample(
            self.record_cmd_linear,
            self.record_cmd_angular,
            self.current_fps,
            outcome=self.record_outcome,
            done=self.record_done,
        )

    def _record_timer_callback(self):
        self._record_current_sample()

    # ------------------------------------------------------------------
    # Validated deadline-based scan gate
    # ------------------------------------------------------------------

    def _forward_scan(self, msg, now):
        self.gated_pub.publish(msg)
        self.gated_scan_count += 1
        self.last_forward_time = now
        self.recorder.note_fresh_observation(now)

    def _on_raw_scan_recorded(self, msg):
        """Raw-scan subscription callback: run the gate (unchanged), then
        record this native scan with whether the gate forwarded it."""
        now = time.monotonic()
        gated_before = self.gated_scan_count
        self._on_raw_scan(msg)
        if self.recorder.lidar_recording:
            self.recorder.note_scan(
                now,
                msg,
                real_scan_beam_indices(msg),
                forwarded=self.gated_scan_count > gated_before,
            )

    def _on_raw_scan(self, msg):
        now = time.monotonic()
        self.raw_scan_count += 1

        if self.force_next_fresh:
            self._forward_scan(msg, now)
            self.force_next_fresh = False
            self.next_sample_time = now + self.period
            return

        if self.next_sample_time is None:
            self.next_sample_time = now

        if now < self.next_sample_time:
            return

        self._forward_scan(msg, now)
        while self.next_sample_time <= now:
            self.next_sample_time += self.period

    def reset_scan_gate_for_episode(self):
        self.current_fps = self.fixed_fps
        self.period = 1.0 / self.current_fps
        self.next_sample_time = None
        self.force_next_fresh = True

    # ------------------------------------------------------------------
    # Environment communication
    # ------------------------------------------------------------------

    def wait_for_services(self):
        self.get_logger().info("Waiting for real_environment services...")
        while rclpy.ok() and not self.step_comm_client.wait_for_service(1.0):
            self.get_logger().info("Waiting for step_comm...")
        while rclpy.ok() and not self.goal_comm_client.wait_for_service(1.0):
            self.get_logger().info("Waiting for goal_comm...")

    def _wait_future(self, future):
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.01)
        if not future.done():
            raise RuntimeError("ROS future did not complete")
        if future.exception() is not None:
            raise future.exception()
        return future.result()

    def goal_is_ready(self):
        return bool(
            self._wait_future(
                self.goal_comm_client.call_async(Goal.Request())
            ).new_goal
        )

    def wait_for_manual_goal(self):
        self.get_logger().info(
            f"Waiting for a new manual goal on {self.goal_topic}..."
        )
        while rclpy.ok():
            if self.goal_is_ready():
                deadline = time.monotonic() + 1.0
                while (
                    self.goal_message_count
                    <= self.consumed_goal_message_count
                    and time.monotonic() < deadline
                ):
                    rclpy.spin_once(self, timeout_sec=0.05)
                if (
                    self.latest_goal is None
                    or self.goal_message_count
                    <= self.consumed_goal_message_count
                ):
                    self.get_logger().warning(
                        "Ignored a stale goal_comm flag because no new goal "
                        f"message was received on {self.goal_topic} after "
                        "this evaluator started/waited for the next episode."
                    )
                    continue
                self.consumed_goal_message_count = self.goal_message_count
                return self.latest_goal
            time.sleep(0.05)
        raise KeyboardInterrupt

    def call_step(self, action, previous_action):
        request = DrlStep.Request()
        request.action = [float(value) for value in action]
        request.previous_action = [float(value) for value in previous_action]
        response = self._wait_future(self.step_comm_client.call_async(request))
        return (
            np.asarray(response.state, dtype=np.float32),
            float(response.reward),
            bool(response.done),
            int(response.success),
            float(response.distance_traveled),
        )

    def init_episode(self):
        return self.call_step([], [0.0, 0.0])

    def wait_for_initial_scan(self):
        before = self.gated_scan_count
        deadline = time.monotonic() + 3.0
        while rclpy.ok() and self.gated_scan_count <= before:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Timed out waiting for a raw LiDAR scan on {self.raw_scan_topic}"
                )
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_odometry(self):
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and self.recorder.latest_pose is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Timed out waiting for odometry on {self.odom_topic}"
                )
            rclpy.spin_once(self, timeout_sec=0.05)

    # ------------------------------------------------------------------
    # Evaluator-owned control and termination
    # ------------------------------------------------------------------

    @staticmethod
    def action_to_command(action):
        if ENABLE_BACKWARD:
            linear = float(action[0]) * REAL_SPEED_LINEAR_MAX
        else:
            linear = (float(action[0]) + 1.0) * 0.5 * REAL_SPEED_LINEAR_MAX
        angular = float(action[1]) * REAL_SPEED_ANGULAR_MAX
        return linear, angular

    def safe_stop(self):
        zero = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)

    def run_episode(self):
        goal = self.wait_for_manual_goal()
        self.reset_scan_gate_for_episode()
        self.wait_for_initial_scan()
        time.sleep(0.10)

        nav_state, _, done, outcome, _ = self.init_episode()
        if done:
            raise RuntimeError(
                "real_environment returned done=True on episode initialization"
            )

        self.wait_for_odometry()
        time.sleep(0.5)
        self.recorder.start_episode(
            goal=goal,
            initial_fresh_time=self.last_forward_time,
            ros_time_origin=self.get_clock().now().nanoseconds * 1e-9,
        )
        self._set_record_state(0.0, 0.0, UNKNOWN, False)
        self._record_current_sample()
        self.record_timer.reset()

        previous_action = [0.0, 0.0]
        nav_step = 0

        while rclpy.ok() and not done:
            if (
                time.monotonic() - self.recorder.start_monotonic
                >= self.timeout_seconds
            ):
                self.record_timer.cancel()
                self.safe_stop()
                self._set_record_state(0.0, 0.0, TIMEOUT, True)
                self._record_current_sample()
                return self.recorder.finish_episode(TIMEOUT)

            nav_step += 1
            action = self.nav_model.get_action(
                nav_state,
                False,
                nav_step,
                False,
            )
            cmd_linear, cmd_angular = self.action_to_command(action)
            nav_state, _, done, outcome, _ = self.call_step(
                action,
                previous_action,
            )
            previous_action = copy.deepcopy(action)
            self._set_record_state(cmd_linear, cmd_angular, outcome, done)

            if done:
                self.record_timer.cancel()
                self._record_current_sample()

            if not done:
                step_time = float(getattr(self.nav_model, "step_time", 0.0))
                if step_time > 0.0:
                    time.sleep(step_time)

        self.safe_stop()
        return self.recorder.finish_episode(outcome)


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-FPS real-robot evaluator with passive recording"
    )
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--nav-model-run", default="examples/td3_0_stage9")
    parser.add_argument("--nav-episode", type=int, default=7400)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--timeout",
        type=float,
        default=EPISODE_TIMEOUT_SECONDS,
        help="Evaluator-owned episode timeout in seconds",
    )
    parser.add_argument("--output-root", default=default_output_root())
    parser.add_argument("--raw-scan-topic", default=RAW_SCAN_TOPIC)
    parser.add_argument("--gated-scan-topic", default=absolute_topic("robot1/scan_gated"))
    parser.add_argument("--odom-topic", default=absolute_topic(REAL_TOPIC_ODOM))
    parser.add_argument("--goal-topic", default=GOAL_TOPIC)
    parser.add_argument("--cmd-vel-topic", default=absolute_topic(REAL_TOPIC_VELO))
    parser.add_argument("--step-service", default="/step_comm")
    parser.add_argument("--goal-service", default="/goal_comm")
    args, ros_args = parser.parse_known_args()

    if args.episodes <= 0:
        raise SystemExit("--episodes must be greater than zero")

    rclpy.init(args=ros_args)
    node = RealFixedEvaluator(
        fixed_fps=args.fps,
        policy_name=args.policy,
        output_root=args.output_root,
        timeout_seconds=args.timeout,
        nav_model_run=args.nav_model_run,
        nav_episode=args.nav_episode,
        raw_scan_topic=args.raw_scan_topic,
        gated_scan_topic=args.gated_scan_topic,
        odom_topic=args.odom_topic,
        goal_topic=args.goal_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        step_service=args.step_service,
        goal_service=args.goal_service,
    )

    try:
        node.wait_for_services()
        for index in range(args.episodes):
            node.run_episode()
            if index + 1 < args.episodes:
                print("\nEpisode complete. Publish a new goal for the next episode.\n")
    except KeyboardInterrupt:
        print("\nCtrl+C received: evaluator is stopping the physical robot.")
    finally:
        if node.recorder.active:
            node.recorder.abort_episode("Ctrl+C or evaluator shutdown")
        try:
            node.safe_stop()
        except Exception as exc:
            print(f"WARNING: final safe-stop failed: {exc}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
