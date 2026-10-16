#!/usr/bin/env python3
"""
Real-robot Sim2Real evaluator for the trained recurrent Adaptive-FPS PPO policy.

Runtime architecture
--------------------
/robot1/scan
    -> this script's time-based gate
    -> /robot1/scan_gated
    -> turtlebot3_drl real_environment
    -> DrlStep state (44-D)
    -> frozen TD3 navigation policy
    -> DrlStep action
    -> real_environment publishes cmd_vel

The recurrent PPO adaptive policy runs at 10 Hz wall time.  It observes the
canonical 43-D Gazebo observation (adaptive_obs.py):
    [40 held LiDAR rays, fps_ratio, obs_age_ratio, frame_count_ratio]
and chooses one of:
    [0.2, 0.5, 1.0, 5.0, 10.0] Hz

As in the Gazebo AdaptiveFPSEnv, PPO produces an action every 0.1 s, but that
action becomes causal only if a fresh gated LiDAR frame is consumed during
that PPO interval.  The selected FPS then controls the NEXT sensing interval.

IMPORTANT
---------
1. Do NOT run turtlebot3_drl real_agent at the same time.  This script replaces
   the real_agent process and owns the frozen TD3 inference loop.
2. The real environment must subscribe to /robot1/scan_gated (typically by
   setting REAL_TOPIC_SCAN accordingly).
3. Episode recording is passive.  Robot stopping and timeout handling remain
   evaluator responsibilities.
"""

import argparse
import copy
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.srv import DrlStep, Goal

# The upstream model loader expects this variable during module import. Keep
# direct script execution self-contained when the Docker shell did not set it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DRLNAV_BASE_PATH", str(REPOSITORY_ROOT))

from turtlebot3_drl.common import utilities as util
from turtlebot3_drl.common.storagemanager import StorageManager
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
from turtlebot3_drl.drl_agent.td3 import TD3
from turtlebot3_drl.drl_environment.drl_environment_real import (
    real_scan_beam_indices,
)

from real_episode_recorder import RealEpisodeRecorder
from adaptive_obs import (
    ADAPTIVE_FRAME_BUDGET,
    ADAPTIVE_OBS_DIM,
    ADAPTIVE_OBS_LAYOUT,
    check_adaptive_checkpoint,
)


# ---------------------------------------------------------------------------
# Experiment constants: must match training.
# ---------------------------------------------------------------------------
FPS_CHOICES = [0.2, 0.5, 1.0, 5.0, 10.0]
PPO_RATE_HZ = 10.0
PPO_DT = 1.0 / PPO_RATE_HZ

# Gazebo AdaptiveFPSEnv._max_obs_interval: observation age is counted in PPO
# steps; slowest choice 0.2 Hz -> 10 / 0.2 = 50 PPO steps (5 s).
MAX_OBS_AGE_STEPS = PPO_RATE_HZ / min(FPS_CHOICES)

# The TD3 navigation state stays 44-D. The PPO observation is the canonical
# 43-D one: the 40 held LiDAR rays (first entries of the TD3 state) + 3 features.
NAV_STATE_DIM = REAL_N_SCAN_SAMPLES + 4
PPO_STATE_DIM = REAL_N_SCAN_SAMPLES + 3
if PPO_STATE_DIM != ADAPTIVE_OBS_DIM:
    raise RuntimeError(
        f"REAL_N_SCAN_SAMPLES={REAL_N_SCAN_SAMPLES} gives a {PPO_STATE_DIM}-D "
        f"PPO observation; the checkpoints expect {ADAPTIVE_OBS_DIM}-D "
        f"({ADAPTIVE_OBS_LAYOUT})"
    )

RAW_SCAN_TOPIC = "/robot1/scan"
GOAL_TOPIC = "/goal_pose"
RECORD_INTERVAL_S = 0.1


def absolute_topic(name):
    return "/" + name.lstrip("/")


def default_output_root():
    return str(Path(os.environ["DRLNAV_BASE_PATH"]) / "RealRobot" / "eval")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentEval(nn.Module):
    """
    Inference-only mirror of train_adaptive_fps_ppo.py's recurrent Agent.
    Architecture/submodule names must match the training checkpoint.
    """

    def __init__(self, obs_dim, n_actions, lstm_hidden_size=64):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(64, lstm_hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))

        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h)

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def predict(
        self,
        obs,
        lstm_state,
        done=False,
        deterministic=True,
        return_probs=False,
    ):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        done_tensor = torch.tensor([float(done)], dtype=torch.float32)

        with torch.no_grad():
            hidden, lstm_state = self.get_states(
                obs_tensor, lstm_state, done_tensor
            )
            logits = self.actor(hidden)

            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()

            if return_probs:
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                return int(action.item()), lstm_state, probs

        return int(action.item()), lstm_state


class RealAdaptiveEvaluator(Node):
    def __init__(
        self,
        adaptive_model_path,
        nav_model_run,
        nav_episode,
        raw_scan_topic,
        gated_scan_topic,
        step_service,
        goal_service,
        cmd_vel_topic,
        odom_topic,
        goal_topic,
        policy_name,
        output_root,
        timeout_seconds,
        diagnose_probs=False,
        budget=ADAPTIVE_FRAME_BUDGET,
    ):
        super().__init__("real_adaptive_fps_evaluator")

        if raw_scan_topic == gated_scan_topic:
            raise ValueError(
                "raw_scan_topic and gated_scan_topic must be different. "
                "The real environment should subscribe to the gated topic."
            )

        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be greater than zero")

        self.raw_scan_topic = absolute_topic(raw_scan_topic)
        self.gated_scan_topic = absolute_topic(gated_scan_topic)
        self.odom_topic = absolute_topic(odom_topic)
        self.goal_topic = absolute_topic(goal_topic)
        self.cmd_vel_topic = absolute_topic(cmd_vel_topic)
        self.timeout_seconds = float(timeout_seconds)
        self.diagnose_probs = diagnose_probs
        self.latest_goal = None
        self.goal_message_count = 0
        self.consumed_goal_message_count = 0
        self.record_cmd_linear = 0.0
        self.record_cmd_angular = 0.0
        self.record_outcome = UNKNOWN
        self.record_done = False

        recorder_configuration = {
            "goal_threshold": REAL_THRESHOLD_GOAL,
            "collision_threshold": REAL_THRESHOLD_COLLISION,
            "timeout_seconds": self.timeout_seconds,
            "odom_topic": self.odom_topic,
            "goal_topic": self.goal_topic,
            "cmd_vel_topic": self.cmd_vel_topic,
            "raw_scan_topic": self.raw_scan_topic,
            "gated_scan_topic": self.gated_scan_topic,
            "adaptive_model_path": str(Path(adaptive_model_path).resolve()),
            "fps_action_choices": list(FPS_CHOICES),
            "navigation_model_run": nav_model_run,
            "navigation_model_episode": int(nav_episode),
            "record_interval_s": RECORD_INTERVAL_S,
        }
        self.recorder = RealEpisodeRecorder(
            output_root=output_root,
            policy_name=policy_name,
            policy_type="adaptive",
            configuration=recorder_configuration,
        )

        # ------------------------------------------------------------------
        # ROS communication with the already-running REAL environment.
        # ------------------------------------------------------------------
        self.step_comm_client = self.create_client(DrlStep, step_service)
        self.goal_comm_client = self.create_client(Goal, goal_service)

        self.cmd_vel_pub = self.create_publisher(
            Twist, self.cmd_vel_topic, QoSProfile(depth=10)
        )

        # ------------------------------------------------------------------
        # Integrated real-time LiDAR gate.
        # ------------------------------------------------------------------
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

        self.current_fps = max(FPS_CHOICES)
        self.period = 1.0 / self.current_fps

        self.next_sample_time = None
        self.last_forward_time = None
        self.force_next_fresh = True

        self.raw_scan_count = 0
        self.gated_scan_count = 0

        # ------------------------------------------------------------------
        # Frozen TD3 navigation policy: same StorageManager path as repo agent.
        # ------------------------------------------------------------------
        if ENABLE_STACKING:
            raise RuntimeError(
                "This evaluator currently assumes ENABLE_STACKING=False, "
                "matching the adaptive Gazebo setup."
            )

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
            raise TypeError(
                f"Expected TD3 navigation model, got {type(self.nav_model)}"
            )

        if self.nav_model.state_size != NAV_STATE_DIM:
            raise ValueError(
                "Navigation state-size mismatch: "
                f"model={self.nav_model.state_size}, "
                f"real env expected={NAV_STATE_DIM}"
            )

        # ------------------------------------------------------------------
        # Recurrent PPO adaptive sensing policy.
        # ------------------------------------------------------------------
        checkpoint = torch.load(adaptive_model_path, map_location="cpu")

        # Same compatibility rules as the Gazebo eval.py: 43-D input, matching
        # fps_choices, and a matching frame budget (missing -> legacy 450).
        try:
            checkpoint_summary = check_adaptive_checkpoint(
                checkpoint, FPS_CHOICES, budget
            )
        except ValueError as exc:
            raise SystemExit(
                f"cannot evaluate {adaptive_model_path}:\n  {exc}"
            )
        self.budget = float(checkpoint_summary["budget"])
        if not checkpoint_summary["budget_stored"]:
            self.get_logger().warning(
                f"Checkpoint stores no frame budget; assuming the legacy "
                f"default {checkpoint_summary['budget']} for frame_ratio. "
                "Verify this matches the budget it was trained with."
            )
        self.recorder.configuration.update(
            {
                "policy_obs_dim": PPO_STATE_DIM,
                "policy_obs_layout": ADAPTIVE_OBS_LAYOUT,
                "frame_budget": self.budget,
                "checkpoint_budget_stored": checkpoint_summary["budget_stored"],
                "checkpoint_frame_cost": checkpoint_summary["frame_cost"],
                "checkpoint_env_id": checkpoint_summary["env_id"],
                "checkpoint_scene": checkpoint_summary["scene"],
            }
        )

        lstm_hidden_size = checkpoint.get("args", {}).get(
            "lstm_hidden_size", 64
        )

        self.adaptive_model = AgentEval(
            obs_dim=PPO_STATE_DIM,
            n_actions=len(FPS_CHOICES),
            lstm_hidden_size=lstm_hidden_size,
        )
        self.adaptive_model.load_state_dict(checkpoint["model_state_dict"])
        self.adaptive_model.eval()

        self.get_logger().info(
            "Loaded models:\n"
            f"  TD3: {nav_model_run}, episode={nav_episode}, "
            f"state_dim={self.nav_model.state_size}, "
            f"step_time={self.nav_model.step_time}\n"
            f"  PPO: {adaptive_model_path}, obs_dim={PPO_STATE_DIM} "
            f"({ADAPTIVE_OBS_LAYOUT}), actions={FPS_CHOICES}, "
            f"lstm_hidden={lstm_hidden_size}, budget={self.budget:g} "
            f"(stored={checkpoint_summary['budget_stored']}), "
            f"trained frame_cost={checkpoint_summary['frame_cost']}"
        )

        self.get_logger().info(
            f"Integrated LiDAR gate: {raw_scan_topic} -> "
            f"{gated_scan_topic}; initial requested FPS={self.current_fps}"
        )

        self.get_logger().info(
            f"Passive episode output: {self.recorder.policy_dir}; "
            f"timeout={self.timeout_seconds:g}s"
        )

    # ======================================================================
    # Passive recording inputs
    # ======================================================================

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

    # ======================================================================
    # LiDAR gate
    # ======================================================================

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

        # Episode start / startup: immediately acquire one physical frame.
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

        # Fixed-phase/deadline scheduler.  This is necessary for non-integer
        # ratios such as a ~7.5-Hz physical source -> requested 5 Hz.
        while self.next_sample_time <= now:
            self.next_sample_time += self.period

    def set_target_fps(self, fps):
        fps = float(fps)
        if fps not in FPS_CHOICES:
            raise ValueError(f"FPS must be one of {FPS_CHOICES}; got {fps}")

        self.current_fps = fps
        self.period = 1.0 / fps

        # The selected action controls the interval AFTER the fresh frame that
        # caused the decision.  Anchor the next deadline to that fresh frame.
        anchor = (
            self.last_forward_time
            if self.last_forward_time is not None
            else time.monotonic()
        )
        self.next_sample_time = anchor + self.period

    def force_fresh_scan(self):
        self.force_next_fresh = True

    # ======================================================================
    # ROS service helpers
    # ======================================================================

    def wait_for_services(self):
        self.get_logger().info("Waiting for real_environment services...")

        while rclpy.ok() and not self.step_comm_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info("Waiting for step_comm...")

        while rclpy.ok() and not self.goal_comm_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info("Waiting for goal_comm...")

        self.get_logger().info("real_environment services are available.")

    def _wait_future(self, future):
        while rclpy.ok() and not future.done():
            # This also services /robot1/scan, which is important because the
            # integrated gate lives on THIS evaluator node.
            rclpy.spin_once(self, timeout_sec=0.01)

        if not future.done():
            raise RuntimeError("ROS future did not complete")

        exc = future.exception()
        if exc is not None:
            raise exc

        return future.result()

    def goal_is_ready(self):
        req = Goal.Request()
        response = self._wait_future(self.goal_comm_client.call_async(req))
        return bool(response.new_goal)

    def wait_for_manual_goal(self):
        self.get_logger().info(
            "Waiting for a manual goal. Publish it with your usual "
            "./spawn_goal x y command."
        )

        next_print = time.monotonic() + 3.0

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
                self.get_logger().info("Manual goal detected.")
                return self.latest_goal

            now = time.monotonic()
            if now >= next_print:
                self.get_logger().info("Still waiting for a manual goal...")
                next_print = now + 3.0

            # goal_is_ready() already spins this node; a short sleep prevents
            # unnecessary service polling.
            time.sleep(0.05)

        raise KeyboardInterrupt

    def call_step(self, action, previous_action):
        req = DrlStep.Request()
        req.action = [float(v) for v in action]
        req.previous_action = [float(v) for v in previous_action]

        response = self._wait_future(self.step_comm_client.call_async(req))

        return (
            np.asarray(response.state, dtype=np.float32),
            float(response.reward),
            bool(response.done),
            int(response.success),
            float(response.distance_traveled),
        )

    def init_episode(self):
        # Empty action triggers real_environment.initalize_episode().
        return self.call_step([], [0.0, 0.0])

    def wait_for_odometry(self):
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and self.recorder.latest_pose is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Timed out waiting for odometry on {self.odom_topic}"
                )
            rclpy.spin_once(self, timeout_sec=0.05)

    # ======================================================================
    # Safety
    # ======================================================================

    def safe_stop(self):
        # Publish several zero commands because this is physical hardware.
        zero = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)

    @staticmethod
    def action_to_command(action):
        if ENABLE_BACKWARD:
            linear = float(action[0]) * REAL_SPEED_LINEAR_MAX
        else:
            linear = (float(action[0]) + 1.0) * 0.5 * REAL_SPEED_LINEAR_MAX
        angular = float(action[1]) * REAL_SPEED_ANGULAR_MAX
        return linear, angular

    # ======================================================================
    # PPO observation
    # ======================================================================

    def make_ppo_observation(
        self, nav_state, episode_frame_count, steps_since_last_obs
    ):
        """Canonical 43-D observation, as Gazebo AdaptiveFPSEnv.get_augmented_obs():
        [40 held LiDAR rays, fps_ratio, obs_age_ratio, frame_count_ratio]."""
        if len(nav_state) != NAV_STATE_DIM:
            raise ValueError(
                f"Expected {NAV_STATE_DIM}-D navigation state, "
                f"got {len(nav_state)}"
            )

        # Logical PPO-selected rate (not the delivered LiDAR rate): 10 Hz -> 1.0.
        fps_ratio = self.current_fps / PPO_RATE_HZ

        # PPO steps since the last consumed fresh scan / 50, as in Gazebo.
        obs_age_ratio = float(
            np.clip(steps_since_last_obs / MAX_OBS_AGE_STEPS, 0.0, 1.0)
        )

        # Fresh scans this episode / checkpoint frame budget, as in Gazebo.
        frame_count_ratio = float(
            np.clip(episode_frame_count / self.budget, 0.0, 1.0)
        )

        obs = np.concatenate(
            [
                np.asarray(nav_state[:REAL_N_SCAN_SAMPLES], dtype=np.float32),
                np.asarray(
                    [fps_ratio, obs_age_ratio, frame_count_ratio],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        return obs, fps_ratio, obs_age_ratio, frame_count_ratio

    # ======================================================================
    # Episode
    # ======================================================================

    def run_episode(self, episode_index, max_episode_seconds):
        goal = self.wait_for_manual_goal()

        # Start every episode at the fastest adaptive action, matching reset()
        # in the Gazebo AdaptiveFPSEnv.
        self.current_fps = max(FPS_CHOICES)
        self.period = 1.0 / self.current_fps
        self.next_sample_time = None
        self.force_fresh_scan()

        # Force one fresh physical LiDAR frame before obtaining initial state.
        before = self.gated_scan_count
        deadline = time.monotonic() + 3.0

        while rclpy.ok() and self.gated_scan_count <= before:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Timed out waiting for initial raw LiDAR scan. "
                    f"Check topic {self.raw_scan_topic} and ROS_DOMAIN_ID."
                )
            rclpy.spin_once(self, timeout_sec=0.05)

        # Give the separately running real_environment a moment to process the
        # just-published /scan_gated message before init_episode().
        time.sleep(0.10)

        nav_state, _, done, outcome, distance_traveled = self.init_episode()

        if done:
            raise RuntimeError(
                "real_environment returned done=True on episode initialization"
            )

        self.wait_for_odometry()
        self.recorder.start_episode(
            goal=goal,
            initial_fresh_time=self.last_forward_time,
            ros_time_origin=self.get_clock().now().nanoseconds * 1e-9,
        )
        self._set_record_state(0.0, 0.0, UNKNOWN, False)
        self._record_current_sample()
        self.record_timer.reset()

        previous_nav_action = [0.0, 0.0]
        nav_step = 0

        # Gazebo reset semantics count the initial acquired frame as frame #1.
        episode_frame_count = 1
        last_seen_gated_count = self.gated_scan_count
        # Gazebo reset(): steps_since_last_obs = 0.
        steps_since_last_obs = 0

        ppo_obs, _, _, _ = self.make_ppo_observation(
            nav_state, episode_frame_count, steps_since_last_obs
        )

        hidden_size = self.adaptive_model.lstm.hidden_size
        lstm_state = (
            torch.zeros(1, 1, hidden_size),
            torch.zeros(1, 1, hidden_size),
        )

        episode_start = self.recorder.start_monotonic
        ppo_step = 0
        causal_decisions = 0
        causal_fps_counts = Counter()

        self.get_logger().info(
            f"===== REAL ADAPTIVE EPISODE {episode_index} START ====="
        )

        while rclpy.ok() and not done:
            if time.monotonic() - episode_start >= max_episode_seconds:
                self.record_timer.cancel()
                self.safe_stop()
                self._set_record_state(0.0, 0.0, TIMEOUT, True)
                self._record_current_sample()
                metadata = self.recorder.finish_episode(
                    TIMEOUT,
                    configuration={
                        "causal_decisions": causal_decisions,
                        "fps_action_counts": dict(causal_fps_counts),
                    },
                )
                self.get_logger().warning(
                    f"Evaluator safety timeout after "
                    f"{max_episode_seconds:.1f} s"
                )
                return {
                    "episode": metadata["episode"],
                    "done": True,
                    "outcome": TIMEOUT,
                    "safety_timeout": True,
                    "frames": metadata["n_fresh_observations"],
                    "causal_decisions": causal_decisions,
                    "fps_counts": dict(causal_fps_counts),
                }

            # --------------------------------------------------------------
            # PPO decision at 10 Hz.
            # IMPORTANT: recurrent state advances EVERY PPO step, even if no
            # fresh observation arrives in this interval.
            # --------------------------------------------------------------
            adaptive_action, lstm_state, probs = self.adaptive_model.predict(
                ppo_obs,
                lstm_state,
                done=False,
                deterministic=True,
                return_probs=True,
            )
            requested_fps = FPS_CHOICES[adaptive_action]

            # Gazebo step(): incremented at the start of every PPO step and
            # reset to 0 if that step consumes a fresh scan.
            steps_since_last_obs += 1

            ppo_step_start = time.monotonic()
            frame_consumed_this_ppo_step = False
            causal_action_applied = False
            inner_ticks = 0

            # --------------------------------------------------------------
            # Navigation controller continues at its normal real-agent rate
            # inside this 0.1-s PPO window.
            # --------------------------------------------------------------
            while rclpy.ok() and not done:
                inner_ticks += 1
                nav_step += 1

                nav_action = self.nav_model.get_action(
                    nav_state,
                    False,       # training
                    nav_step,
                    False,       # visualization
                )
                cmd_linear, cmd_angular = self.action_to_command(nav_action)

                (
                    next_nav_state,
                    nav_reward,
                    done,
                    outcome,
                    distance_traveled,
                ) = self.call_step(nav_action, previous_nav_action)

                nav_state = next_nav_state
                previous_nav_action = copy.deepcopy(nav_action)
                self._set_record_state(
                    cmd_linear,
                    cmd_angular,
                    outcome,
                    done,
                )

                # Did the integrated gate forward a new physical scan while
                # the service call / callbacks were being processed?
                new_frames = (
                    self.gated_scan_count - last_seen_gated_count
                )

                if new_frames > 0:
                    episode_frame_count += new_frames
                    last_seen_gated_count = self.gated_scan_count
                    frame_consumed_this_ppo_step = True
                    steps_since_last_obs = 0

                    # Same causal semantics as Gazebo:
                    # the action chosen at the BEGINNING of this PPO step
                    # controls the NEXT sensing interval.
                    if not causal_action_applied:
                        self.set_target_fps(requested_fps)
                        causal_action_applied = True
                        causal_decisions += 1
                        causal_fps_counts[requested_fps] += 1

                        if self.diagnose_probs:
                            prob_text = "  ".join(
                                f"{fps:g}Hz={probs[i]:.3f}"
                                for i, fps in enumerate(FPS_CHOICES)
                            )
                            print(
                                f"[CAUSAL] ppo_step={ppo_step:05d} "
                                f"frame={episode_frame_count:04d} "
                                f"chosen={requested_fps:g} Hz | "
                                f"{prob_text}"
                            )
                        else:
                            print(
                                f"[CAUSAL] ppo_step={ppo_step:05d} "
                                f"frame={episode_frame_count:04d} "
                                f"next_fps={requested_fps:g} Hz"
                )

                if done:
                    self.record_timer.cancel()
                    self._record_current_sample()
                    break

                # Preserve the stock real_agent's navigation timing.
                step_time = float(getattr(self.nav_model, "step_time", 0.0))
                if step_time > 0:
                    time.sleep(step_time)

                if time.monotonic() - ppo_step_start >= PPO_DT:
                    break

            ppo_step += 1

            # PPO observation for the NEXT 10-Hz recurrent decision.
            (
                ppo_obs,
                fps_ratio,
                obs_age_ratio,
                frame_count_ratio,
            ) = self.make_ppo_observation(
                nav_state, episode_frame_count, steps_since_last_obs
            )

            if (
                self.diagnose_probs
                and ppo_step % 10 == 0
            ):
                elapsed = time.monotonic() - episode_start
                print(
                    f"[PPO] t={elapsed:6.2f}s "
                    f"step={ppo_step:05d} "
                    f"inner={inner_ticks:02d} "
                    f"fresh={int(frame_consumed_this_ppo_step)} "
                    f"active_fps={self.current_fps:g} "
                    f"fps_ratio={fps_ratio:.3f} "
                    f"age_steps={steps_since_last_obs} "
                    f"age_ratio={obs_age_ratio:.3f} "
                    f"frame_ratio={frame_count_ratio:.3f}"
                )

        # real_environment already sends Twist() on success/collision, but
        # explicitly stop again for hardware safety.
        self.safe_stop()

        elapsed = time.monotonic() - episode_start

        metadata = self.recorder.finish_episode(
            outcome,
            configuration={
                "causal_decisions": causal_decisions,
                "fps_action_counts": dict(causal_fps_counts),
            },
        )

        self.get_logger().info(
            f"===== EPISODE {episode_index} END =====\n"
            f"  outcome: {outcome}\n"
            f"  wall time: {elapsed:.2f} s\n"
            f"  TD3 steps: {nav_step}\n"
            f"  PPO steps: {ppo_step}\n"
            f"  fresh frames: {episode_frame_count}\n"
            f"  causal decisions: {causal_decisions}\n"
            f"  FPS counts: {dict(causal_fps_counts)}\n"
            f"  distance_traveled: {distance_traveled:.3f}"
        )

        return {
            "episode": metadata["episode"],
            "done": True,
            "outcome": outcome,
            "safety_timeout": False,
            "wall_time": elapsed,
            "nav_steps": nav_step,
            "ppo_steps": ppo_step,
            "frames": metadata["n_fresh_observations"],
            "causal_decisions": causal_decisions,
            "fps_counts": dict(causal_fps_counts),
            "distance_traveled": distance_traveled,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Real-robot adaptive-FPS Sim2Real evaluator"
    )

    parser.add_argument(
        "--adaptive-model",
        required=True,
        help="Path to recurrent PPO checkpoint model.pt",
    )

    parser.add_argument(
        "--nav-model-run",
        default="examples/td3_0_stage9",
        help="StorageManager run/session containing frozen TD3",
    )
    parser.add_argument(
        "--nav-episode",
        type=int,
        default=7400,
        help="Frozen TD3 checkpoint episode",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of manually-goaled real episodes",
    )

    parser.add_argument(
        "--policy",
        required=True,
        help="Policy/run name used under RealRobot/eval/",
    )

    parser.add_argument(
        "--raw-scan-topic",
        default=RAW_SCAN_TOPIC,
    )
    parser.add_argument(
        "--gated-scan-topic",
        default="/robot1/scan_gated",
    )
    parser.add_argument(
        "--cmd-vel-topic",
        default=REAL_TOPIC_VELO,
    )
    parser.add_argument(
        "--odom-topic",
        default=REAL_TOPIC_ODOM,
    )
    parser.add_argument(
        "--goal-topic",
        default=GOAL_TOPIC,
    )
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
    )
    parser.add_argument(
        "--step-service",
        default="/step_comm",
    )
    parser.add_argument(
        "--goal-service",
        default="/goal_comm",
    )

    parser.add_argument(
        "--timeout",
        "--max-episode-seconds",
        dest="timeout_seconds",
        type=float,
        default=EPISODE_TIMEOUT_SECONDS,
        help="Evaluator-owned timeout in seconds",
    )

    parser.add_argument(
        "--diagnose-probs",
        action="store_true",
        help="Print recurrent PPO probability distribution at causal decisions",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=ADAPTIVE_FRAME_BUDGET,
        help="Frame budget: frame_ratio = episode frame count / budget. Must "
        "match the checkpoint's training budget (default: %(default)s)",
    )

    args, ros_args = parser.parse_known_args()

    if not os.path.isfile(args.adaptive_model):
        raise SystemExit(
            f"Adaptive checkpoint not found: {args.adaptive_model}"
        )

    if args.episodes <= 0:
        raise SystemExit("--episodes must be greater than zero")
    if args.timeout_seconds <= 0.0:
        raise SystemExit("--timeout must be greater than zero")

    rclpy.init(args=ros_args)

    node = RealAdaptiveEvaluator(
        adaptive_model_path=args.adaptive_model,
        nav_model_run=args.nav_model_run,
        nav_episode=args.nav_episode,
        raw_scan_topic=args.raw_scan_topic,
        gated_scan_topic=args.gated_scan_topic,
        step_service=args.step_service,
        goal_service=args.goal_service,
        cmd_vel_topic=args.cmd_vel_topic,
        odom_topic=args.odom_topic,
        goal_topic=args.goal_topic,
        policy_name=args.policy,
        output_root=args.output_root,
        timeout_seconds=args.timeout_seconds,
        diagnose_probs=args.diagnose_probs,
        budget=args.budget,
    )

    try:
        node.wait_for_services()

        results = []
        for episode in range(1, args.episodes + 1):
            result = node.run_episode(
                episode,
                max_episode_seconds=args.timeout_seconds,
            )
            results.append(result)

            if episode < args.episodes:
                print()
                print(
                    "Episode complete. Move/reset the physical robot as "
                    "needed, then publish the next manual goal."
                )
                print()

        print("\nAll requested real adaptive episodes finished.")
        for result in results:
            print(result)

    except KeyboardInterrupt:
        print("\nCtrl+C received: stopping physical robot.")

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
