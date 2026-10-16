#!/usr/bin/env python3
"""Passive, reusable episode recording for real-robot evaluators."""

import csv
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from turtlebot3_drl.common.settings import (
    COLLISION_OBSTACLE,
    COLLISION_WALL,
    REAL_LIDAR_YAW_IN_BASE,
    REAL_N_SCAN_SAMPLES,
    SUCCESS,
    TIMEOUT,
    TUMBLE,
    UNKNOWN,
)


TRAJECTORY_FIELDS = [
    "episode_time_s",
    "x",
    "y",
    "yaw",
    "goal_x",
    "goal_y",
    "cmd_linear",
    "cmd_angular",
    "goal_distance",
    "distance_traveled_m",
    "current_fps",
    "fresh_obs",
    "fresh_obs_counter",
    "outcome",
    "done",
]

EPISODE_FIELDS = [
    "episode",
    "policy_name",
    "policy_type",
    "success",
    "collision",
    "collision_wall",
    "collision_obstacle",
    "tumble",
    "timeout",
    "outcome_code",
    "outcome_str",
    "episode_duration_s",
    "distance_traveled_m",
    "mean_fps",
    "n_fresh_observations",
    "start_x",
    "start_y",
    "final_x",
    "final_y",
    "goal_x",
    "goal_y",
    "final_goal_distance",
]

# LiDAR recording (lidar.npz / lidar.csv). The first columns follow the Gazebo
# npz_to_csv.py layout exactly; the real-robot extras are appended after them.
LIDAR_CSV_FIELDS = (
    ["scan_index", "t_sim_s", "t_wall_s"]
    + [f"beam_{i:02d}" for i in range(REAL_N_SCAN_SAMPLES)]
    + ["forwarded", "t_ros_s", "stamp_ros_s"]
    + [f"angle_{i:02d}" for i in range(REAL_N_SCAN_SAMPLES)]
)
LIDAR_CSV_FORMATS = (
    ["%d", "%.9f", "%.9f"]
    + ["%.6f"] * REAL_N_SCAN_SAMPLES
    + ["%d", "%.9f", "%.9f"]
    + ["%.9f"] * REAL_N_SCAN_SAMPLES
)

OUTCOME_NAMES = {
    UNKNOWN: "UNKNOWN",
    SUCCESS: "SUCCESS",
    COLLISION_WALL: "COLLISION_WALL",
    COLLISION_OBSTACLE: "COLLISION_OBSTACLE",
    TIMEOUT: "TIMEOUT",
    TUMBLE: "TUMBLE",
}


class RealEpisodeRecorder:
    """Store episode data without publishing, controlling, or ending navigation."""

    _EPISODE_PATTERN = re.compile(r"^episode_(\d+)$")
    _POLICY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    def __init__(
        self,
        output_root,
        policy_name,
        policy_type,
        configuration=None,
    ):
        if not self._POLICY_PATTERN.fullmatch(policy_name):
            raise ValueError(
                "policy_name must contain only letters, numbers, '.', '_', "
                "or '-', and must start with a letter or number"
            )
        if policy_type not in {"fixed", "adaptive"}:
            raise ValueError("policy_type must be 'fixed' or 'adaptive'")

        self.output_root = Path(output_root).expanduser().resolve()
        self.policy_name = policy_name
        self.policy_type = policy_type
        self.policy_dir = self.output_root / policy_name
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.configuration = dict(configuration or {})

        self.latest_pose = None
        self.latest_goal = None
        self._latest_odom_sample_id = None

        self.active = False
        self.episode_number = None
        self.episode_dir = None
        self.start_monotonic = None
        self.start_timestamp = None
        self.start_pose = None
        self.distance_traveled_m = 0.0
        self._last_distance_pose = None
        self._last_distance_sample_id = None
        self._fresh_times = []
        self._fresh_count_at_last_row = 0
        self._current_fps_sum = 0.0
        self._current_fps_sample_count = 0
        self._episode_configuration = {}
        self._trajectory_file = None
        self._trajectory_writer = None
        self._ros_time_origin = None
        self._reset_lidar_buffers()

    @property
    def fresh_observation_count(self):
        return len(self._fresh_times)

    @property
    def lidar_recording(self):
        """True while native scans are being recorded (episode start to terminal row)."""
        return self.active and not self._lidar_stopped

    def _reset_lidar_buffers(self):
        self._lidar_stopped = False
        self._lidar_t_wall = []
        self._lidar_stamp_ros = []
        self._lidar_ranges = []
        self._lidar_angles = []
        self._lidar_forwarded = []

    def note_scan(self, now_monotonic, msg, indices, forwarded):
        """Record one native LiDAR scan at the 40 beams the navigation state uses.

        msg is a sensor_msgs/LaserScan (only ranges, angle_min, angle_increment
        and header.stamp are read). indices are the native beam indices from
        drl_environment_real.real_scan_beam_indices(msg). Ranges are kept in
        meters exactly as published (inf/NaN preserved, no cap, no correction,
        no normalization), matching what Gazebo records. forwarded is True if
        the gate published this exact scan to the gated topic.
        """
        if not self.lidar_recording or indices is None:
            return False

        raw_index = np.asarray(indices, dtype=np.int32)
        all_ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = (
            float(msg.angle_min)
            + raw_index.astype(np.float64) * float(msg.angle_increment)
            + REAL_LIDAR_YAW_IN_BASE
        )
        angles = (angles + math.pi) % (2.0 * math.pi) - math.pi   # robot frame, [-pi, pi)

        stamp = msg.header.stamp
        self._lidar_t_wall.append(float(now_monotonic) - self.start_monotonic)
        self._lidar_stamp_ros.append(int(stamp.sec) + int(stamp.nanosec) * 1e-9)
        self._lidar_ranges.append(all_ranges[raw_index])
        self._lidar_angles.append(angles.astype(np.float32))
        self._lidar_forwarded.append(1 if forwarded else 0)
        return True

    def update_pose(self, x, y, yaw, sample_id=None):
        """Cache odometry and accumulate path length only during an episode."""
        pose = (float(x), float(y), float(yaw))
        self.latest_pose = pose
        self._latest_odom_sample_id = sample_id

        if not self.active:
            return
        if sample_id is not None and sample_id == self._last_distance_sample_id:
            return

        if self._last_distance_pose is not None:
            self.distance_traveled_m += math.hypot(
                pose[0] - self._last_distance_pose[0],
                pose[1] - self._last_distance_pose[1],
            )

        self._last_distance_pose = pose
        self._last_distance_sample_id = sample_id

    def update_goal(self, x, y):
        self.latest_goal = (float(x), float(y))

    def start_episode(
        self,
        goal=None,
        initial_fresh_time=None,
        configuration=None,
        now_monotonic=None,
        ros_time_origin=None,
    ):
        """Create a new episode directory when navigation is ready to begin.

        ros_time_origin (seconds, the evaluator node's ROS clock at this
        instant) is the origin of the LiDAR t_ros column; None leaves it NaN.
        """
        if self.active:
            raise RuntimeError("an episode is already being recorded")
        if self.latest_pose is None:
            raise RuntimeError("cannot start recording before receiving odometry")

        if goal is not None:
            self.update_goal(*goal)
        if self.latest_goal is None:
            raise RuntimeError("cannot start recording before receiving a goal")

        episode_number, episode_dir = self._reserve_episode_directory()
        trajectory_path = episode_dir / "trajectory.csv"

        self._trajectory_file = trajectory_path.open(
            "w", newline="", encoding="utf-8"
        )
        self._trajectory_writer = csv.DictWriter(
            self._trajectory_file,
            fieldnames=TRAJECTORY_FIELDS,
        )
        self._trajectory_writer.writeheader()
        self._trajectory_file.flush()

        self.active = True
        self.episode_number = episode_number
        self.episode_dir = episode_dir
        self.start_monotonic = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        self.start_timestamp = time.time()
        self.start_pose = self.latest_pose
        self.distance_traveled_m = 0.0
        self._last_distance_pose = self.latest_pose
        self._last_distance_sample_id = self._latest_odom_sample_id
        self._fresh_times = []
        self._fresh_count_at_last_row = 0
        self._current_fps_sum = 0.0
        self._current_fps_sample_count = 0
        self._episode_configuration = dict(configuration or {})
        self._ros_time_origin = (
            None if ros_time_origin is None else float(ros_time_origin)
        )
        self._reset_lidar_buffers()

        if initial_fresh_time is not None:
            self.note_fresh_observation(initial_fresh_time)

        return episode_number, episode_dir

    def note_fresh_observation(self, now_monotonic=None):
        """Count a scan only when a gate actually forwards a fresh message."""
        if not self.active:
            return False

        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if self._fresh_times and now == self._fresh_times[-1]:
            return False

        self._fresh_times.append(now)
        return True

    def log_sample(
        self,
        cmd_linear,
        cmd_angular,
        current_fps,
        outcome=UNKNOWN,
        done=False,
        goal_distance=None,
        now_monotonic=None,
    ):
        """Write one snapshot from the evaluator's independent record timer."""
        if not self.active:
            raise RuntimeError("cannot log a step without an active episode")
        if self.latest_pose is None or self.latest_goal is None:
            raise RuntimeError("pose and goal are required before logging")

        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        episode_time = max(0.0, now - self.start_monotonic)
        x, y, yaw = self.latest_pose
        goal_x, goal_y = self.latest_goal

        if goal_distance is None:
            goal_distance = math.hypot(goal_x - x, goal_y - y)

        fresh_count = self.fresh_observation_count
        fresh_obs = fresh_count > self._fresh_count_at_last_row
        row_outcome = OUTCOME_NAMES.get(int(outcome), f"UNKNOWN_{outcome}")
        if not done:
            row_outcome = "RUNNING"

        current_fps = float(current_fps)
        row = {
            "episode_time_s": episode_time,
            "x": x,
            "y": y,
            "yaw": yaw,
            "goal_x": goal_x,
            "goal_y": goal_y,
            "cmd_linear": float(cmd_linear),
            "cmd_angular": float(cmd_angular),
            "goal_distance": float(goal_distance),
            "distance_traveled_m": self.distance_traveled_m,
            "current_fps": current_fps,
            "fresh_obs": int(fresh_obs),
            "fresh_obs_counter": fresh_count,
            "outcome": row_outcome,
            "done": int(bool(done)),
        }

        self._trajectory_writer.writerow(row)
        self._trajectory_file.flush()
        self._current_fps_sum += current_fps
        self._current_fps_sample_count += 1
        self._fresh_count_at_last_row = fresh_count
        if done:
            # LiDAR recording ends at the terminal row, like Gazebo pausing at
            # the terminal step; later scans (safe stop, wrap-up) are not kept.
            self._lidar_stopped = True
        return row

    def finish_episode(self, outcome, configuration=None, now_monotonic=None):
        """Finalize files after the evaluator has already ended the episode."""
        if not self.active:
            raise RuntimeError("cannot finish without an active episode")

        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        duration = max(0.0, now - self.start_monotonic)
        outcome = int(outcome)
        outcome_str = OUTCOME_NAMES.get(outcome, f"UNKNOWN_{outcome}")
        final_x, final_y, _ = self.latest_pose
        start_x, start_y, _ = self.start_pose
        goal_x, goal_y = self.latest_goal
        final_goal_distance = math.hypot(goal_x - final_x, goal_y - final_y)
        n_fresh = self.fresh_observation_count
        mean_fps = (
            self._current_fps_sum / self._current_fps_sample_count
            if self._current_fps_sample_count > 0
            else 0.0
        )

        collision_wall = outcome == COLLISION_WALL
        collision_obstacle = outcome == COLLISION_OBSTACLE

        metadata = dict(self.configuration)
        metadata.update(self._episode_configuration)
        metadata.update(configuration or {})
        metadata.update(
            {
                "episode": self.episode_number,
                "policy_name": self.policy_name,
                "policy_type": self.policy_type,
                "success": outcome == SUCCESS,
                "collision": collision_wall or collision_obstacle,
                "collision_wall": collision_wall,
                "collision_obstacle": collision_obstacle,
                "timeout": outcome == TIMEOUT,
                "tumble": outcome == TUMBLE,
                "outcome_code": outcome,
                "outcome_str": outcome_str,
                "episode_duration_s": duration,
                "distance_traveled_m": self.distance_traveled_m,
                "mean_fps": mean_fps,
                "n_fresh_observations": n_fresh,
                "start_x": start_x,
                "start_y": start_y,
                "final_x": final_x,
                "final_y": final_y,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "final_goal_distance": final_goal_distance,
                "start_timestamp": self.start_timestamp,
                "completed": True,
            }
        )

        self._close_trajectory()
        self._write_lidar()
        self._write_json_atomic(self.episode_dir / "metadata.json", metadata)
        self._append_episode_summary(metadata)
        self._print_summary(metadata)
        self.active = False
        return metadata

    def abort_episode(self, reason="interrupted", now_monotonic=None):
        """Preserve partial trajectory data without marking it completed."""
        if not self.active:
            return None

        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        duration = max(0.0, now - self.start_monotonic)
        partial = {
            "episode": self.episode_number,
            "policy_name": self.policy_name,
            "policy_type": self.policy_type,
            "completed": False,
            "interruption_reason": str(reason),
            "episode_duration_s": duration,
            "distance_traveled_m": self.distance_traveled_m,
            "mean_fps": (
                self._current_fps_sum / self._current_fps_sample_count
                if self._current_fps_sample_count > 0
                else 0.0
            ),
            "n_fresh_observations": self.fresh_observation_count,
        }
        self._close_trajectory()
        self._write_lidar()
        self._write_json_atomic(
            self.episode_dir / "metadata.partial.json",
            partial,
        )
        print(
            f"Partial episode {self.episode_number:04d} preserved at "
            f"{self.episode_dir}; episodes.csv was not updated."
        )
        self.active = False
        return partial

    def _reserve_episode_directory(self):
        existing = []
        for path in self.policy_dir.iterdir():
            if not path.is_dir():
                continue
            match = self._EPISODE_PATTERN.fullmatch(path.name)
            if match:
                existing.append(int(match.group(1)))

        episode_number = max(existing, default=0) + 1
        while True:
            episode_dir = self.policy_dir / f"episode_{episode_number:04d}"
            try:
                episode_dir.mkdir()
                return episode_number, episode_dir
            except FileExistsError:
                episode_number += 1

    def _append_episode_summary(self, metadata):
        summary_path = self.policy_dir / "episodes.csv"
        write_header = not summary_path.exists() or summary_path.stat().st_size == 0
        with summary_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({field: metadata[field] for field in EPISODE_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())

    def lidar_arrays(self):
        """The recorded native scans as arrays: Gazebo keys (t_wall, t_sim,
        ranges) plus real-robot extras. lidar.npz and lidar.csv are both
        written from exactly these arrays."""
        n = len(self._lidar_t_wall)
        beams = REAL_N_SCAN_SAMPLES
        stamp_ros = np.asarray(self._lidar_stamp_ros, dtype=np.float64)
        origin = np.nan if self._ros_time_origin is None else self._ros_time_origin
        return {
            # Gazebo keys. t_sim is NaN: the real robot has no simulation clock.
            "t_wall": np.asarray(self._lidar_t_wall, dtype=np.float64),
            "t_sim": np.full(n, np.nan, dtype=np.float64),
            "ranges": (np.stack(self._lidar_ranges).astype(np.float32) if n
                       else np.zeros((0, beams), dtype=np.float32)),
            # Real-robot extras.
            "forwarded": np.asarray(self._lidar_forwarded, dtype=np.int8),
            "t_ros": stamp_ros - origin,
            "stamp_ros": stamp_ros,
            "beam_angle_rad": (np.stack(self._lidar_angles).astype(np.float32) if n
                               else np.zeros((0, beams), dtype=np.float32)),
        }

    def _write_lidar(self):
        """Write lidar.npz and lidar.csv (Gazebo npz_to_csv.py column layout
        and number formats first, real-robot extras appended), atomically."""
        arrays = self.lidar_arrays()
        n = len(arrays["t_wall"])

        npz_path = self.episode_dir / "lidar.npz"
        temporary_npz = npz_path.with_suffix(".npz.tmp")
        with temporary_npz.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_npz, npz_path)

        table = np.column_stack((
            np.arange(n, dtype=np.float64),
            arrays["t_sim"],
            arrays["t_wall"],
            arrays["ranges"].astype(np.float64),
            arrays["forwarded"].astype(np.float64),
            arrays["t_ros"],
            arrays["stamp_ros"],
            arrays["beam_angle_rad"].astype(np.float64),
        )) if n else np.zeros((0, len(LIDAR_CSV_FIELDS)))

        csv_path = self.episode_dir / "lidar.csv"
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        with temporary_csv.open("w", encoding="utf-8") as handle:
            np.savetxt(
                handle,
                table,
                delimiter=",",
                header=",".join(LIDAR_CSV_FIELDS),
                comments="",
                fmt=LIDAR_CSV_FORMATS,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_csv, csv_path)

    @staticmethod
    def _write_json_atomic(path, payload):
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)

    def _close_trajectory(self):
        if self._trajectory_file is not None:
            self._trajectory_file.flush()
            os.fsync(self._trajectory_file.fileno())
            self._trajectory_file.close()
        self._trajectory_file = None
        self._trajectory_writer = None

    def _print_summary(self, metadata):
        print("=" * 60)
        print(f"Episode {metadata['episode']:04d} finished")
        print(f"Policy: {metadata['policy_name']}")
        print(f"Outcome: {metadata['outcome_str']}")
        print()
        print(f"Duration:             {metadata['episode_duration_s']:9.2f} s")
        print(f"Distance traveled:    {metadata['distance_traveled_m']:9.2f} m")
        print(f"Mean current FPS:     {metadata['mean_fps']:9.2f} Hz")
        print(f"Fresh observations:   {metadata['n_fresh_observations']:9d}")
        print()
        print(
            f"Start:  x={metadata['start_x']:.2f}, "
            f"y={metadata['start_y']:.2f}"
        )
        print(f"Goal:   x={metadata['goal_x']:.2f}, y={metadata['goal_y']:.2f}")
        print(
            f"Final:  x={metadata['final_x']:.2f}, "
            f"y={metadata['final_y']:.2f}"
        )
        print(
            f"Final goal distance:  {metadata['final_goal_distance']:9.2f} m"
        )
        print()
        print(f"Success:   {metadata['success']}")
        print(f"Collision: {metadata['collision']}")
        print(f"Timeout:   {metadata['timeout']}")
        print()
        print(f"Saved: {self.episode_dir}")
        print("=" * 60)
