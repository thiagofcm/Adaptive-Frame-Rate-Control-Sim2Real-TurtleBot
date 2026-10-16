import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import xml.etree.ElementTree as ET

from real_episode_recorder import LIDAR_CSV_FIELDS, RealEpisodeRecorder
from turtlebot3_drl.common.settings import (
    REAL_LIDAR_CORRECTION,
    REAL_LIDAR_DISTANCE_CAP,
    REAL_LIDAR_YAW_IN_BASE,
    REAL_N_SCAN_SAMPLES,
    SIM_LIDAR_ANGLE_MAX,
    SIM_LIDAR_ANGLE_MIN,
    SUCCESS,
)
from turtlebot3_drl.drl_environment.drl_environment_real import (
    DRLEnvironment,
    real_scan_beam_indices,
)


GAZEBO_CSV_HEADER = ["scan_index", "t_sim_s", "t_wall_s"] + [
    f"beam_{i:02d}" for i in range(40)
]


def make_scan(ranges, angle_min=-math.pi, angle_increment=None, sec=100, nanosec=0):
    ranges = list(ranges)
    if angle_increment is None:
        angle_increment = 2.0 * math.pi / len(ranges)
    return SimpleNamespace(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
    )


MODEL_SDF = (Path(__file__).resolve().parents[2] / "src" / "turtlebot3_simulations"
             / "turtlebot3_gazebo" / "models" / "turtlebot3_burger" / "model.sdf")
GAZEBO_ANGLES = SIM_LIDAR_ANGLE_MIN + np.arange(40) * (SIM_LIDAR_ANGLE_MAX - SIM_LIDAR_ANGLE_MIN) / 39


def wrapped(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def selected_robot_angles(msg):
    idx = np.asarray(real_scan_beam_indices(msg))
    return wrapped(msg.angle_min + idx * msg.angle_increment + REAL_LIDAR_YAW_IN_BASE)


def reference_scan_callback(msg):
    """Pre-change scan_callback normalization (inf/NaN -> cap, correction, /cap,
    clip, minimum), applied to the beams real_scan_beam_indices selects."""
    scan_ranges = [REAL_LIDAR_DISTANCE_CAP] * REAL_N_SCAN_SAMPLES
    indices = real_scan_beam_indices(msg)
    if indices is None:
        return None
    obstacle_distance = 1
    for i, idx in enumerate(indices):
        raw = float(msg.ranges[idx])
        if np.isinf(raw) or np.isnan(raw):
            raw = REAL_LIDAR_DISTANCE_CAP
        scan_ranges[i] = np.clip((raw - REAL_LIDAR_CORRECTION) / REAL_LIDAR_DISTANCE_CAP, 0, 1)
        if scan_ranges[i] < obstacle_distance:
            obstacle_distance = scan_ranges[i]
    return scan_ranges, obstacle_distance * REAL_LIDAR_DISTANCE_CAP


class BeamSelectionTest(unittest.TestCase):
    def test_settings_match_gazebo_model(self):
        root = ET.parse(MODEL_SDF).getroot()
        sensor = next(link for link in root.find("model").findall("link")
                      if link.get("name") == "base_scan").find("sensor")
        horizontal = sensor.find("ray").find("scan").find("horizontal")
        self.assertEqual(int(horizontal.find("samples").text), REAL_N_SCAN_SAMPLES)
        self.assertEqual(float(horizontal.find("min_angle").text), SIM_LIDAR_ANGLE_MIN)
        self.assertEqual(float(horizontal.find("max_angle").text), SIM_LIDAR_ANGLE_MAX)
        self.assertEqual(float(sensor.find("pose").text.split()[5]), 0.0)   # no sensor yaw

    def test_selects_gazebo_directions(self):
        # Full-circle scans, both directions: within half a native step of every Gazebo beam.
        for n, angle_min, sign in ((720, -math.pi, 1), (1147, -3.1, 1), (360, math.pi, -1),
                                   (1440, 0.0, 1), (533, -math.pi, -1)):
            msg = make_scan([1.0] * n, angle_min, sign * 2.0 * math.pi / n)
            error = wrapped(selected_robot_angles(msg) - GAZEBO_ANGLES)
            self.assertLessEqual(np.abs(error).max(), math.pi / n + 1e-9)
            idx = real_scan_beam_indices(msg)
            self.assertLessEqual(min(abs(idx[39] - idx[0]), n - abs(idx[39] - idx[0])), 1)

    def test_selects_gazebo_directions_on_recorded_rplidar_geometry(self):
        # The robot's driver: 720 samples from -3.12414 rad to +pi (about 359 deg, ~1 deg seam gap).
        n, angle_min = 720, -3.12414
        msg = make_scan([1.0] * n, angle_min, (math.pi - angle_min) / (n - 1))
        error = np.degrees(wrapped(selected_robot_angles(msg) - GAZEBO_ANGLES))
        self.assertLessEqual(np.abs(error).max(), 0.6)
        self.assertAlmostEqual(float(np.degrees(selected_robot_angles(msg)[39])), -0.183, delta=0.3)

    def test_unusable_scan(self):
        self.assertIsNone(real_scan_beam_indices(make_scan([], angle_increment=0.01)))
        self.assertIsNone(real_scan_beam_indices(make_scan([1.0] * 10, angle_increment=0.0)))

    def test_scan_callback_output_unchanged(self):
        rng = np.random.default_rng(0)
        for n, angle_min, sign in ((720, -math.pi, 1), (1147, -3.12, 1), (360, math.pi, -1),
                                   (1440, 0.0, 1), (533, -math.pi, -1)):
            for _ in range(5):
                ranges = rng.uniform(0.05, 6.0, n)
                ranges[rng.choice(n, 30, replace=False)] = np.inf
                ranges[rng.choice(n, 10, replace=False)] = np.nan
                msg = make_scan(ranges, angle_min, sign * 2.0 * math.pi / n)
                fake = SimpleNamespace(scan_ranges=[REAL_LIDAR_DISTANCE_CAP] * REAL_N_SCAN_SAMPLES,
                                       obstacle_distance=REAL_LIDAR_DISTANCE_CAP)
                DRLEnvironment.scan_callback(fake, msg)
                expected_ranges, expected_obstacle = reference_scan_callback(msg)
                self.assertEqual([float(v) for v in fake.scan_ranges],
                                 [float(v) for v in expected_ranges])
                self.assertEqual(float(fake.obstacle_distance), float(expected_obstacle))


class LidarRecordingTest(unittest.TestCase):
    def _recorder(self, directory):
        recorder = RealEpisodeRecorder(directory, "adaptive_test", "adaptive")
        recorder.update_pose(0.0, 0.0, 0.0, sample_id=(1, 0))
        recorder.start_episode(goal=(2.0, 0.0), initial_fresh_time=9.9,
                               now_monotonic=10.0, ros_time_origin=100.0)
        return recorder

    def _scans(self, rng):
        scans = []
        for k in range(6):
            ranges = rng.uniform(0.1, 5.0, 720).astype(np.float32)
            ranges[::50] = np.inf
            selected = real_scan_beam_indices(make_scan([1.0] * 720))
            ranges[[selected[2], selected[15]]] = np.nan   # on selected beams
            scans.append(make_scan(ranges, sec=100, nanosec=int(k * 128e6)))
        return scans

    def test_files_agree_and_follow_gazebo_layout(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            forwarded = [True, False, True, False, False, True]
            for k, (msg, fwd) in enumerate(zip(self._scans(rng), forwarded)):
                now = 10.0 + 0.128 * (k + 1)
                if fwd:
                    recorder.note_fresh_observation(now)   # what _forward_scan does
                recorder.note_scan(now, msg, real_scan_beam_indices(msg), forwarded=fwd)
            row = recorder.log_sample(0.0, 0.0, 5.0, outcome=SUCCESS, done=True,
                                      now_monotonic=10.9)
            # Scans after the terminal row are not recorded.
            late = make_scan([1.0] * 720, sec=101)
            self.assertFalse(recorder.note_scan(11.0, late, real_scan_beam_indices(late), True))
            recorder.finish_episode(SUCCESS, now_monotonic=11.0)

            episode = Path(directory) / "adaptive_test" / "episode_0001"
            data = np.load(episode / "lidar.npz")
            self.assertEqual(set(data.files), {"t_wall", "t_sim", "ranges", "forwarded", "t_ros",
                                               "stamp_ros", "beam_angle_rad"})
            self.assertEqual(data["ranges"].shape, (6, 40))
            self.assertEqual(data["ranges"].dtype, np.float32)
            self.assertTrue(np.isnan(data["t_sim"]).all())
            np.testing.assert_allclose(data["t_wall"], 0.128 * np.arange(1, 7), atol=1e-9)
            np.testing.assert_allclose(data["t_ros"], 0.128 * np.arange(6), atol=1e-6)
            self.assertEqual(data["forwarded"].tolist(), [1, 0, 1, 0, 0, 1])

            # Raw meters at the selected indices, inf/NaN preserved.
            for k, msg in enumerate(self._scans(np.random.default_rng(1))):
                idx = real_scan_beam_indices(msg)
                np.testing.assert_array_equal(data["ranges"][k],
                                              np.asarray(msg.ranges, np.float32)[idx])
            self.assertTrue(np.isinf(data["ranges"]).any())
            self.assertTrue(np.isnan(data["ranges"]).any())

            # Beam 0 points forward; angles are the actual native beam directions.
            first_indices = np.asarray(real_scan_beam_indices(self._scans(np.random.default_rng(1))[0]))
            expected_angles = (-math.pi + first_indices * 2 * math.pi / 720
                               + REAL_LIDAR_YAW_IN_BASE + math.pi) % (2 * math.pi) - math.pi
            np.testing.assert_allclose(data["beam_angle_rad"][0], expected_angles, atol=1e-6)
            self.assertAlmostEqual(float(data["beam_angle_rad"][0][0]), 0.0, places=6)

            # The terminal row's fresh count = forced first scan + forwarded native scans.
            self.assertEqual(row["fresh_obs_counter"], 1 + int(data["forwarded"].sum()))

            with (episode / "lidar.csv").open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], LIDAR_CSV_FIELDS)
            self.assertEqual(rows[0][:43], GAZEBO_CSV_HEADER)
            self.assertEqual(len(rows) - 1, 6)
            table = np.array([[float(v) for v in r] for r in rows[1:]])
            self.assertEqual(table[:, 0].tolist(), list(range(6)))
            self.assertTrue(np.isnan(table[:, 1]).all())
            np.testing.assert_allclose(table[:, 2], data["t_wall"], atol=5e-10)
            csv_ranges, npz_ranges = table[:, 3:43], data["ranges"].astype(np.float64)
            np.testing.assert_array_equal(np.isinf(csv_ranges), np.isinf(npz_ranges))
            np.testing.assert_array_equal(np.isnan(csv_ranges), np.isnan(npz_ranges))
            finite = np.isfinite(npz_ranges)
            np.testing.assert_allclose(csv_ranges[finite], npz_ranges[finite], atol=5e-7)
            self.assertEqual(table[:, 43].tolist(), data["forwarded"].tolist())
            np.testing.assert_allclose(table[:, 46:86], data["beam_angle_rad"], atol=5e-9)
            self.assertEqual(table.shape[1], 86)

            # Gazebo npz_to_csv.py on this npz gives exactly the first 43 CSV columns.
            legacy = np.column_stack((np.arange(6), data["t_sim"], data["t_wall"], data["ranges"]))
            with tempfile.NamedTemporaryFile("w+", suffix=".csv") as converted:
                np.savetxt(converted.name, legacy, delimiter=",", header=",".join(GAZEBO_CSV_HEADER),
                           comments="", fmt=["%d", "%.9f", "%.9f"] + ["%.6f"] * 40)
                with open(converted.name) as handle:
                    legacy_rows = list(csv.reader(handle))
            self.assertEqual([r[:43] for r in rows], legacy_rows)

    def test_interrupted_episode_keeps_partial_lidar(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            msg = make_scan([2.0] * 720)
            recorder.note_scan(10.1, msg, real_scan_beam_indices(msg), forwarded=False)
            recorder.abort_episode("test")
            episode = Path(directory) / "adaptive_test" / "episode_0001"
            self.assertTrue((episode / "metadata.partial.json").is_file())
            self.assertEqual(np.load(episode / "lidar.npz")["ranges"].shape, (1, 40))
            with (episode / "lidar.csv").open() as handle:
                self.assertEqual(len(list(csv.reader(handle))), 2)
            self.assertFalse(json.loads((episode / "metadata.partial.json").read_text())["completed"])

    def test_episodes_have_separate_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            msg = make_scan([2.0] * 720)
            recorder.note_scan(10.1, msg, real_scan_beam_indices(msg), forwarded=False)
            recorder.finish_episode(SUCCESS, now_monotonic=10.5)
            recorder.start_episode(goal=(1.0, 0.0), now_monotonic=20.0, ros_time_origin=200.0)
            recorder.finish_episode(SUCCESS, now_monotonic=20.5)
            policy = Path(directory) / "adaptive_test"
            self.assertEqual(len(np.load(policy / "episode_0001" / "lidar.npz")["t_wall"]), 1)
            self.assertEqual(len(np.load(policy / "episode_0002" / "lidar.npz")["t_wall"]), 0)
            with (policy / "episode_0002" / "lidar.csv").open() as handle:
                self.assertEqual(list(csv.reader(handle)), [LIDAR_CSV_FIELDS])


if __name__ == "__main__":
    unittest.main()
