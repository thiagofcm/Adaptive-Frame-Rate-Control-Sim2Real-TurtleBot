import csv
import json
import tempfile
import unittest
from pathlib import Path

from real_episode_recorder import RealEpisodeRecorder
from turtlebot3_drl.common.settings import SUCCESS


class RealEpisodeRecorderTest(unittest.TestCase):
    def make_recorder(self, root):
        return RealEpisodeRecorder(
            output_root=root,
            policy_name="fixed_5Hz",
            policy_type="fixed",
            configuration={"fixed_requested_fps": 5.0},
        )

    def test_completed_episode_files_and_measurements(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            recorder = self.make_recorder(temporary_root)
            recorder.update_pose(0.0, 0.0, 0.0, sample_id=(1, 0))
            recorder.update_goal(6.0, 8.0)

            number, episode_dir = recorder.start_episode(
                initial_fresh_time=10.0,
                now_monotonic=10.0,
            )
            self.assertEqual(number, 1)

            # Re-reporting the forced-fresh timestamp must not double count it.
            self.assertFalse(recorder.note_fresh_observation(10.0))
            recorder.update_pose(3.0, 4.0, 0.5, sample_id=(2, 0))
            recorder.update_pose(3.0, 4.0, 0.5, sample_id=(2, 0))
            self.assertTrue(recorder.note_fresh_observation(10.5))
            row = recorder.log_sample(
                cmd_linear=0.1,
                cmd_angular=-0.2,
                current_fps=5.0,
                now_monotonic=11.0,
            )
            recorder.log_sample(
                cmd_linear=0.0,
                cmd_angular=0.0,
                current_fps=1.0,
                outcome=SUCCESS,
                done=True,
                now_monotonic=11.1,
            )

            self.assertEqual(row["fresh_obs_counter"], 2)
            self.assertEqual(row["fresh_obs"], 1)
            self.assertAlmostEqual(row["current_fps"], 5.0)
            self.assertNotIn("timestamp", row)
            self.assertAlmostEqual(row["distance_traveled_m"], 5.0)
            self.assertAlmostEqual(row["goal_distance"], 5.0)

            metadata = recorder.finish_episode(SUCCESS, now_monotonic=12.0)
            self.assertTrue(metadata["success"])
            self.assertFalse(metadata["collision"])
            self.assertAlmostEqual(metadata["mean_fps"], 3.0)
            self.assertAlmostEqual(metadata["distance_traveled_m"], 5.0)

            trajectory_path = episode_dir / "trajectory.csv"
            metadata_path = episode_dir / "metadata.json"
            summary_path = Path(temporary_root) / "fixed_5Hz" / "episodes.csv"
            self.assertTrue(trajectory_path.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertTrue(summary_path.is_file())

            with trajectory_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertNotIn("timestamp", reader.fieldnames)
            self.assertEqual(rows[0]["outcome"], "RUNNING")
            self.assertEqual(rows[-1]["outcome"], "SUCCESS")

            with metadata_path.open(encoding="utf-8") as handle:
                saved_metadata = json.load(handle)
            self.assertEqual(saved_metadata["outcome_str"], "SUCCESS")

            with summary_path.open(newline="", encoding="utf-8") as handle:
                summaries = list(csv.DictReader(handle))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["episode"], "1")
            self.assertEqual(summaries[0]["mean_fps"], "3.0")

    def test_numbering_and_interrupted_episode(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            policy_dir = Path(temporary_root) / "fixed_5Hz"
            (policy_dir / "episode_0003").mkdir(parents=True)
            (policy_dir / "episode_notes").mkdir()

            recorder = self.make_recorder(temporary_root)
            recorder.update_pose(1.0, 2.0, 0.0, sample_id=1)
            recorder.update_goal(3.0, 4.0)
            number, episode_dir = recorder.start_episode(now_monotonic=1.0)
            self.assertEqual(number, 4)
            recorder.log_sample(0.0, 0.0, 1.0, now_monotonic=2.0)
            recorder.abort_episode("test interrupt", now_monotonic=3.0)

            self.assertTrue((episode_dir / "trajectory.csv").is_file())
            self.assertTrue((episode_dir / "metadata.partial.json").is_file())
            self.assertFalse((episode_dir / "metadata.json").exists())
            self.assertFalse((policy_dir / "episodes.csv").exists())


if __name__ == "__main__":
    unittest.main()
