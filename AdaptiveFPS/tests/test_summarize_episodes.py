import csv
import tempfile
import unittest
from pathlib import Path

from summarize_episodes import OUTPUT_FIELDS, summarize, write_summary


class SummarizeEpisodesTest(unittest.TestCase):
    def write_episodes(self, directory, rows):
        path = Path(directory) / "episodes.csv"
        fields = [
            "policy_name",
            "success",
            "episode_duration_s",
            "distance_traveled_m",
            "mean_fps",
            "n_fresh_observations",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_summary_and_output_location(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            episodes_path = self.write_episodes(
                temporary_directory,
                [
                    {
                        "policy_name": "adaptive_test",
                        "success": "True",
                        "episode_duration_s": "1",
                        "distance_traveled_m": "2",
                        "mean_fps": "1",
                        "n_fresh_observations": "10",
                    },
                    {
                        "policy_name": "adaptive_test",
                        "success": "False",
                        "episode_duration_s": "2",
                        "distance_traveled_m": "4",
                        "mean_fps": "2",
                        "n_fresh_observations": "20",
                    },
                    {
                        "policy_name": "adaptive_test",
                        "success": "1",
                        "episode_duration_s": "3",
                        "distance_traveled_m": "6",
                        "mean_fps": "3",
                        "n_fresh_observations": "30",
                    },
                ],
            )

            summary = summarize(episodes_path)
            self.assertEqual(summary["policy_name"], "adaptive_test")
            self.assertEqual(summary["n_episodes_total"], 3)
            self.assertAlmostEqual(summary["success_rate_percent"], 200 / 3)
            self.assertAlmostEqual(summary["mean_fps"], 2.0)
            self.assertAlmostEqual(summary["std_fps"], 1.0)
            self.assertAlmostEqual(summary["mean_duration_s"], 2.0)
            self.assertAlmostEqual(summary["std_duration_s"], 1.0)
            self.assertAlmostEqual(summary["mean_distance_traveled_m"], 4.0)
            self.assertAlmostEqual(summary["std_distance_traveled_m"], 2.0)
            self.assertAlmostEqual(summary["mean_n_fresh_observations"], 20.0)
            self.assertAlmostEqual(summary["std_n_fresh_observations"], 10.0)

            output_path = write_summary(episodes_path, summary)
            self.assertEqual(output_path, Path(temporary_directory) / "summary.csv")
            with output_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, OUTPUT_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["success_rate_percent"], "66.666667")

    def test_one_episode_has_zero_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            episodes_path = self.write_episodes(
                temporary_directory,
                [
                    {
                        "policy_name": "fixed_10Hz",
                        "success": "False",
                        "episode_duration_s": "4.5",
                        "distance_traveled_m": "0.8",
                        "mean_fps": "10",
                        "n_fresh_observations": "31",
                    }
                ],
            )
            summary = summarize(episodes_path)
            self.assertEqual(summary["std_fps"], 0.0)
            self.assertEqual(summary["std_duration_s"], 0.0)
            self.assertEqual(summary["std_distance_traveled_m"], 0.0)
            self.assertEqual(summary["std_n_fresh_observations"], 0.0)


if __name__ == "__main__":
    unittest.main()
