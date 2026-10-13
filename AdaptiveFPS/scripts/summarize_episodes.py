#!/usr/bin/env python3
"""Create one policy-level summary row from an episodes.csv file."""

import argparse
import csv
import math
import os
import statistics
from pathlib import Path


OUTPUT_FIELDS = [
    "policy_name",
    "mean_fps",
    "std_fps",
    "n_episodes_total",
    "success_rate_percent",
    "mean_duration_s",
    "std_duration_s",
    "mean_distance_traveled_m",
    "std_distance_traveled_m",
    "mean_n_fresh_observations",
    "std_n_fresh_observations",
]

REQUIRED_FIELDS = {
    "policy_name",
    "success",
    "episode_duration_s",
    "distance_traveled_m",
    "mean_fps",
    "n_fresh_observations",
}


def parse_finite_float(row, field, row_number):
    value = row.get(field, "").strip()
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"row {row_number}: {field!r} must be numeric; got {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"row {row_number}: {field!r} must be finite; got {value!r}"
        )
    return number


def parse_success(row, row_number):
    value = row.get("success", "").strip().lower()
    if value in {"true", "1", "yes"}:
        return 1.0
    if value in {"false", "0", "no"}:
        return 0.0
    raise ValueError(
        f"row {row_number}: 'success' must be True/False or 1/0; got {value!r}"
    )


def mean_and_sample_std(values):
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, standard_deviation


def summarize(episodes_csv):
    episodes_csv = Path(episodes_csv).expanduser().resolve()
    if not episodes_csv.is_file():
        raise FileNotFoundError(f"episodes CSV not found: {episodes_csv}")

    with episodes_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {episodes_csv}")

        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "episodes CSV is missing required columns: " + ", ".join(missing)
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"episodes CSV contains no episode rows: {episodes_csv}")

    policy_names = {
        row["policy_name"].strip()
        for row in rows
        if row["policy_name"].strip()
    }
    if len(policy_names) > 1:
        raise ValueError(
            "episodes CSV contains multiple policy names: "
            + ", ".join(sorted(policy_names))
        )
    policy_name = next(iter(policy_names), episodes_csv.parent.name)

    successes = []
    episode_fps = []
    durations = []
    distances = []
    fresh_observations = []

    for row_number, row in enumerate(rows, start=2):
        successes.append(parse_success(row, row_number))
        episode_fps.append(parse_finite_float(row, "mean_fps", row_number))
        durations.append(
            parse_finite_float(row, "episode_duration_s", row_number)
        )
        distances.append(
            parse_finite_float(row, "distance_traveled_m", row_number)
        )
        fresh_observations.append(
            parse_finite_float(row, "n_fresh_observations", row_number)
        )

    mean_fps, std_fps = mean_and_sample_std(episode_fps)
    mean_duration, std_duration = mean_and_sample_std(durations)
    mean_distance, std_distance = mean_and_sample_std(distances)
    mean_fresh, std_fresh = mean_and_sample_std(fresh_observations)

    return {
        "policy_name": policy_name,
        "mean_fps": mean_fps,
        "std_fps": std_fps,
        "n_episodes_total": len(rows),
        "success_rate_percent": 100.0 * statistics.fmean(successes),
        "mean_duration_s": mean_duration,
        "std_duration_s": std_duration,
        "mean_distance_traveled_m": mean_distance,
        "std_distance_traveled_m": std_distance,
        "mean_n_fresh_observations": mean_fresh,
        "std_n_fresh_observations": std_fresh,
    }


def write_summary(episodes_csv, summary):
    output_path = Path(episodes_csv).expanduser().resolve().parent / "summary.csv"
    temporary_path = output_path.with_suffix(".csv.tmp")

    formatted = dict(summary)
    for field in OUTPUT_FIELDS:
        if field not in {"policy_name", "n_episodes_total"}:
            formatted[field] = f"{summary[field]:.6f}"

    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerow(formatted)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize one policy's completed real-robot episodes"
    )
    parser.add_argument(
        "episodes_csv",
        help="Path to RealRobot/eval/<policy>/episodes.csv",
    )
    args = parser.parse_args()

    try:
        summary = summarize(args.episodes_csv)
        output_path = write_summary(args.episodes_csv, summary)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote {output_path}")
    print(
        f"Policy={summary['policy_name']}  "
        f"episodes={summary['n_episodes_total']}  "
        f"success={summary['success_rate_percent']:.2f}%  "
        f"mean_fps={summary['mean_fps']:.3f}"
    )


if __name__ == "__main__":
    main()
