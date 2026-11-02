"""Per-episode and aggregate LiDAR statistics for one real-robot fixed-FPS evaluation run.

Real-robot counterpart of adaptive_sensor_policy/analyze_lidar.py (simulation). Reads every
<eval-dir>/episode_*/lidar.npz written by AdaptiveFPS/scripts/real_episode_recorder.py:
    ranges (N,40) m       native scan at the 40 policy beams, as published (Inf/NaN kept, no cap)
    forwarded (N,) int8   1 if the gate delivered this scan to the navigation controller
    t_wall (N,) s         monotonic receive time since episode start
    t_ros (N,) s          scan header stamp relative to episode start (NaN if no origin)
    t_sim (N,)            always NaN (no simulation clock), ignored
    beam_angle_rad (N,40) beam angles in the robot frame, [-pi, pi)
Every row is a distinct native scan; forwarded rows play the role of sim fresh=True rows.

Timing: intervals between forwarded scans (delivered observations) from t_wall. Scans stamped before
the episode (t_ros < 0, the startup backlog delivered in one burst) are excluded from timing only.
Ranges (forwarded scans): shared columns use the drl_environment_real.scan_callback preprocessing
(Inf/NaN -> cap, - correction, / cap, clip [0,1]) reported in meters (x cap), i.e. what the policy
received, like the sim recording; raw_* columns use the ranges as recorded.
lidar_offset_x is NaN: the real sensor offset is not verified yet.

Column schema is shared with the simulation analyze_lidar.py: shared metrics first, with identical names,
order and definitions (fresh = forwarded, physical time = t_wall), then real-only columns.

Outputs go to RealRobot/eval/lidar_analysis/<policy>/{episode_metrics.csv, summary.csv}.

    python3 RealRobot/analyze_lidar_real.py --eval-dir RealRobot/eval/fixed_10Hz
    python3 RealRobot/analyze_lidar_real.py --eval-dir RealRobot/eval/fixed_10Hz --policy 10Hz
"""
import argparse
import importlib.util
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REAL_ROOT = Path(__file__).resolve().parent
EVAL_ROOT = REAL_ROOT / "eval"
OUT_ROOT = EVAL_ROOT / "lidar_analysis"
SETTINGS = REAL_ROOT.parent / "src" / "turtlebot3_drl" / "turtlebot3_drl" / "common" / "settings.py"
FAR_THRESHOLD_M = 3.5
CAP_TOL = 1e-6


def load_settings():
    """REAL_* constants straight from settings.py, so preprocessing matches drl_environment_real.py."""
    spec = importlib.util.spec_from_file_location("drl_settings", SETTINGS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.REAL_N_SCAN_SAMPLES, float(module.REAL_LIDAR_DISTANCE_CAP), float(module.REAL_LIDAR_CORRECTION)


NUM_BEAMS, DISTANCE_CAP, LIDAR_CORRECTION = load_settings()


def warn(msg):
    print(f"  WARNING: {msg}")


# ---------------------------------------------------------------- policy / paths

def identify_policy(eval_dir, override):
    """Configured policy label (e.g. '10Hz') from --policy, episode metadata, or the directory name."""
    if override:
        return override
    labels = set()
    for meta_path in sorted(eval_dir.glob("episode_*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("policy_type") == "fixed" and meta.get("fixed_requested_fps") is not None:
            labels.add(f"{float(meta['fixed_requested_fps']):g}Hz")
    if len(labels) > 1:
        raise SystemExit(f"Inconsistent fixed FPS across episode metadata: {sorted(labels)}; use --policy")
    if labels:
        return labels.pop()
    m = re.fullmatch(r"fixed_([0-9.]+)Hz", eval_dir.name)
    if m:
        return f"{float(m.group(1)):g}Hz"
    raise SystemExit(f"Cannot identify a fixed-FPS policy for {eval_dir}; use --policy")


def output_dir(eval_dir, policy):
    """eval/lidar_analysis/[<subfolders>/]<policy>, resolved against the known eval root."""
    try:
        rel = eval_dir.resolve().relative_to(EVAL_ROOT.resolve())
    except ValueError:
        raise SystemExit(f"--eval-dir must be inside {EVAL_ROOT}")
    if rel.parts and rel.parts[0] == OUT_ROOT.name:
        raise SystemExit("--eval-dir must not point inside lidar_analysis/")
    return OUT_ROOT.joinpath(*rel.parts[:-1], policy)


# ---------------------------------------------------------------- loading / validation

def load_lidar(path):
    """Return a dict of validated arrays, or raise ValueError with the reason."""
    with np.load(path, allow_pickle=False) as d:
        missing = {"ranges", "forwarded", "t_wall", "t_ros", "beam_angle_rad"} - set(d.files)
        if missing:
            raise ValueError(f"missing fields {sorted(missing)}")
        data = {k: d[k] for k in ("ranges", "forwarded", "t_wall", "t_ros", "beam_angle_rad")}

    ranges = data["ranges"].astype(np.float64)       # NaN/Inf preserved
    if ranges.ndim != 2 or ranges.shape[1] != NUM_BEAMS:
        raise ValueError(f"ranges shape {ranges.shape}, expected (N,{NUM_BEAMS})")
    n = ranges.shape[0]
    for k in ("forwarded", "t_wall", "t_ros"):
        if data[k].shape != (n,):
            raise ValueError(f"{k} shape {data[k].shape}, expected ({n},)")
    angles = data["beam_angle_rad"].astype(np.float64)
    if angles.shape not in ((NUM_BEAMS,), (n, NUM_BEAMS)):
        raise ValueError(f"beam_angle_rad shape {angles.shape}, expected ({NUM_BEAMS},) or ({n},{NUM_BEAMS})")
    return {"ranges": ranges, "forwarded": data["forwarded"] != 0, "angles": angles,
            "t_wall": data["t_wall"].astype(np.float64), "t_ros": data["t_ros"].astype(np.float64)}


def check_timestamps(name, t):
    """Warn about non-finite or non-increasing timestamps (all rows)."""
    if not np.all(np.isfinite(t)):
        warn(f"{name}: {np.count_nonzero(~np.isfinite(t))} non-finite timestamps")
    dt = np.diff(t)
    if np.any(dt[np.isfinite(dt)] < 0):
        warn(f"{name}: timestamps decrease ({np.count_nonzero(dt < 0)} times)")


# ---------------------------------------------------------------- metrics

def timing_mask(t_ros):
    """Rows used for scan timing: drop the pre-episode backlog (t_ros < 0), which arrives in one burst.
    Without a ROS time origin (t_ros all NaN) nothing can be identified, so all rows are kept."""
    if not np.isfinite(t_ros).any():
        return np.ones(t_ros.shape, dtype=bool)
    return ~(t_ros < 0)


def preprocess(ranges):
    """Exactly drl_environment_real.scan_callback, returned in meters (normalized value x cap)."""
    raw = np.where(np.isinf(ranges) | np.isnan(ranges), DISTANCE_CAP, ranges)
    return np.clip((raw - LIDAR_CORRECTION) / DISTANCE_CAP, 0.0, 1.0) * DISTANCE_CAP


def interval_stats(t):
    """Mean/std of consecutive intervals within one episode; NaN with < 2 samples."""
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return np.nan, np.nan
    return float(dt.mean()), float(dt.std(ddof=1)) if dt.size > 1 else np.nan


def duration(t):
    finite = t[np.isfinite(t)]
    return float(finite[-1] - finite[0]) if finite.size else np.nan


def rate(mean_interval):
    return 1.0 / mean_interval if mean_interval > 0 else np.nan


def range_stats(ranges, prefix=""):
    """Distance statistics over one set of (N,40) ranges; the shared schema applies it to the
    policy-input (processed) ranges, the real-only raw_* columns to the recorded ranges."""
    finite = np.isfinite(ranges)
    valid = ranges[finite]
    return {
        f"{prefix}range_mean_m": float(valid.mean()) if valid.size else np.nan,
        f"{prefix}range_std_m": float(valid.std(ddof=1)) if valid.size > 1 else np.nan,
        f"{prefix}invalid_pct": 100.0 * (1 - finite.mean()) if finite.size else np.nan,
        f"{prefix}far_gt_3p5m_pct": 100.0 * np.mean(valid > FAR_THRESHOLD_M) if valid.size else np.nan,
    }


def beam_invalid_pct(ranges):
    return 100.0 * (1 - np.isfinite(ranges).mean(axis=0)) if ranges.shape[0] else np.full(NUM_BEAMS, np.nan)


def circular_mean(angles, axis=0):
    with warnings.catch_warnings():                  # all-NaN columns -> NaN, no noise
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.angle(np.nanmean(np.exp(1j * angles), axis=axis))


def beam_angle_means(angles):
    """Per-beam circular mean, unwrapped along the beam index (beam 0 in [-pi, pi), then increasing),
    so real ([-pi, pi)) and sim ([0, 2pi)) recordings share one convention."""
    if angles.shape[0] == 0:
        return np.full(NUM_BEAMS, np.nan)
    return np.unwrap(circular_mean(angles))


def episode_metrics(episode, policy, d):
    fwd = d["forwarded"]
    timed = timing_mask(d["t_ros"])
    raw = d["ranges"][fwd]
    processed = preprocess(raw)
    angles = d["angles"][fwd] if d["angles"].ndim == 2 else d["angles"][None, :]

    fwd_mean, fwd_std = interval_stats(d["t_wall"][fwd & timed])

    # Shared schema (same names, order and definitions as the simulation analyze_lidar.py):
    # forwarded = fresh, physical time = t_wall, ranges = what the policy received (processed).
    row = {
        "episode": episode, "policy": policy,
        "n_fresh": int(fwd.sum()), "n_fresh_timing": int((fwd & timed).sum()),
        "interval_mean_s": fwd_mean, "interval_std_s": fwd_std, "effective_fps_hz": rate(fwd_mean),
        "lidar_offset_x_m": np.nan,
        **range_stats(processed),
        "at_cap_pct": 100.0 * np.mean(processed >= DISTANCE_CAP - CAP_TOL) if processed.size else np.nan,
        "duration_s": duration(d["t_wall"]),
    }
    beam_angle = beam_angle_means(angles)
    beam_invalid = beam_invalid_pct(processed)
    row.update({f"beam{k:02d}_angle_mean_rad": float(beam_angle[k]) for k in range(NUM_BEAMS)})
    row.update({f"beam{k:02d}_invalid_pct": float(beam_invalid[k]) for k in range(NUM_BEAMS)})

    # Real-only sensor characterization.
    row.update(range_stats(raw, prefix="raw_"))
    raw_invalid = beam_invalid_pct(raw)
    row.update({f"beam{k:02d}_raw_invalid_pct": float(raw_invalid[k]) for k in range(NUM_BEAMS)})
    return row


def unit(metric):
    if metric.endswith("_rad"): return "rad"
    if metric.endswith("_pct"): return "%"
    if metric.endswith("_hz"): return "Hz"
    if metric.endswith("_s"): return "s"
    if metric.endswith("_m"): return "m"
    return "count"


def summarize(df):
    """Equal-weight mean/std across episodes; beam angles use circular mean/std, unwrapped along beams."""
    angle_cols = [c for c in df.columns if c.endswith("_angle_mean_rad")]
    a = df[angle_cols].to_numpy(dtype=np.float64)
    r = np.abs(np.nanmean(np.exp(1j * a), axis=0))
    angle_mean = dict(zip(angle_cols, np.unwrap(circular_mean(a))))
    angle_std = dict(zip(angle_cols, np.sqrt(np.abs(2 * np.log(np.minimum(r, 1.0))))))
    rows = []
    for col in df.columns.drop(["episode", "policy"]):
        x = pd.to_numeric(df[col], errors="coerce")
        n = int(x.notna().sum())
        if col in angle_mean:
            mean, std = angle_mean[col], angle_std[col] if n > 1 else np.nan
        else:
            mean, std = x.mean(), x.std(ddof=1)
        rows.append({"metric": col, "mean": mean, "std": std, "n_episodes": n, "unit": unit(col)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", type=Path, required=True, help="real fixed-FPS evaluation run with episode_*/lidar.npz")
    p.add_argument("--policy", help="configured policy label, e.g. 10Hz (overrides metadata / directory name)")
    args = p.parse_args()

    eval_dir = args.eval_dir
    if not eval_dir.is_dir():
        raise SystemExit(f"Not a directory: {eval_dir}")
    policy = identify_policy(eval_dir, args.policy)
    out_dir = output_dir(eval_dir, policy)
    print(f"Eval dir: {eval_dir}\nPolicy:   {policy}\n"
          f"Preprocessing: cap {DISTANCE_CAP} m, correction {LIDAR_CORRECTION} m (from {SETTINGS.name})")

    rows, skipped = [], []
    for ep_dir in sorted(q for q in eval_dir.glob("episode_*") if q.is_dir()):
        npz = ep_dir / "lidar.npz"
        if not npz.is_file():
            skipped.append((ep_dir.name, "no lidar.npz"))
            continue
        try:
            d = load_lidar(npz)
        except (ValueError, OSError) as e:
            skipped.append((ep_dir.name, str(e)))
            continue
        check_timestamps(f"{ep_dir.name} t_wall", d["t_wall"])
        check_timestamps(f"{ep_dir.name} t_ros", d["t_ros"])
        n_timed = (d["forwarded"] & timing_mask(d["t_ros"])).sum()
        if n_timed < 2:
            warn(f"{ep_dir.name}: {n_timed} forwarded scans after startup, forwarded timing statistics are NaN")
        if not d["forwarded"].any():
            warn(f"{ep_dir.name}: no forwarded scans, range statistics are NaN")
        rows.append(episode_metrics(ep_dir.name, policy, d))

    if not rows:
        raise SystemExit(f"No usable episodes in {eval_dir} (skipped: {skipped})")
    df = pd.DataFrame(rows)
    summary = summarize(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "episode_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    s = summary.set_index("metric")["mean"]
    print(f"Processed {len(rows)} episodes, skipped {len(skipped)}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")
    print(f"  forwarded/episode {s['n_fresh']:.1f} | "
          f"effective FPS {s['effective_fps_hz']:.3f} Hz | raw invalid {s['raw_invalid_pct']:.2f}% | "
          f"raw >3.5 m {s['raw_far_gt_3p5m_pct']:.2f}% | processed mean {s['range_mean_m']:.3f} m")
    print(f"Wrote {out_dir}/episode_metrics.csv and summary.csv")


if __name__ == "__main__":
    main()
