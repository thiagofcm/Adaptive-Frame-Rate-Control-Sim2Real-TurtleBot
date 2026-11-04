#!/usr/bin/env python3
"""Quantitative analysis of Experiment 1 (linear velocity step response, actuator_step_response.py).

Discovers runs/run_<N>/ (the linear-motion runs; angular_run_* and combined_run_* are ignored), computes metrics
for every commanded linear-velocity transition of every run independently, and aggregates them across runs.
Standard library only. No plots, no model fitting.

Inputs per run (written by actuator_step_response.py):
    commands.csv  t_s, ros_time_s, cmd_linear, cmd_angular, phase, forward_m, odom_age_s
    odometry.csv  t_s, ros_time_s, stamp_s, x, y, yaw, v_linear, v_angular, forward_m
    run_info.json schedule, outcome, start_pose, ...

Timeline
    Command times are commands.csv t_s: time.monotonic() right after publish(), relative to the recording start.
    Odometry t_s is the arrival time on the same monotonic origin, but arrivals are jittery (messages bunch, some
    arrive < 1 ms apart). stamp_s is the robot's sampling time (20 Hz, regular) on the robot's clock, which is offset
    from the PC clock by ~0.55 s and must not be compared with command times directly. Each odometry sample is
    therefore placed at
        t = stamp_s - max(stamp_s - t_s)          (one constant per run)
    i.e. the robot clock mapped onto the shared monotonic origin so that the fastest-delivered message keeps its
    arrival time and every other sample is earlier than its arrival time. The remaining unknown is the minimum
    one-way odometry latency, a constant included in every delay (as is the command delivery latency). The offset
    drifts by < 2 ms over a run (checked in the data).
    No filtering or resampling: all metrics use the native odometry samples. Time metrics are quantized to the
    odometry sample period (~0.05 s); delay_resolution_s records the gap before the detected sample.

Transitions
    A transition is a change of cmd_linear between consecutive command segments. Its observation window runs from
    the command time to the next change (or, for the final zero command, to the end of the recording; zero commands
    labelled STOP, SETTLE and EXIT_STOP form one segment). Samples at or after the next change never enter a
    transition's metrics. In aborted runs the window ends at the first EXIT_STOP command. Expected transitions come
    from run_info.json's schedule (pre-roll at 0 m/s, then the schedule); transitions that were never reached get a
    row of NaN with the reason.

Metric definitions (dv = v_new - v_prev, commanded; dev(t) = (v(t) - ref) * sign(dv))
    response_delay_s   first sample after the command at which dev, relative to the measured mean velocity over
                       the BASELINE_S before the command, is >= max(10 % |dv|, NOISE_MPS) on two consecutive samples;
                       delay = that sample's time - command time.
    rise_time_s        commanded 10 %-90 % levels (v_prev + 0.1 dv, v_prev + 0.9 dv): t90 = first sample at or past
                       the 90 % level; t10 = first sample past the 10 % level on the edge leading to t90 (after the last
                       sample still below it). Works for acceleration and deceleration.
    settling_time_s    time from the command to the first sample after which |v - v_new| <= SETTLE_TOL_MPS for every
                       sample over SETTLE_HOLD_S (the hold must fit in the window). NaN if never settled.
    steady_state_error v_new - mean(v) over the last STEADY_S of the window; requires the steady window to start at
                       least TRANSIENT_S after the response onset (or after the command when no onset is needed) and to
                       hold >= MIN_STEADY_SAMPLES samples. abs_steady_state_error is its absolute value.
    overshoot_pct      100 * max(0, max dev relative to v_new) / |dv| over [command, onset + TRANSIENT_S). For a
                       deceleration it is the undershoot below v_new; for a stop, reverse velocity.
    stopping_time_s    (v_new == 0 only) time from the command to the first sample after which |v| <= STOP_MPS for
                       STOP_HOLD_S.
    stopping_dist_m    (v_new == 0 only) odometry path length (sum of |dxy| between samples) from the command to the
                       stopping sample; the position at the command time is linearly interpolated between the two
                       bracketing samples (the only interpolation used).

Run-level metrics: total path length over the whole recording; net displacement and maximum lateral deviation
(|perpendicular offset from the line through the start pose along the start heading|) using run_info start_pose;
recording duration and commanded-motion duration (first non-zero command to the stop command).

Output: analysis/linear/per_transition_metrics.csv, per_run_metrics.csv, summary.csv

Usage:
    python3 RealRobot/motion_dynamic_charac/analyze_linear_motion.py [--runs-root DIR] [--out-dir DIR]
"""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

BASELINE_S = 1.0            # pre-command window for the baseline mean
NOISE_MPS = 0.01            # minimum deviation counted as a response (moving noise sd ~0.003-0.006 m/s)
DETECT_SAMPLES = 2          # consecutive samples beyond the threshold
SETTLE_TOL_MPS = 0.015      # settling band around the commanded velocity (~3 sd of moving noise)
SETTLE_HOLD_S = 0.5         # time the velocity must stay within the band
TRANSIENT_S = 1.0           # after onset: overshoot search window / excluded from the steady-state window
STEADY_S = 1.5              # steady-state averaging window at the end of the phase
MIN_STEADY_SAMPLES = 10
STOP_MPS = 0.01             # near-zero threshold for stopping time
STOP_HOLD_S = 0.5
MAX_GAP_S = 0.1             # odometry stamp gaps longer than this inside a transient are flagged

# Per-transition metrics, (column, unit). Full definitions in the module docstring.
TRANSITION_METRICS = [
    # Dead time: command -> first sample where the velocity has clearly left its pre-command level
    # (>= max(10 % of the step, NOISE_MPS) on DETECT_SAMPLES consecutive samples).
    ("response_delay_s", "s"),
    # Speed of the change itself: time to go from 10 % to 90 % of the commanded step.
    ("rise_time_s", "s"),
    # Command -> velocity enters +/-SETTLE_TOL_MPS of the new command and stays there for SETTLE_HOLD_S.
    ("settling_time_s", "s"),
    # Tracking accuracy once settled: commanded - mean measured velocity over the last STEADY_S of the phase
    # (positive = robot slower than commanded).
    ("steady_state_error_mps", "m/s"),
    ("abs_steady_state_error_mps", "m/s"),
    # Largest excursion past the new command, in the step direction, as % of the step size
    # (undershoot below the target for decelerations, reverse velocity for a stop).
    ("overshoot_pct", "%"),
    # Stop transitions only: command -> |v| <= STOP_MPS held for STOP_HOLD_S.
    ("stopping_time_s", "s"),
    # Stop transitions only: odometry path length travelled between the stop command and the stopping sample.
    ("stopping_distance_m", "m"),
]
# Per-run metrics, (column, unit).
RUN_METRICS = [
    ("duration_s", "s"),                # length of the recording (commands and odometry)
    ("motion_duration_s", "s"),         # first non-zero command -> stop command
    ("total_distance_m", "m"),          # odometry path length over the whole recording
    ("net_displacement_m", "m"),        # straight-line distance from the start pose to the final pose
    ("max_lateral_deviation_m", "m"),   # largest sideways offset from the line along the starting heading
]
NAN = float("nan")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_run(run_dir):
    info = json.loads((run_dir / "run_info.json").read_text())
    cmds = [{"t": float(r["t_s"]), "v": float(r["cmd_linear"]), "phase": r["phase"]}
            for r in read_csv(run_dir / "commands.csv")]
    rows = read_csv(run_dir / "odometry.csv")
    arrival = [float(r["t_s"]) for r in rows]
    stamp = [float(r["stamp_s"]) for r in rows]
    offset = max(s - a for s, a in zip(stamp, arrival)) if rows else NAN
    odom = {"t": [s - offset for s in stamp], "t_arrival": arrival, "stamp": stamp,
            "v": [float(r["v_linear"]) for r in rows],
            "x": [float(r["x"]) for r in rows], "y": [float(r["y"]) for r in rows]}
    return info, cmds, odom, offset


def segments(cmds, completed, t_record_end):
    """Constant-command segments: [{v, t_start, t_end, phases, truncated}] ordered in time."""
    segs = []
    for c in cmds:
        # Abort/interrupt: the experiment ends at the first EXIT_STOP. In a completed run the EXIT_STOP zeros simply
        # extend the STOP segment.
        if c["phase"] == "EXIT_STOP" and not completed:
            if segs:
                segs[-1]["t_end"] = c["t"]
                segs[-1]["truncated"] = True
            break
        if not segs or segs[-1]["v"] != c["v"]:
            if segs:
                segs[-1]["t_end"] = c["t"]
            segs.append({"v": c["v"], "t_start": c["t"], "t_end": t_record_end, "phases": [], "truncated": False})
        if c["phase"] not in segs[-1]["phases"]:
            segs[-1]["phases"].append(c["phase"])
    return segs


def first_hold(t, ok, i0, i1, hold):
    """First index k in [i0, i1) such that ok[j] holds for every j in [k, i1) with t[j] <= t[k] + hold, and the
    window [t[k], t[k] + hold] is covered by samples before i1. None if there is none."""
    for k in range(i0, i1):
        if not ok[k]:
            continue
        j, good = k, True
        while j < i1 and t[j] <= t[k] + hold:
            if not ok[j]:
                good = False
                break
            j += 1
        if good and j < i1:                            # a later sample exists, so the hold was fully observed
            return k
    return None


def path_length(xs, ys):
    return sum(math.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]))


def transition_metrics(odom, t_cmd, t_end, v_prev, v_new):
    """Metrics of one transition from native samples; odom = dict of equal-length lists t, v, x, y.
    Returns (metrics dict, flags list, notes list)."""
    t, v = odom["t"], odom["v"]
    m = {name: NAN for name, _ in TRANSITION_METRICS}
    m.update({"baseline_mps": NAN, "peak_mps": NAN, "delay_resolution_s": NAN, "n_samples": 0})
    flags, notes = [], []
    dv = v_new - v_prev
    sgn = 1.0 if dv > 0 else -1.0
    stop = v_new == 0.0

    i0 = next((i for i, ti in enumerate(t) if ti >= t_cmd), len(t))
    i1 = next((i for i, ti in enumerate(t) if ti >= t_end), len(t))
    m["n_samples"] = i1 - i0
    if i1 - i0 < DETECT_SAMPLES:
        notes.append("all: fewer than 2 odometry samples in the transition window")
        return m, flags + ["no_data"], notes

    # Baseline: measured mean over the BASELINE_S before the command.
    base = [v[i] for i in range(i0) if t[i] >= t_cmd - BASELINE_S]
    if len(base) >= 3:
        m["baseline_mps"] = statistics.fmean(base)
    else:
        m["baseline_mps"] = v_prev
        flags.append("baseline_from_command")

    # Response delay.
    thr = max(0.1 * abs(dv), NOISE_MPS)
    above = [(v[i] - m["baseline_mps"]) * sgn >= thr for i in range(len(t))]
    onset = next((i for i in range(i0, i1 - DETECT_SAMPLES + 1) if all(above[i:i + DETECT_SAMPLES])), None)
    if onset is None:
        notes.append("response_delay/rise/overshoot: no response detected before the next command")
        flags.append("no_response")
    else:
        m["response_delay_s"] = t[onset] - t_cmd
        m["delay_resolution_s"] = t[onset] - t[onset - 1] if onset > 0 else NAN
        if any(t[i] - t[i - 1] > MAX_GAP_S for i in range(max(i0, 1), onset + 1)):
            flags.append("odom_gap_before_onset")

    # Rise time (commanded 10 %-90 % levels). The rise often spans a single 50 ms odometry interval, so each level
    # crossing is placed by linear interpolation between the two native samples that bracket it.
    dev_cmd = [(v[i] - v_prev) * sgn for i in range(len(t))]

    def crossing(i, level):
        if i == 0 or dev_cmd[i - 1] >= level:
            return t[i]
        return t[i - 1] + (level - dev_cmd[i - 1]) / (dev_cmd[i] - dev_cmd[i - 1]) * (t[i] - t[i - 1])

    i90 = next((i for i in range(i0, i1) if dev_cmd[i] >= 0.9 * abs(dv)), None)
    if i90 is None:
        notes.append("rise_time: 90 % level not reached before the next command")
    else:
        i10 = i90
        while i10 - 1 >= i0 and dev_cmd[i10 - 1] >= 0.1 * abs(dv):
            i10 -= 1
        if i10 == i0 and dev_cmd[i0] >= 0.1 * abs(dv):
            flags.append("rise_started_before_first_sample")
        m["rise_time_s"] = crossing(i90, 0.9 * abs(dv)) - max(crossing(i10, 0.1 * abs(dv)), t_cmd)
        if i90 - i10 <= 1:
            flags.append("rise_spans_le_2_samples")

    # Overshoot over [command, onset + TRANSIENT_S).
    t_os_end = min(t_end, t[onset] + TRANSIENT_S) if onset is not None else t_end
    win = [i for i in range(i0, i1) if t[i] < t_os_end]
    if onset is not None:
        peak = max(win, key=lambda i: (v[i] - v_new) * sgn)
        m["peak_mps"] = v[peak]
        m["overshoot_pct"] = 100.0 * max(0.0, (v[peak] - v_new) * sgn) / abs(dv)
        if t_os_end < t[onset] + TRANSIENT_S:
            flags.append("overshoot_window_truncated")

    # Settling time.
    k = first_hold(t, [abs(vi - v_new) <= SETTLE_TOL_MPS for vi in v], i0, i1, SETTLE_HOLD_S)
    if k is None:
        notes.append(f"settling_time: not within +/-{SETTLE_TOL_MPS} m/s for {SETTLE_HOLD_S} s before the next command")
    else:
        m["settling_time_s"] = t[k] - t_cmd

    # Steady-state error over the last STEADY_S of the window.
    t_ss_start = (t[i1 - 1] if t_end == math.inf else t_end) - STEADY_S
    earliest = (t[onset] if onset is not None else t_cmd) + TRANSIENT_S
    ss = [v[i] for i in range(i0, i1) if t[i] >= t_ss_start]
    if onset is None:
        notes.append("steady_state_error: no response detected")
    elif t_ss_start < earliest:
        notes.append(f"steady_state_error: phase too short (steady window would start {earliest - t_ss_start:.2f} s"
                     f" inside the transient)")
    elif len(ss) < MIN_STEADY_SAMPLES:
        notes.append(f"steady_state_error: only {len(ss)} samples in the steady window")
    else:
        m["steady_state_error_mps"] = v_new - statistics.fmean(ss)
        m["abs_steady_state_error_mps"] = abs(m["steady_state_error_mps"])

    # Stopping time and distance.
    if stop:
        k = first_hold(t, [abs(vi) <= STOP_MPS for vi in v], i0, i1, STOP_HOLD_S)
        if k is None:
            notes.append(f"stopping: |v| did not stay <= {STOP_MPS} m/s for {STOP_HOLD_S} s within the recording")
        else:
            m["stopping_time_s"] = t[k] - t_cmd
            x, y = odom["x"], odom["y"]
            if i0 == 0:
                x_c, y_c = x[0], y[0]
                flags.append("no_sample_before_stop_command")
            else:
                a = (t_cmd - t[i0 - 1]) / (t[i0] - t[i0 - 1])
                x_c, y_c = x[i0 - 1] + a * (x[i0] - x[i0 - 1]), y[i0 - 1] + a * (y[i0] - y[i0 - 1])
            m["stopping_distance_m"] = path_length([x_c] + x[i0:k + 1], [y_c] + y[i0:k + 1])
    return m, flags, notes


def fmt(x):
    if isinstance(x, float):
        return "NaN" if math.isnan(x) else f"{x:.6g}" if abs(x) < 1e5 else f"{x:.6f}"
    return x


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in fields})


def transition_id(index, v_prev, v_new):
    return f"T{index}_{v_prev:.2f}->{v_new:.2f}"


def analyze_run(run_dir):
    info, cmds, odom, offset = load_run(run_dir)
    run = run_dir.name
    outcome = info.get("outcome", "UNKNOWN")
    completed = outcome == "COMPLETED"
    t_record_end = max([c["t"] for c in cmds] + odom["t_arrival"]) if cmds or odom["t"] else NAN

    expected = [0.0] + [float(p[2]) for p in info.get("schedule", [])]
    # A completed run's final zero command is observed to the end of the recording.
    segs = segments(cmds, completed, math.inf if completed else t_record_end)

    rows = []
    for n in range(1, len(expected)):
        v_prev, v_new = expected[n - 1], expected[n]
        row = {"run": run, "transition_id": transition_id(n, v_prev, v_new), "transition_index": n,
               "v_prev_cmd_mps": v_prev, "v_new_cmd_mps": v_new, "delta_v_cmd_mps": v_new - v_prev}
        flags, notes = [], []
        if n >= len(segs) or segs[n]["v"] != v_new or segs[n - 1]["v"] != v_prev:
            row.update({name: NAN for name, _ in TRANSITION_METRICS})
            row.update({"executed": False, "phase": "", "t_cmd_s": NAN, "t_phase_end_s": NAN, "phase_duration_s": NAN})
            if n >= len(segs):
                notes.append(f"not executed: run {outcome}")
            else:
                notes.append(f"recorded command sequence differs from the schedule at this transition")
                flags.append("unexpected_sequence")
        else:
            seg = segs[n]
            t_end = seg["t_end"]
            m, flags, notes = transition_metrics(odom, seg["t_start"], t_end, v_prev, v_new)
            row.update(m)
            t_last = t_end if t_end != math.inf else odom["t"][-1]
            row.update({"executed": True, "phase": "+".join(seg["phases"]), "t_cmd_s": seg["t_start"],
                        "t_phase_end_s": t_last, "phase_duration_s": t_last - seg["t_start"]})
            if seg["truncated"]:
                flags.append("phase_truncated_by_abort")
        row["flags"] = ";".join(flags)
        row["notes"] = " | ".join(notes)
        rows.append(row)

    # Run-level metrics.
    start = info.get("start_pose") or ({"x": odom["x"][0], "y": odom["y"][0], "yaw": 0.0} if odom["x"] else None)
    run_row = {"run": run, "outcome": outcome, "completed": completed, "n_cmd": len(cmds), "n_odom": len(odom["t"]),
               "clock_offset_s": offset}
    run_row["duration_s"] = t_record_end
    moving = [c["t"] for c in cmds if c["v"] != 0.0]
    t_stop = next((s["t_start"] for s in segs[1:] if s["v"] == 0.0), NAN)
    run_row["motion_duration_s"] = (t_stop - moving[0]) if moving and not math.isnan(t_stop) else NAN
    if start and odom["x"]:
        x0, y0, yaw0 = start["x"], start["y"], start["yaw"]
        run_row["total_distance_m"] = path_length(odom["x"], odom["y"])
        run_row["net_displacement_m"] = math.hypot(odom["x"][-1] - x0, odom["y"][-1] - y0)
        lat = [-(x - x0) * math.sin(yaw0) + (y - y0) * math.cos(yaw0) for x, y in zip(odom["x"], odom["y"])]
        i = max(range(len(lat)), key=lambda j: abs(lat[j]))
        run_row["max_lateral_deviation_m"] = abs(lat[i])
        run_row["lateral_at_max_signed_m"] = lat[i]
    else:
        for k in ("total_distance_m", "net_displacement_m", "max_lateral_deviation_m", "lateral_at_max_signed_m"):
            run_row[k] = NAN
    run_row["n_transitions_expected"] = len(expected) - 1
    run_row["n_transitions_executed"] = sum(r["executed"] for r in rows)
    run_row["n_valid_transitions"] = sum(r["executed"] and not math.isnan(r["response_delay_s"]) for r in rows)
    run_row["notes"] = "" if completed else f"{outcome}; reached {len(segs)} command segment(s)"
    return rows, run_row


def summarize(trans_rows, run_rows):
    out = []
    ids = list(dict.fromkeys(r["transition_id"] for r in trans_rows))
    for tid in ids:
        rs = [r for r in trans_rows if r["transition_id"] == tid]
        for name, unit in TRANSITION_METRICS:
            if name.startswith("stopping") and rs[0]["v_new_cmd_mps"] != 0.0:
                continue                                # only defined for stop transitions
            vals = [(r["run"], r[name]) for r in rs if not math.isnan(r[name])]
            out.append(aggregate(tid, rs[0]["delta_v_cmd_mps"], rs[0]["v_prev_cmd_mps"], rs[0]["v_new_cmd_mps"],
                                 name, unit, vals, len(rs)))
    for name, unit in RUN_METRICS:
        vals = [(r["run"], r[name]) for r in run_rows if r["completed"] and not math.isnan(r.get(name, NAN))]
        out.append(aggregate("RUN(completed runs)", NAN, NAN, NAN, name, unit, vals, len(run_rows)))
    return out


def aggregate(tid, dv, v_prev, v_new, name, unit, vals, n_total):
    x = [v for _, v in vals]
    return {"transition_id": tid, "v_prev_cmd_mps": v_prev, "v_new_cmd_mps": v_new, "delta_v_cmd_mps": dv,
            "metric": name, "unit": unit,
            "mean": statistics.fmean(x) if x else NAN,
            "std": statistics.stdev(x) if len(x) >= 2 else NAN,
            "min": min(x) if x else NAN, "max": max(x) if x else NAN,
            "n_valid_runs": len(x), "n_runs": n_total, "runs": ";".join(r for r, _ in vals)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", default=str(HERE / "runs"))
    parser.add_argument("--out-dir", default=str(HERE / "analysis" / "linear"))
    args = parser.parse_args()

    run_dirs = sorted((d for d in Path(args.runs_root).iterdir() if d.is_dir() and re.fullmatch(r"run_\d+", d.name)),
                      key=lambda d: int(d.name[4:]))
    trans_rows, run_rows = [], []
    for d in run_dirs:
        if not all((d / f).exists() for f in ("commands.csv", "odometry.csv", "run_info.json")):
            run_rows.append({"run": d.name, "outcome": "MISSING_FILES", "completed": False,
                             "notes": "commands.csv, odometry.csv or run_info.json missing"})
            continue
        rows, run_row = analyze_run(d)
        trans_rows += rows
        run_rows.append(run_row)
        print(f"{d.name}: {run_row['outcome']}, {run_row['n_transitions_executed']}/"
              f"{run_row['n_transitions_expected']} transitions executed")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trans_fields = ["run", "transition_id", "transition_index", "phase", "executed", "v_prev_cmd_mps",
                    "v_new_cmd_mps", "delta_v_cmd_mps", "t_cmd_s", "t_phase_end_s", "phase_duration_s", "n_samples",
                    "baseline_mps", "peak_mps"] + [n for n, _ in TRANSITION_METRICS] + \
                   ["delay_resolution_s", "flags", "notes"]
    run_fields = ["run", "outcome", "completed", "duration_s", "motion_duration_s", "total_distance_m",
                  "net_displacement_m", "max_lateral_deviation_m", "lateral_at_max_signed_m",
                  "n_transitions_expected", "n_transitions_executed", "n_valid_transitions", "n_cmd", "n_odom",
                  "clock_offset_s", "notes"]
    sum_fields = ["transition_id", "v_prev_cmd_mps", "v_new_cmd_mps", "delta_v_cmd_mps", "metric", "unit", "mean",
                  "std", "min", "max", "n_valid_runs", "n_runs", "runs"]
    write_csv(out / "per_transition_metrics.csv", trans_fields, trans_rows)
    write_csv(out / "per_run_metrics.csv", run_fields, run_rows)
    write_csv(out / "summary.csv", sum_fields, summarize(trans_rows, run_rows))
    print(f"Wrote {out}/per_transition_metrics.csv, per_run_metrics.csv, summary.csv")


if __name__ == "__main__":
    main()
