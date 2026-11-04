#!/usr/bin/env python3
"""Quantitative analysis of Experiment 3 (combined linear + angular step response, combined_motion_response.py).

Companion of analyze_linear_motion.py and analyze_angular_motion.py: same timeline, transition windows, onset
detection, metric definitions, NaN policy and CSV layout. Each transition (v, w) -> (v', w') is analyzed twice,
once on the measured linear velocity (odometry v_linear) and once on the measured angular velocity (v_angular),
and both results stay on the same row so the two responses to one command event can be compared.
Standard library only for the analysis; plots need matplotlib and are skipped (with a message) without it.

Inputs per run (written by combined_motion_response.py):
    commands.csv  t_s, ros_time_s, cmd_linear, cmd_angular, phase, ...
    odometry.csv  t_s, ros_time_s, stamp_s, x, y, yaw, yaw_unwrapped, x_s, y_s, rotation_rad, v_linear, v_angular
    run_info.json mode, sequence, outcome, start_pose, ideal_phase_ends_start_frame, ...

Timeline, transitions, aborted runs: exactly as in the angular analyzer (odometry stamps mapped onto the command
clock with one offset per run; a transition is a change of the (v, w) command pair; the final zero command of a
completed run is observed to the end of the recording; in aborted runs the running phase ends at the first
EXIT_STOP, which is not a transition; never-reached transitions get NaN rows). Heading unwrapped from raw yaw.

Metric definitions: identical to the single-axis analyzers, applied per component with that component's own
thresholds (linear: analyze_linear_motion.py, angular: analyze_angular_motion.py):
    response_delay_s, rise_time_s (10-90 %, interpolated crossings), settling_time_s, steady_state_error,
    abs_steady_state_error, tracking_ratio (mean measured / command, non-zero commands), overshoot_pct,
    stopping_time_s; plus stopping_distance_m (linear) and stopping_angle_rad/deg (angular) for stops.
Differences from the single-axis analyzers (stated, not silent):
    - Windows: TRANSIENT_S = 0.8 s and STEADY_S = 1.0 s for BOTH components (the angular analyzer's values). The
      linear analyzer used 1.0 / 1.5 s, which does not fit the 2 s stop phases; using one window pair keeps the
      linear and angular metrics of the same event comparable.
    - Baseline: median of the BASELINE_S before the command for both components (angular analyzer); equal to the
      mean for a steady baseline, robust to a previous transient still in the window.
    - The linear analyzer had no tracking_ratio; it is added here for the linear component too.
    - A component whose command does not change keeps steady-state error / tracking ratio, but its delay, rise,
      settling and overshoot are NaN (flag <comp>:unchanged_command).
    - The angular-only "in_place" / pivot classification is not used (translation is commanded here). Instead,
      <comp>:tracking_anomaly flags a steady tracking ratio outside 1 +/- TRACKING_ANOMALY.
Paired metric: delta_tau_s = angular response_delay_s - linear response_delay_s (> 0: angular starts later),
only when both delays are valid. Similar or correlated delays do not by themselves prove a shared mechanism.
Per-phase motion: lin_path_length_m vs lin_cmd_path_length_m (|v'| x phase duration) and ang_heading_change_rad
vs ang_cmd_heading_change_rad (w' x phase duration) quantify what the transients cost against the ideal unicycle.

Run-level metrics: durations, odometry path length, net displacement, final pose in the start frame (x_s forward,
y_s left) and heading change, the ideal final pose from run_info and the errors to it, total |rotation|,
max |v| and |w|.

Output: analysis/combined/per_transition_metrics.csv, per_run_metrics.csv, summary.csv, plots/*.png
    summary.csv: one row per transition, component (linear / angular / pair) and metric; cmd_linear_mps and
    cmd_angular_radps are the commands given at that transition (the previous ones are in transition_id).

Usage:
    python3 RealRobot/motion_dynamic_charac/analyze_combined_motion.py [--runs-root DIR] [--out-dir DIR] [--no-plots]
"""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from analyze_linear_motion import first_hold, path_length, read_csv, write_csv

HERE = Path(__file__).resolve().parent

# Shared with both single-axis analyzers.
BASELINE_S = 1.0            # pre-command window for the baseline median
DETECT_SAMPLES = 2          # consecutive samples beyond the detection threshold
SETTLE_HOLD_S = 0.5         # time the signal must stay within the settling band
STOP_HOLD_S = 0.5           # time the signal must stay near zero to count as stopped
TRANSIENT_S = 0.8           # after onset: overshoot window / excluded from the steady window (angular analyzer)
STEADY_S = 1.0              # steady-state window at the end of the phase (angular analyzer)
MIN_STEADY_SAMPLES = 10
MAX_GAP_S = 0.1             # odometry stamp gaps longer than this before the onset are flagged
TRACKING_ANOMALY = 0.2      # flag a steady tracking ratio outside 1 +/- this
# Per component: thresholds of the corresponding single-axis analyzer.
COMPONENTS = {
    "lin": {"name": "linear", "unit": "m/s", "noise": 0.01, "settle_tol": 0.015, "stop": 0.01},
    "ang": {"name": "angular", "unit": "rad/s", "noise": 0.05, "settle_tol": 0.08, "stop": 0.02},
}

# Per-component metrics, (suffix, unit or None = the component's velocity unit). Definitions in the docstring.
COMPONENT_METRICS = [
    ("response_delay_s", "s"),          # command -> clear departure from the pre-command level
    ("rise_time_s", "s"),               # 10 % -> 90 % of the commanded step
    ("settling_time_s", "s"),           # command -> within the settling band for SETTLE_HOLD_S
    ("steady_state_error", None),       # command - mean measured over the steady window
    ("abs_steady_state_error", None),
    ("tracking_ratio", "-"),            # mean measured / command at the end of the phase
    ("overshoot_pct", "%"),             # excursion past the new command, % of the step
    ("stopping_time_s", "s"),           # stops only: command -> near zero for STOP_HOLD_S
]
LIN_EXTRA = [("stopping_distance_m", "m"),          # stops only: odometry path from command to stopped
             ("path_length_m", "m"),                # odometry path length during the phase
             ("cmd_path_length_m", "m")]            # |v'| x phase duration (ideal)
ANG_EXTRA = [("stopping_angle_rad", "rad"),         # stops only: heading change from command to stopped (+ = CCW)
             ("abs_stopping_angle_rad", "rad"),
             ("stopping_angle_deg", "deg"),
             ("heading_change_rad", "rad"),         # measured heading change during the phase
             ("cmd_heading_change_rad", "rad")]     # w' x phase duration (ideal)
PAIR_METRICS = [("delta_tau_s", "s")]               # angular delay - linear delay
RUN_METRICS = [
    ("duration_s", "s"), ("motion_duration_s", "s"),
    ("total_distance_m", "m"),                      # odometry path length
    ("net_displacement_m", "m"),                    # start -> final position
    ("final_x_s_m", "m"), ("final_y_s_m", "m"),     # final position, start frame (forward, left)
    ("final_heading_change_deg", "deg"),
    ("ideal_final_x_s_m", "m"), ("ideal_final_y_s_m", "m"),
    ("final_position_error_m", "m"),                # |final - ideal final position|
    ("final_heading_error_deg", "deg"),             # final - ideal final heading
    ("total_rotation_deg", "deg"),                  # accumulated |heading change|
    ("max_abs_v_linear_mps", "m/s"), ("max_abs_angular_radps", "rad/s"),
]
NAN = float("nan")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def unwrap(yaw):
    out = [yaw[0]] if yaw else []
    for a, b in zip(yaw, yaw[1:]):
        out.append(out[-1] + wrap(b - a))
    return out


def metric_columns():
    cols = [("lin_" + m, u or COMPONENTS["lin"]["unit"]) for m, u in COMPONENT_METRICS + LIN_EXTRA]
    cols += [("ang_" + m, u or COMPONENTS["ang"]["unit"]) for m, u in COMPONENT_METRICS + ANG_EXTRA]
    return cols + PAIR_METRICS


def load_run(run_dir):
    info = json.loads((run_dir / "run_info.json").read_text())
    cmds = [{"t": float(r["t_s"]), "v": float(r["cmd_linear"]), "w": float(r["cmd_angular"]), "phase": r["phase"]}
            for r in read_csv(run_dir / "commands.csv")]
    rows = read_csv(run_dir / "odometry.csv")
    arrival = [float(r["t_s"]) for r in rows]
    stamp = [float(r["stamp_s"]) for r in rows]
    offset = max(s - a for s, a in zip(stamp, arrival)) if rows else NAN
    odom = {"t": [s - offset for s in stamp], "t_arrival": arrival,
            "v": [float(r["v_linear"]) for r in rows], "w": [float(r["v_angular"]) for r in rows],
            "x": [float(r["x"]) for r in rows], "y": [float(r["y"]) for r in rows],
            "yaw": unwrap([float(r["yaw"]) for r in rows])}
    return info, cmds, odom, offset


def segments(cmds, completed, t_record_end):
    """Constant-command segments: [{v, w, t_start, t_end, phases, truncated}] ordered in time."""
    segs = []
    for c in cmds:
        # Abort/interrupt: the experiment ends at the first EXIT_STOP. In a completed run the EXIT_STOP zeros simply
        # extend the final stop segment.
        if c["phase"] == "EXIT_STOP" and not completed:
            if segs:
                segs[-1]["t_end"] = c["t"]
                segs[-1]["truncated"] = True
            break
        if not segs or (segs[-1]["v"], segs[-1]["w"]) != (c["v"], c["w"]):
            if segs:
                segs[-1]["t_end"] = c["t"]
            segs.append({"v": c["v"], "w": c["w"], "t_start": c["t"], "t_end": t_record_end, "phases": [],
                         "truncated": False})
        if c["phase"] not in segs[-1]["phases"]:
            segs[-1]["phases"].append(c["phase"])
    return segs


def interp_at(t, vals, i0, t_q):
    """Value of vals at t_q, linear between samples i0 - 1 and i0 (t[i0 - 1] < t_q <= t[i0])."""
    if i0 == 0:
        return vals[0]
    a = (t_q - t[i0 - 1]) / (t[i0] - t[i0 - 1])
    return vals[i0 - 1] + a * (vals[i0] - vals[i0 - 1])


def component_metrics(t, y, t_cmd, t_end, prev, new, p):
    """Single-axis metrics of signal y (same definitions as the single-axis analyzers) with thresholds p.
    Returns (metrics, flags, notes, i_stop) where i_stop is the stopping sample index (or None)."""
    m = {name: NAN for name, _ in COMPONENT_METRICS}
    m.update({"baseline": NAN, "peak": NAN})
    flags, notes = [], []
    d = new - prev
    changed = d != 0.0
    sgn = 1.0 if d >= 0 else -1.0

    i0 = next((i for i, ti in enumerate(t) if ti >= t_cmd), len(t))
    i1 = next((i for i, ti in enumerate(t) if ti >= t_end), len(t))
    if i1 - i0 < DETECT_SAMPLES:
        notes.append("all: fewer than 2 odometry samples in the transition window")
        return m, flags + ["no_data"], notes, None
    idx = range(i0, i1)
    t_last = t[i1 - 1] if t_end == math.inf else t_end

    base = [y[i] for i in range(i0) if t[i] >= t_cmd - BASELINE_S]
    if len(base) >= 3:
        m["baseline"] = statistics.median(base)
    else:
        m["baseline"] = prev
        flags.append("baseline_from_command")

    onset = None
    if not changed:
        flags.append("unchanged_command")
        notes.append("response_delay/rise/settling/overshoot: command did not change")
    else:
        thr = max(0.1 * abs(d), p["noise"])
        above = [(y[i] - m["baseline"]) * sgn >= thr for i in range(len(t))]
        onset = next((i for i in range(i0, i1 - DETECT_SAMPLES + 1) if all(above[i:i + DETECT_SAMPLES])), None)
        if onset is None:
            notes.append("response_delay/rise/overshoot: no response detected before the next command")
            flags.append("no_response")
        else:
            m["response_delay_s"] = t[onset] - t_cmd
            if any(t[i] - t[i - 1] > MAX_GAP_S for i in range(max(i0, 1), onset + 1)):
                flags.append("odom_gap_before_onset")

        # Rise time: commanded 10/90 % levels, crossings interpolated between bracketing native samples.
        dev = [(y[i] - prev) * sgn for i in range(len(t))]

        def crossing(i, level):
            if i == 0 or dev[i - 1] >= level:
                return t[i]
            return t[i - 1] + (level - dev[i - 1]) / (dev[i] - dev[i - 1]) * (t[i] - t[i - 1])

        i90 = next((i for i in idx if dev[i] >= 0.9 * abs(d)), None)
        if i90 is None:
            notes.append("rise_time: 90 % level not reached before the next command")
        else:
            i10 = i90
            while i10 - 1 >= i0 and dev[i10 - 1] >= 0.1 * abs(d):
                i10 -= 1
            m["rise_time_s"] = crossing(i90, 0.9 * abs(d)) - max(crossing(i10, 0.1 * abs(d)), t_cmd)
            if i90 - i10 <= 1:
                flags.append("rise_spans_le_2_samples")

        if onset is not None:
            t_os_end = min(t_end, t[onset] + TRANSIENT_S)
            peak = max((i for i in idx if t[i] < t_os_end), key=lambda i: (y[i] - new) * sgn)
            m["peak"] = y[peak]
            m["overshoot_pct"] = 100.0 * max(0.0, (y[peak] - new) * sgn) / abs(d)

        k = first_hold(t, [abs(yi - new) <= p["settle_tol"] for yi in y], i0, i1, SETTLE_HOLD_S)
        if k is None:
            notes.append(f"settling_time: not within +/-{p['settle_tol']} for {SETTLE_HOLD_S} s before the next command")
        else:
            m["settling_time_s"] = t[k] - t_cmd

    # Steady state over the last STEADY_S of the window (reference: onset, or the command if unchanged).
    t_ss_start = t_last - STEADY_S
    ss = [y[i] for i in idx if t[i] >= t_ss_start]
    if changed and onset is None:
        notes.append("steady_state_error: no response detected")
    elif t_ss_start < (t[onset] if onset is not None else t_cmd) + TRANSIENT_S:
        notes.append("steady_state_error: window too short for a steady window after the transient")
    elif len(ss) < MIN_STEADY_SAMPLES:
        notes.append(f"steady_state_error: only {len(ss)} samples in the steady window")
    else:
        mean = statistics.fmean(ss)
        m["steady_state_error"] = new - mean
        m["abs_steady_state_error"] = abs(new - mean)
        if new != 0.0:
            m["tracking_ratio"] = mean / new
            if abs(m["tracking_ratio"] - 1.0) > TRACKING_ANOMALY:
                flags.append("tracking_anomaly")

    i_stop = None
    if new == 0.0 and prev != 0.0:
        i_stop = first_hold(t, [abs(yi) <= p["stop"] for yi in y], i0, i1, STOP_HOLD_S)
        if i_stop is None:
            notes.append(f"stopping: did not stay within {p['stop']} of 0 for {STOP_HOLD_S} s within the window")
        else:
            m["stopping_time_s"] = t[i_stop] - t_cmd
    return m, flags, notes, i_stop


def transition_metrics(odom, t_cmd, t_end, v_prev, v_new, w_prev, w_new):
    """Linear and angular metrics of one (v, w) transition. Returns (metrics with lin_/ang_ prefixes, flags, notes)."""
    t = odom["t"]
    out = {name: NAN for name, _ in metric_columns()}
    flags, notes = [], []
    i0 = next((i for i, ti in enumerate(t) if ti >= t_cmd), len(t))
    i1 = next((i for i, ti in enumerate(t) if ti >= t_end), len(t))
    t_last = t[i1 - 1] if t_end == math.inf and i1 > 0 else t_end
    out["n_samples"] = i1 - i0

    for comp, sig, prev, new in (("lin", "v", v_prev, v_new), ("ang", "w", w_prev, w_new)):
        m, f, n, i_stop = component_metrics(t, odom[sig], t_cmd, t_end, prev, new, COMPONENTS[comp])
        for key, val in m.items():
            out[f"{comp}_{key}"] = val
        flags += [f"{comp}:{x}" for x in f]
        notes += [f"{comp}: {x}" for x in n]
        if "no_data" in f:
            continue
        # Phase totals run from the pose interpolated at the command to the pose interpolated at the next command
        # (the last sample when the recording ends first), matching the commanded |v'| or w' x duration.
        ends_inside = i1 < len(t)
        if comp == "lin":
            xc, yc = interp_at(t, odom["x"], i0, t_cmd), interp_at(t, odom["y"], i0, t_cmd)
            xe = [interp_at(t, odom["x"], i1, t_end)] if ends_inside else []
            ye = [interp_at(t, odom["y"], i1, t_end)] if ends_inside else []
            out["lin_path_length_m"] = path_length([xc] + odom["x"][i0:i1] + xe, [yc] + odom["y"][i0:i1] + ye)
            out["lin_cmd_path_length_m"] = abs(v_new) * (t_last - t_cmd)
            if i_stop is not None:
                out["lin_stopping_distance_m"] = path_length([xc] + odom["x"][i0:i_stop + 1],
                                                             [yc] + odom["y"][i0:i_stop + 1])
        else:
            yaw_c = interp_at(t, odom["yaw"], i0, t_cmd)
            yaw_e = interp_at(t, odom["yaw"], i1, t_end) if ends_inside else odom["yaw"][i1 - 1]
            out["ang_heading_change_rad"] = yaw_e - yaw_c
            out["ang_cmd_heading_change_rad"] = w_new * (t_last - t_cmd)
            if i_stop is not None:
                ang = odom["yaw"][i_stop] - yaw_c
                out["ang_stopping_angle_rad"], out["ang_abs_stopping_angle_rad"] = ang, abs(ang)
                out["ang_stopping_angle_deg"] = math.degrees(ang)
    out["delta_tau_s"] = out["ang_response_delay_s"] - out["lin_response_delay_s"]     # NaN unless both valid
    return out, flags, notes


def transition_id(index, v_prev, w_prev, v_new, w_new):
    return f"T{index}_({v_prev:.2f},{w_prev:+.2f})->({v_new:.2f},{w_new:+.2f})"


def start_frame(info, odom):
    """(x_s forward, y_s left, heading change) of every sample relative to run_info start_pose."""
    s = info.get("start_pose") or {"x": odom["x"][0], "y": odom["y"][0], "yaw": 0.0}
    c, sn = math.cos(s["yaw"]), math.sin(s["yaw"])
    xs = [(x - s["x"]) * c + (y - s["y"]) * sn for x, y in zip(odom["x"], odom["y"])]
    ys = [-(x - s["x"]) * sn + (y - s["y"]) * c for x, y in zip(odom["x"], odom["y"])]
    rot = [a - odom["yaw"][0] for a in odom["yaw"]]
    return xs, ys, rot


def ideal_path(segs, t_end, substeps=200):
    """Ideal unicycle path (start frame) under the recorded command segments, exact arcs."""
    x = y = th = 0.0
    px, py = [0.0], [0.0]
    for s in segs:
        dur = min(s["t_end"], t_end) - s["t_start"]
        dt = dur / substeps
        for _ in range(substeps):
            if abs(s["w"]) > 1e-12:
                x += s["v"] / s["w"] * (math.sin(th + s["w"] * dt) - math.sin(th))
                y -= s["v"] / s["w"] * (math.cos(th + s["w"] * dt) - math.cos(th))
            else:
                x += s["v"] * math.cos(th) * dt
                y += s["v"] * math.sin(th) * dt
            th += s["w"] * dt
            px.append(x)
            py.append(y)
    return px, py


def analyze_run(run_dir):
    info, cmds, odom, offset = load_run(run_dir)
    run = run_dir.name
    outcome = info.get("outcome", "UNKNOWN")
    completed = outcome == "COMPLETED"
    t_record_end = max([c["t"] for c in cmds] + odom["t_arrival"]) if cmds or odom["t"] else NAN

    expected = [(p[0], float(p[2]), float(p[3])) for p in info.get("sequence", [])]
    expected = [p for i, p in enumerate(expected) if i == 0 or p[1:] != expected[i - 1][1:]]
    segs = segments(cmds, completed, math.inf if completed else t_record_end)

    rows = []
    for n in range(1, len(expected)):
        (_, v_prev, w_prev), (phase_name, v_new, w_new) = expected[n - 1], expected[n]
        row = {"run": run, "transition_id": transition_id(n, v_prev, w_prev, v_new, w_new), "transition_index": n,
               "scheduled_phase": phase_name, "v_prev_cmd_mps": v_prev, "v_new_cmd_mps": v_new,
               "delta_v_cmd_mps": v_new - v_prev, "w_prev_cmd_radps": w_prev, "w_new_cmd_radps": w_new,
               "delta_w_cmd_radps": w_new - w_prev}
        flags, notes = [], []
        if n >= len(segs) or (segs[n]["v"], segs[n]["w"]) != (v_new, w_new) \
                or (segs[n - 1]["v"], segs[n - 1]["w"]) != (v_prev, w_prev):
            row.update({name: NAN for name, _ in metric_columns()})
            row.update({"executed": False, "phase": "", "t_cmd_s": NAN, "t_phase_end_s": NAN,
                        "phase_duration_s": NAN, "n_samples": 0})
            if n >= len(segs):
                notes.append(f"not executed: run {outcome}")
            else:
                notes.append("recorded command sequence differs from the schedule at this transition")
                flags.append("unexpected_sequence")
        else:
            seg = segs[n]
            m, flags, notes = transition_metrics(odom, seg["t_start"], seg["t_end"], v_prev, v_new, w_prev, w_new)
            row.update(m)
            t_last = seg["t_end"] if seg["t_end"] != math.inf else odom["t"][-1]
            row.update({"executed": True, "phase": "+".join(seg["phases"]), "t_cmd_s": seg["t_start"],
                        "t_phase_end_s": t_last, "phase_duration_s": t_last - seg["t_start"]})
            if seg["truncated"]:
                flags.append("phase_truncated_by_abort")
        row["flags"] = ";".join(flags)
        row["notes"] = " | ".join(notes)
        rows.append(row)

    run_row = {"run": run, "mode": info.get("mode", ""), "outcome": outcome, "completed": completed,
               "n_cmd": len(cmds), "n_odom": len(odom["t"]), "clock_offset_s": offset, "duration_s": t_record_end}
    moving = [c["t"] for c in cmds if (c["v"], c["w"]) != (0.0, 0.0) and c["phase"] != "EXIT_STOP"]
    t_end_motion = segs[-1]["t_start"] if segs and (segs[-1]["v"], segs[-1]["w"]) == (0.0, 0.0) \
        else (segs[-1]["t_end"] if segs else NAN)
    run_row["motion_duration_s"] = t_end_motion - moving[0] if moving else NAN
    if odom["t"]:
        xs, ys, rot = start_frame(info, odom)
        run_row.update({
            "total_distance_m": path_length(odom["x"], odom["y"]), "net_displacement_m": math.hypot(xs[-1], ys[-1]),
            "final_x_s_m": xs[-1], "final_y_s_m": ys[-1], "final_heading_change_deg": math.degrees(rot[-1]),
            "total_rotation_deg": math.degrees(sum(abs(b - a) for a, b in zip(odom["yaw"], odom["yaw"][1:]))),
            "max_abs_v_linear_mps": max(abs(x) for x in odom["v"]),
            "max_abs_angular_radps": max(abs(x) for x in odom["w"])})
        ideal = (info.get("ideal_phase_ends_start_frame") or [None])[-1]
        if completed and ideal:
            run_row.update({"ideal_final_x_s_m": ideal["x"], "ideal_final_y_s_m": ideal["y"],
                            "final_position_error_m": math.hypot(xs[-1] - ideal["x"], ys[-1] - ideal["y"]),
                            "final_heading_error_deg": math.degrees(rot[-1] - ideal["theta_rad"])})
    for name, _ in RUN_METRICS:
        run_row.setdefault(name, NAN)
    run_row["n_transitions_expected"] = len(expected) - 1
    run_row["n_transitions_executed"] = sum(r["executed"] for r in rows)
    run_row["n_tracking_anomalies"] = sum(r["flags"].count("tracking_anomaly") for r in rows)
    run_row["notes"] = "" if completed else f"{outcome}; reached {len(segs)} command segment(s)"
    return rows, run_row, {"info": info, "odom": odom, "segs": segs}


def aggregate(tid, comp, v_new, w_new, name, unit, vals, n_total):
    x = [v for _, v in vals]
    return {"transition_id": tid, "component": comp, "cmd_linear_mps": v_new, "cmd_angular_radps": w_new,
            "metric": name, "unit": unit,
            "mean": statistics.fmean(x) if x else NAN, "std": statistics.stdev(x) if len(x) >= 2 else NAN,
            "min": min(x) if x else NAN, "max": max(x) if x else NAN,
            "n_valid_runs": len(x), "n_runs": n_total, "runs": ";".join(r for r, _ in vals)}


def summarize(trans_rows, run_rows):
    """Per transition (by index), component and metric, over runs; NaN values are excluded and not counted."""
    out = []
    for tid in dict.fromkeys(r["transition_id"] for r in trans_rows):
        rs = [r for r in trans_rows if r["transition_id"] == tid]
        r0 = rs[0]
        for col, unit in metric_columns():
            comp = {"lin": "linear", "ang": "angular"}.get(col[:3], "pair")
            new, prev = (r0["v_new_cmd_mps"], r0["v_prev_cmd_mps"]) if comp == "linear" else \
                (r0["w_new_cmd_radps"], r0["w_prev_cmd_radps"])
            if "stopping" in col and (comp == "pair" or new != 0.0 or prev == 0.0):
                continue                                # only defined for stops of that component
            if col.endswith("tracking_ratio") and new == 0.0:
                continue
            name = col if comp == "pair" else col[4:]
            vals = [(r["run"], r[col]) for r in rs if not math.isnan(r[col])]
            out.append(aggregate(tid, comp, r0["v_new_cmd_mps"], r0["w_new_cmd_radps"], name, unit, vals, len(rs)))
    for name, unit in RUN_METRICS:
        vals = [(r["run"], r[name]) for r in run_rows if r["completed"] and not math.isnan(r.get(name, NAN))]
        out.append(aggregate("RUN(completed runs)", "run", NAN, NAN, name, unit, vals, len(run_rows)))
    return out


# ---- Plots (matplotlib, optional) -------------------------------------------------------------------------------
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CMD_COLOR = "#52514e"         # commanded signal: neutral, dashed
GRID_COLOR = "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a998f")
    ax.tick_params(colors="#52514e", labelsize=8)


def _cmd_step(segs, key, t_end):
    tt, yy = [], []
    for s in segs:
        e = min(s["t_end"], t_end)
        tt += [s["t_start"], e]
        yy += [s[key], s[key]]
    return tt, yy


def make_plots(runs, trans_rows, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: plots skipped (CSV files are complete).")
        return
    out.mkdir(parents=True, exist_ok=True)
    color = {name: RUN_COLORS[i % len(RUN_COLORS)] for i, name in enumerate(runs)}

    # 1. Per run: commanded vs measured v and w over the whole run (two panels, never a dual axis).
    for name, d in runs.items():
        o, segs = d["odom"], d["segs"]
        t_end = o["t"][-1]
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
        for ax, key, sig, label in ((axes[0], "v", "v", "linear velocity (m/s)"),
                                    (axes[1], "w", "w", "angular velocity (rad/s)")):
            tt, yy = _cmd_step(segs, key, t_end)
            ax.plot(tt, yy, color=CMD_COLOR, linestyle="--", linewidth=1.5, label="commanded")
            ax.plot(o["t"], o[sig], color=color[name], linewidth=1.5, label="measured (odometry)")
            for s in segs[1:]:
                ax.axvline(s["t_start"], color=GRID_COLOR, linewidth=1, zorder=0)
            ax.set_ylabel(label, fontsize=9, color="#0b0b0b")
            _style(ax)
        axes[0].legend(loc="upper right", fontsize=8, frameon=False)
        axes[1].set_xlabel("time since recording start (s)", fontsize=9)
        fig.suptitle(f"{name}: commanded vs measured velocity", fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"{name}_velocities.png", dpi=130)
        plt.close(fig)

    # 2. Response around every transition, all runs aligned to the command time.
    tids = list(dict.fromkeys(r["transition_id"] for r in trans_rows))
    fig, axes = plt.subplots(len(tids), 2, figsize=(11, 2.3 * len(tids)), squeeze=False)
    for row_i, tid in enumerate(tids):
        for col_i, (sig, key_prev, key_new, unit) in enumerate(
                (("v", "v_prev_cmd_mps", "v_new_cmd_mps", "m/s"), ("w", "w_prev_cmd_radps", "w_new_cmd_radps", "rad/s"))):
            ax = axes[row_i][col_i]
            r0 = next(r for r in trans_rows if r["transition_id"] == tid)
            ax.plot([-0.5, 0, 0, 2.0], [r0[key_prev], r0[key_prev], r0[key_new], r0[key_new]], color=CMD_COLOR,
                    linestyle="--", linewidth=1.5, label="commanded")
            for name, d in runs.items():
                r = next((x for x in trans_rows if x["run"] == name and x["transition_id"] == tid), None)
                if r is None or not r["executed"]:
                    continue
                o = d["odom"]
                pts = [(t - r["t_cmd_s"], y) for t, y in zip(o["t"], o[sig]) if -0.5 <= t - r["t_cmd_s"] <= 2.0]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color[name], linewidth=1.5, label=name)
            ax.set_title(f"{tid}  {'linear' if sig == 'v' else 'angular'} ({unit})", fontsize=8.5, loc="left")
            _style(ax)
            if row_i == len(tids) - 1:
                ax.set_xlabel("time since command (s)", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / len(tids)))
    fig.savefig(out / "transition_responses.png", dpi=110)
    plt.close(fig)

    # 3. Trajectories in the start frame vs the ideal unicycle path of the commands.
    fig, ax = plt.subplots(figsize=(7, 6.5))
    first = next(iter(runs.values()))
    t_end = first["odom"]["t"][-1]
    ix, iy = ideal_path(first["segs"], t_end)
    ax.plot(ix, iy, color=CMD_COLOR, linestyle="--", linewidth=1.5, label="ideal unicycle (commands)")
    for name, d in runs.items():
        xs, ys, _ = start_frame(d["info"], d["odom"])
        ax.plot(xs, ys, color=color[name], linewidth=1.5, label=name)
        ax.plot(xs[-1], ys[-1], marker="o", markersize=8, color=color[name], markeredgecolor="white")
    ax.plot(0, 0, marker="s", markersize=8, color="#0b0b0b", label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("x_s forward of start pose (m)", fontsize=9)
    ax.set_ylabel("y_s left of start pose (m)", fontsize=9)
    ax.set_title("Odometry trajectory vs ideal unicycle path", fontsize=11, loc="left")
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=130)
    plt.close(fig)
    print(f"Wrote plots to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", default=str(HERE / "runs"))
    parser.add_argument("--out-dir", default=str(HERE / "analysis" / "combined"))
    parser.add_argument("--no-plots", action="store_true", help="CSV files only")
    args = parser.parse_args()

    run_dirs = sorted((d for d in Path(args.runs_root).iterdir()
                       if d.is_dir() and re.fullmatch(r"combined_run_\d+", d.name)),
                      key=lambda d: int(d.name.rsplit("_", 1)[1]))
    trans_rows, run_rows, raw = [], [], {}
    for d in run_dirs:
        if not all((d / f).exists() for f in ("commands.csv", "odometry.csv", "run_info.json")):
            run_rows.append({"run": d.name, "outcome": "MISSING_FILES", "completed": False,
                             "notes": "commands.csv, odometry.csv or run_info.json missing"})
            continue
        rows, run_row, data = analyze_run(d)
        trans_rows += rows
        run_rows.append(run_row)
        raw[d.name] = data
        print(f"{d.name}: {run_row['mode']}, {run_row['outcome']}, {run_row['n_transitions_executed']}/"
              f"{run_row['n_transitions_expected']} transitions executed")
    modes = {r.get("mode") for r in run_rows if r.get("mode")}
    if len(modes) > 1:
        print(f"WARNING: runs use different command sequences {sorted(modes)}; transitions are matched by index")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trans_fields = ["run", "transition_id", "transition_index", "scheduled_phase", "phase", "executed",
                    "v_prev_cmd_mps", "v_new_cmd_mps", "delta_v_cmd_mps", "w_prev_cmd_radps", "w_new_cmd_radps",
                    "delta_w_cmd_radps", "t_cmd_s", "t_phase_end_s", "phase_duration_s", "n_samples",
                    "lin_baseline", "lin_peak", "ang_baseline", "ang_peak"] + \
                   [n for n, _ in metric_columns()] + ["flags", "notes"]
    run_fields = ["run", "mode", "outcome", "completed"] + [n for n, _ in RUN_METRICS] + \
                 ["n_transitions_expected", "n_transitions_executed", "n_tracking_anomalies", "n_cmd", "n_odom",
                  "clock_offset_s", "notes"]
    sum_fields = ["transition_id", "component", "cmd_linear_mps", "cmd_angular_radps", "metric", "unit", "mean",
                  "std", "min", "max", "n_valid_runs", "n_runs", "runs"]
    write_csv(out / "per_transition_metrics.csv", trans_fields, trans_rows)
    write_csv(out / "per_run_metrics.csv", run_fields, run_rows)
    write_csv(out / "summary.csv", sum_fields, summarize(trans_rows, run_rows))
    print(f"Wrote {out}/per_transition_metrics.csv, per_run_metrics.csv, summary.csv")
    if not args.no_plots:
        make_plots(raw, trans_rows, out / "plots")


if __name__ == "__main__":
    main()
