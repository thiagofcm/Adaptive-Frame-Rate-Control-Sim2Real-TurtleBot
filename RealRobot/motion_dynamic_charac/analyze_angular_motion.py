#!/usr/bin/env python3
"""Quantitative analysis of Experiment 2 (angular velocity step response, angular_step_response.py).

Companion of analyze_linear_motion.py: same timeline, transition windows, baseline / onset detection, NaN policy
and CSV layout, with angular thresholds. Discovers runs/angular_run_<N>/, computes metrics for every commanded
angular-velocity transition of every run independently, and aggregates them across runs per transition.
Standard library only. No plots, no model fitting.

Inputs per run (written by angular_step_response.py):
    commands.csv  t_s, ros_time_s, cmd_linear, cmd_angular, phase, yaw, rotation_rad, odom_age_s
    odometry.csv  t_s, ros_time_s, stamp_s, x, y, yaw, yaw_unwrapped, rotation_rad, v_linear, v_angular
    run_info.json sequence, outcome, start_pose, ...
Positive angular velocity is counter-clockwise (ROS convention). Linear velocity is always commanded 0.

Timeline (as in the linear analyzer)
    Commands at commands.csv t_s (monotonic, recording origin). Each odometry sample is placed at
        t = stamp_s - max(stamp_s - t_s)          (one constant per run)
    i.e. the robot's sampling clock mapped onto the shared monotonic origin through the fastest-delivered message.
    Arrival jitter is removed, the minimum odometry latency stays in every delay. Offset drift < 3 ms per run.
    Native 20 Hz samples, no filtering or resampling; time metrics are quantized to ~0.05 s.

Heading
    The heading is unwrapped here from the raw odometry yaw (each step wrapped to (-pi, pi] and accumulated), so it
    does not depend on the recorder's yaw_unwrapped column (identical in the recorded data). Steps are < 0.06 rad
    at 20 Hz and 0.6 rad/s, far from the +/-pi ambiguity.

Transitions
    A transition is a change of cmd_angular between consecutive command segments; its window runs from the command
    to the next change (FINAL_STOP + EXIT_STOP zeros of a completed run form one segment, observed to the end of the
    recording). In aborted runs the window of the phase that was running ends at the first EXIT_STOP command (flag
    phase_truncated_by_abort); the exit stop itself is not a transition. Expected transitions come from run_info's
    sequence; transitions never reached get a row of NaN with the reason.
    Classes (by the commanded values): startup (0 -> w), stop (w -> 0), reversal (sign change), acceleration
    (same sign, |w| grows), deceleration (same sign, |w| shrinks). 0 -> -0.2 is a clockwise startup, not a
    deceleration.

Metric definitions (dw = w_new - w_prev, commanded; dev = (w - ref) * sign(dw), so signs and reversals are handled
by measuring along the direction of the commanded change)
    response_delay_s   first sample after the command at which dev relative to the measured baseline is
                       >= max(10 % |dw|, NOISE_RADPS) on DETECT_SAMPLES consecutive samples. The baseline is the
                       MEDIAN of w over BASELINE_S before the command (the linear analyzer uses the mean): the
                       2-3 s angular phases often still contain the previous transient (e.g. a 2 s startup dead
                       time) in that window, and the median ignores it. Flag unsteady_baseline when the baseline
                       samples have sd > BASELINE_SD_RADPS.
    rise_time_s        commanded 10 %-90 % levels (w_prev + 0.1 dw, w_prev + 0.9 dw); crossings linearly
                       interpolated between the two bracketing native samples (as in the linear analyzer).
    settling_time_s    command -> first sample after which |w - w_new| <= SETTLE_TOL_RADPS for SETTLE_HOLD_S
                       (the hold must be observed inside the window).
    steady_state_error_radps  e = w_new - mean(w) over the last STEADY_S of the window. Requires the steady window
                       to start >= TRANSIENT_S after the response onset and >= MIN_STEADY_SAMPLES samples.
                       The meaning of the sign depends on the direction: for a CCW (positive) command e < 0 is
                       overtracking (turning faster than commanded); for a CW (negative) command e > 0 is
                       overtracking. tracking_ratio = mean(w) / w_new (> 1 overtracking, < 1 undertracking) is
                       reported for non-zero commands as the direction-independent measure.
    overshoot_pct      100 * max(0, max dev past w_new) / |dw| over [command, onset + TRANSIENT_S). Deceleration:
                       dip below the target; stop: rotation in the opposite direction; reversal: past the new
                       (opposite) command.
    stopping_time_s    (w_new == 0) command -> |w| <= STOP_RADPS held for STOP_HOLD_S.
    stopping_angle_rad (w_new == 0) unwrapped heading at the stopping sample - heading at the command (linearly
                       interpolated between the bracketing samples); signed (+ = counter-clockwise), also |.| and deg.
    Translation (odometry-reported, not ground truth): path_length_m and max/mean |v_linear| over the window, and
    rotation_center_offset_m = path length / accumulated |rotation| in the window (m per rad = distance from the
    odometry frame to the effective centre of rotation; ~0 when rotating in place, ~half the wheel separation when
    pivoting about one wheel), and steady_rotation_center_offset_m, the same over the steady window (last STEADY_S).
    Each needs >= MIN_ROTATION_RAD of rotation. Flag translating_while_rotating when either exceeds PIVOT_OFFSET_M
    (the steady one catches a regime change part-way through a phase).

Run-level metrics: recording and commanded-motion duration, accumulated |rotation| and net rotation (rad, deg),
max |w|, odometry path length, net displacement and max distance from the start pose, max |v_linear|, and the
time and phase at which sustained linear velocity first appears (|v_linear| >= LIN_DRIFT_MPS held for
LIN_DRIFT_HOLD_S).

Output: analysis/angular/per_transition_metrics.csv, per_run_metrics.csv, summary.csv
    summary.csv gives the commanded angular velocity of each transition as cmd_radps (the previous command is
    in the transition_id) and has a subset column: "all" (every valid run) and "in_place" (excluding transitions flagged
    translating_while_rotating), because the two regimes differ strongly.

Usage:
    python3 RealRobot/motion_dynamic_charac/analyze_angular_motion.py [--runs-root DIR] [--out-dir DIR]
"""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from analyze_linear_motion import first_hold, path_length, read_csv, write_csv

HERE = Path(__file__).resolve().parent

BASELINE_S = 1.0            # pre-command window for the baseline median
BASELINE_SD_RADPS = 0.05    # baseline samples with a larger sd are flagged unsteady_baseline
NOISE_RADPS = 0.05          # minimum deviation counted as a response (sd while turning ~0.015-0.04 rad/s)
DETECT_SAMPLES = 2          # consecutive samples beyond the threshold
SETTLE_TOL_RADPS = 0.08     # settling band around the command (~3 sd of the noise while turning in place)
SETTLE_HOLD_S = 0.5         # time the angular velocity must stay within the band
TRANSIENT_S = 0.8           # after onset: overshoot search window / excluded from the steady window
STEADY_S = 1.0              # steady-state averaging window at the end of the phase (phases last 2-3 s)
MIN_STEADY_SAMPLES = 10
STOP_RADPS = 0.02           # near-zero threshold for stopping (at rest noise ~0.002 rad/s)
STOP_HOLD_S = 0.5
MAX_GAP_S = 0.1             # odometry stamp gaps longer than this before the onset are flagged
MIN_ROTATION_RAD = 0.1      # rotation-centre offsets only when the window rotates at least this much
PIVOT_OFFSET_M = 0.05       # rotation-centre offset above which a window is flagged translating_while_rotating
LIN_DRIFT_MPS = 0.005       # run level: sustained odometry linear speed ...
LIN_DRIFT_HOLD_S = 0.5      # ... held this long marks the onset of translation

# Per-transition metrics, (column, unit). Full definitions in the module docstring.
TRANSITION_METRICS = [
    # Dead time: command -> first sample where w has clearly left its pre-command level in the commanded direction.
    ("response_delay_s", "s"),
    # Speed of the change itself: time to go from 10 % to 90 % of the commanded step.
    ("rise_time_s", "s"),
    # Command -> w enters +/-SETTLE_TOL_RADPS of the new command and stays there for SETTLE_HOLD_S.
    ("settling_time_s", "s"),
    # Tracking accuracy at the end of the phase: commanded - mean measured w (signed, see docstring).
    ("steady_state_error_radps", "rad/s"),
    ("abs_steady_state_error_radps", "rad/s"),
    # mean measured w / commanded w at the end of the phase: > 1 turns faster, < 1 slower (non-zero commands).
    ("tracking_ratio", "-"),
    # Largest excursion past the new command, in the direction of the change, as % of the step size.
    ("overshoot_pct", "%"),
    # Stop transitions only: command -> |w| <= STOP_RADPS held for STOP_HOLD_S.
    ("stopping_time_s", "s"),
    # Stop transitions only: extra heading change from the stop command to the stopping sample (+ = CCW).
    ("stopping_angle_rad", "rad"),
    ("abs_stopping_angle_rad", "rad"),
    ("stopping_angle_deg", "deg"),
    # Odometry translation during the window (the robot is commanded to rotate in place).
    ("path_length_m", "m"),
    ("max_abs_v_linear_mps", "m/s"),
    ("mean_abs_v_linear_mps", "m/s"),
    # Path length per radian rotated: distance to the effective centre of rotation.
    ("rotation_center_offset_m", "m"),
    ("steady_rotation_center_offset_m", "m"),
]
# Per-run metrics, (column, unit).
RUN_METRICS = [
    ("duration_s", "s"),                    # length of the recording (commands and odometry)
    ("motion_duration_s", "s"),             # first non-zero command -> start of the last zero command (or abort)
    ("total_rotation_rad", "rad"),          # accumulated |heading change|
    ("total_rotation_deg", "deg"),
    ("net_rotation_rad", "rad"),            # final - initial unwrapped heading (+ = CCW)
    ("net_rotation_deg", "deg"),
    ("max_abs_angular_radps", "rad/s"),     # largest |w| sample
    ("total_distance_m", "m"),              # odometry path length over the whole recording
    ("net_displacement_m", "m"),            # start pose -> final pose
    ("max_displacement_m", "m"),            # largest distance from the start pose
    ("max_abs_v_linear_mps", "m/s"),        # largest |v_linear| sample
]
NAN = float("nan")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def unwrap(yaw):
    out = [yaw[0]] if yaw else []
    for a, b in zip(yaw, yaw[1:]):
        out.append(out[-1] + wrap(b - a))
    return out


def load_run(run_dir):
    info = json.loads((run_dir / "run_info.json").read_text())
    cmds = [{"t": float(r["t_s"]), "w": float(r["cmd_angular"]), "v": float(r["cmd_linear"]), "phase": r["phase"]}
            for r in read_csv(run_dir / "commands.csv")]
    rows = read_csv(run_dir / "odometry.csv")
    arrival = [float(r["t_s"]) for r in rows]
    stamp = [float(r["stamp_s"]) for r in rows]
    offset = max(s - a for s, a in zip(stamp, arrival)) if rows else NAN
    odom = {"t": [s - offset for s in stamp], "t_arrival": arrival,
            "w": [float(r["v_angular"]) for r in rows], "v": [float(r["v_linear"]) for r in rows],
            "x": [float(r["x"]) for r in rows], "y": [float(r["y"]) for r in rows],
            "yaw": unwrap([float(r["yaw"]) for r in rows])}
    return info, cmds, odom, offset


def segments(cmds, completed, t_record_end):
    """Constant-command segments: [{w, t_start, t_end, phases, truncated}] ordered in time."""
    segs = []
    for c in cmds:
        # Abort/interrupt: the experiment ends at the first EXIT_STOP. In a completed run the EXIT_STOP zeros simply
        # extend the FINAL_STOP segment.
        if c["phase"] == "EXIT_STOP" and not completed:
            if segs:
                segs[-1]["t_end"] = c["t"]
                segs[-1]["truncated"] = True
            break
        if not segs or segs[-1]["w"] != c["w"]:
            if segs:
                segs[-1]["t_end"] = c["t"]
            segs.append({"w": c["w"], "t_start": c["t"], "t_end": t_record_end, "phases": [], "truncated": False})
        if c["phase"] not in segs[-1]["phases"]:
            segs[-1]["phases"].append(c["phase"])
    return segs


def classify(w_prev, w_new):
    if w_prev == 0.0:
        return "startup"
    if w_new == 0.0:
        return "stop"
    if w_prev * w_new < 0:
        return "reversal"
    return "acceleration" if abs(w_new) > abs(w_prev) else "deceleration"


def interp_at(t, vals, i0, t_q):
    """Value of vals at t_q, linear between samples i0 - 1 and i0 (t[i0 - 1] < t_q <= t[i0])."""
    if i0 == 0:
        return vals[0]
    a = (t_q - t[i0 - 1]) / (t[i0] - t[i0 - 1])
    return vals[i0 - 1] + a * (vals[i0] - vals[i0 - 1])


def transition_metrics(odom, t_cmd, t_end, w_prev, w_new):
    """Metrics of one transition from native samples; odom = dict of equal-length lists t, w, v, x, y, yaw
    (yaw unwrapped). Returns (metrics dict, flags list, notes list)."""
    t, w = odom["t"], odom["w"]
    m = {name: NAN for name, _ in TRANSITION_METRICS}
    m.update({"baseline_radps": NAN, "peak_radps": NAN, "delay_resolution_s": NAN, "n_samples": 0})
    flags, notes = [], []
    dw = w_new - w_prev
    sgn = 1.0 if dw > 0 else -1.0

    i0 = next((i for i, ti in enumerate(t) if ti >= t_cmd), len(t))
    i1 = next((i for i, ti in enumerate(t) if ti >= t_end), len(t))
    m["n_samples"] = i1 - i0
    if i1 - i0 < DETECT_SAMPLES:
        notes.append("all: fewer than 2 odometry samples in the transition window")
        return m, flags + ["no_data"], notes
    t_last = t[i1 - 1] if t_end == math.inf else t_end
    if t_last - t_cmd < TRANSIENT_S:
        flags.append("short_window")

    # Translation during the window (odometry-reported).
    idx = range(i0, i1)
    m["path_length_m"] = path_length(odom["x"][i0:i1], odom["y"][i0:i1])
    m["max_abs_v_linear_mps"] = max(abs(odom["v"][i]) for i in idx)
    m["mean_abs_v_linear_mps"] = statistics.fmean(abs(odom["v"][i]) for i in idx)

    def center_offset(a, b):
        rot = sum(abs(odom["yaw"][i] - odom["yaw"][i - 1]) for i in range(a + 1, b))
        return path_length(odom["x"][a:b], odom["y"][a:b]) / rot if rot >= MIN_ROTATION_RAD else NAN

    m["rotation_center_offset_m"] = center_offset(i0, i1)
    i_ss = next(i for i in idx if t[i] >= t_last - STEADY_S)
    m["steady_rotation_center_offset_m"] = center_offset(i_ss, i1)
    if m["rotation_center_offset_m"] > PIVOT_OFFSET_M or m["steady_rotation_center_offset_m"] > PIVOT_OFFSET_M:
        flags.append("translating_while_rotating")
    if math.isnan(m["rotation_center_offset_m"]):
        notes.append(f"rotation_center_offset: less than {MIN_ROTATION_RAD} rad rotated in the window")

    # Baseline: measured median over the BASELINE_S before the command.
    base = [w[i] for i in range(i0) if t[i] >= t_cmd - BASELINE_S]
    if len(base) >= 3:
        m["baseline_radps"] = statistics.median(base)
        if statistics.stdev(base) > BASELINE_SD_RADPS:
            flags.append("unsteady_baseline")
    else:
        m["baseline_radps"] = w_prev
        flags.append("baseline_from_command")

    # Response delay.
    thr = max(0.1 * abs(dw), NOISE_RADPS)
    above = [(w[i] - m["baseline_radps"]) * sgn >= thr for i in range(len(t))]
    onset = next((i for i in range(i0, i1 - DETECT_SAMPLES + 1) if all(above[i:i + DETECT_SAMPLES])), None)
    if onset is None:
        notes.append("response_delay/rise/overshoot: no response detected before the next command")
        flags.append("no_response")
    else:
        m["response_delay_s"] = t[onset] - t_cmd
        m["delay_resolution_s"] = t[onset] - t[onset - 1] if onset > 0 else NAN
        if any(t[i] - t[i - 1] > MAX_GAP_S for i in range(max(i0, 1), onset + 1)):
            flags.append("odom_gap_before_onset")

    # Rise time (commanded 10 %-90 % levels), crossings interpolated between bracketing native samples.
    dev_cmd = [(w[i] - w_prev) * sgn for i in range(len(t))]

    def crossing(i, level):
        if i == 0 or dev_cmd[i - 1] >= level:
            return t[i]
        return t[i - 1] + (level - dev_cmd[i - 1]) / (dev_cmd[i] - dev_cmd[i - 1]) * (t[i] - t[i - 1])

    i90 = next((i for i in idx if dev_cmd[i] >= 0.9 * abs(dw)), None)
    if i90 is None:
        notes.append("rise_time: 90 % level not reached before the next command")
    else:
        i10 = i90
        while i10 - 1 >= i0 and dev_cmd[i10 - 1] >= 0.1 * abs(dw):
            i10 -= 1
        if i10 == i0 and dev_cmd[i0] >= 0.1 * abs(dw):
            flags.append("rise_started_before_first_sample")
        m["rise_time_s"] = crossing(i90, 0.9 * abs(dw)) - max(crossing(i10, 0.1 * abs(dw)), t_cmd)
        if i90 - i10 <= 1:
            flags.append("rise_spans_le_2_samples")

    # Overshoot over [command, onset + TRANSIENT_S).
    if onset is not None:
        t_os_end = min(t_end, t[onset] + TRANSIENT_S)
        peak = max((i for i in idx if t[i] < t_os_end), key=lambda i: (w[i] - w_new) * sgn)
        m["peak_radps"] = w[peak]
        m["overshoot_pct"] = 100.0 * max(0.0, (w[peak] - w_new) * sgn) / abs(dw)
        if t_os_end < t[onset] + TRANSIENT_S:
            flags.append("overshoot_window_truncated")

    # Settling time.
    k = first_hold(t, [abs(wi - w_new) <= SETTLE_TOL_RADPS for wi in w], i0, i1, SETTLE_HOLD_S)
    if k is None:
        notes.append(f"settling_time: not within +/-{SETTLE_TOL_RADPS} rad/s for {SETTLE_HOLD_S} s before the next"
                     f" command")
    else:
        m["settling_time_s"] = t[k] - t_cmd

    # Steady-state error over the last STEADY_S of the window.
    t_ss_start = t_last - STEADY_S
    ss = [w[i] for i in idx if t[i] >= t_ss_start]
    if onset is None:
        notes.append("steady_state_error: no response detected")
    elif t_ss_start < t[onset] + TRANSIENT_S:
        notes.append(f"steady_state_error: window too short (steady window would start "
                     f"{t[onset] + TRANSIENT_S - t_ss_start:.2f} s inside the transient)")
    elif len(ss) < MIN_STEADY_SAMPLES:
        notes.append(f"steady_state_error: only {len(ss)} samples in the steady window")
    else:
        mean_w = statistics.fmean(ss)
        m["steady_state_error_radps"] = w_new - mean_w
        m["abs_steady_state_error_radps"] = abs(w_new - mean_w)
        if w_new != 0.0:
            m["tracking_ratio"] = mean_w / w_new

    # Stopping time and angle.
    if w_new == 0.0:
        k = first_hold(t, [abs(wi) <= STOP_RADPS for wi in w], i0, i1, STOP_HOLD_S)
        if k is None:
            notes.append(f"stopping: |w| did not stay <= {STOP_RADPS} rad/s for {STOP_HOLD_S} s within the window")
        else:
            m["stopping_time_s"] = t[k] - t_cmd
            if i0 == 0:
                flags.append("no_sample_before_stop_command")
            ang = odom["yaw"][k] - interp_at(t, odom["yaw"], i0, t_cmd)
            m["stopping_angle_rad"], m["abs_stopping_angle_rad"] = ang, abs(ang)
            m["stopping_angle_deg"] = math.degrees(ang)
    return m, flags, notes


def transition_id(index, w_prev, w_new):
    return f"T{index}_{w_prev:+.2f}->{w_new:+.2f}"


def analyze_run(run_dir):
    info, cmds, odom, offset = load_run(run_dir)
    run = run_dir.name
    outcome = info.get("outcome", "UNKNOWN")
    completed = outcome == "COMPLETED"
    t_record_end = max([c["t"] for c in cmds] + odom["t_arrival"]) if cmds or odom["t"] else NAN

    sequence = info.get("sequence", [])
    expected = [(p[0], float(p[2])) for p in sequence]
    # Consecutive phases with the same command would not be a transition.
    expected = [p for i, p in enumerate(expected) if i == 0 or p[1] != expected[i - 1][1]]
    segs = segments(cmds, completed, math.inf if completed else t_record_end)

    rows = []
    for n in range(1, len(expected)):
        (_, w_prev), (phase_name, w_new) = expected[n - 1], expected[n]
        row = {"run": run, "transition_id": transition_id(n, w_prev, w_new), "transition_index": n,
               "scheduled_phase": phase_name, "class": classify(w_prev, w_new),
               "w_prev_cmd_radps": w_prev, "w_new_cmd_radps": w_new, "delta_w_cmd_radps": w_new - w_prev}
        flags, notes = [], []
        if n >= len(segs) or segs[n]["w"] != w_new or segs[n - 1]["w"] != w_prev:
            row.update({name: NAN for name, _ in TRANSITION_METRICS})
            row.update({"executed": False, "phase": "", "t_cmd_s": NAN, "t_phase_end_s": NAN, "phase_duration_s": NAN})
            if n >= len(segs):
                notes.append(f"not executed: run {outcome}")
            else:
                notes.append("recorded command sequence differs from the schedule at this transition")
                flags.append("unexpected_sequence")
        else:
            seg = segs[n]
            m, flags, notes = transition_metrics(odom, seg["t_start"], seg["t_end"], w_prev, w_new)
            row.update(m)
            t_last = seg["t_end"] if seg["t_end"] != math.inf else odom["t"][-1]
            row.update({"executed": True, "phase": "+".join(seg["phases"]), "t_cmd_s": seg["t_start"],
                        "t_phase_end_s": t_last, "phase_duration_s": t_last - seg["t_start"]})
            if seg["truncated"]:
                flags.append("phase_truncated_by_abort")
        row["flags"] = ";".join(flags)
        row["notes"] = " | ".join(notes)
        rows.append(row)

    # Run-level metrics.
    run_row = {"run": run, "outcome": outcome, "completed": completed, "n_cmd": len(cmds), "n_odom": len(odom["t"]),
               "clock_offset_s": offset, "duration_s": t_record_end,
               "max_abs_cmd_linear_mps": max((abs(c["v"]) for c in cmds), default=NAN)}
    moving = [c["t"] for c in cmds if c["w"] != 0.0]
    t_end_motion = segs[-1]["t_start"] if segs and segs[-1]["w"] == 0.0 else (segs[-1]["t_end"] if segs else NAN)
    run_row["motion_duration_s"] = t_end_motion - moving[0] if moving else NAN
    if odom["t"]:
        start = info.get("start_pose") or {"x": odom["x"][0], "y": odom["y"][0]}
        x0, y0 = start["x"], start["y"]
        yaw = odom["yaw"]
        run_row["total_rotation_rad"] = sum(abs(b - a) for a, b in zip(yaw, yaw[1:]))
        run_row["net_rotation_rad"] = yaw[-1] - yaw[0]
        run_row["total_rotation_deg"] = math.degrees(run_row["total_rotation_rad"])
        run_row["net_rotation_deg"] = math.degrees(run_row["net_rotation_rad"])
        run_row["max_abs_angular_radps"] = max(abs(x) for x in odom["w"])
        run_row["total_distance_m"] = path_length(odom["x"], odom["y"])
        run_row["net_displacement_m"] = math.hypot(odom["x"][-1] - x0, odom["y"][-1] - y0)
        run_row["max_displacement_m"] = max(math.hypot(x - x0, y - y0) for x, y in zip(odom["x"], odom["y"]))
        run_row["max_abs_v_linear_mps"] = max(abs(x) for x in odom["v"])
        k = first_hold(odom["t"], [abs(x) >= LIN_DRIFT_MPS for x in odom["v"]], 0, len(odom["t"]),
                       LIN_DRIFT_HOLD_S)
        run_row["t_linear_onset_s"] = odom["t"][k] if k is not None else NAN
        run_row["phase_at_linear_onset"] = next(
            ("+".join(s["phases"]) for s in segs if s["t_start"] <= odom["t"][k] < s["t_end"]), "") \
            if k is not None else ""
    run_row["n_transitions_expected"] = len(expected) - 1
    run_row["n_transitions_executed"] = sum(r["executed"] for r in rows)
    run_row["n_valid_transitions"] = sum(r["executed"] and not math.isnan(r["response_delay_s"]) for r in rows)
    run_row["n_translating_transitions"] = sum("translating_while_rotating" in r["flags"] for r in rows)
    run_row["notes"] = "" if completed else f"{outcome}; reached {len(segs)} command segment(s)"
    return rows, run_row


def aggregate(tid, cls, cmd, subset, name, unit, vals, n_total):
    x = [v for _, v in vals]
    return {"transition_id": tid, "class": cls, "cmd_radps": cmd, "subset": subset, "metric": name, "unit": unit,
            "mean": statistics.fmean(x) if x else NAN,
            "std": statistics.stdev(x) if len(x) >= 2 else NAN,
            "min": min(x) if x else NAN, "max": max(x) if x else NAN,
            "n_valid_runs": len(x), "n_runs": n_total, "runs": ";".join(r for r, _ in vals)}


def summarize(trans_rows, run_rows):
    """Per transition (by index, so repeated command pairs stay separate) and metric, over runs."""
    out = []
    for tid in dict.fromkeys(r["transition_id"] for r in trans_rows):
        rs = [r for r in trans_rows if r["transition_id"] == tid]
        r0 = rs[0]
        for subset in ("all", "in_place"):
            for name, unit in TRANSITION_METRICS:
                if name.endswith(("stopping_angle_rad", "stopping_angle_deg", "stopping_time_s")) \
                        and r0["w_new_cmd_radps"] != 0.0:
                    continue                            # only defined for stop transitions
                if name == "tracking_ratio" and r0["w_new_cmd_radps"] == 0.0:
                    continue
                vals = [(r["run"], r[name]) for r in rs if not math.isnan(r[name])
                        and (subset == "all" or "translating_while_rotating" not in r["flags"])]
                out.append(aggregate(tid, r0["class"], r0["w_new_cmd_radps"], subset, name, unit, vals, len(rs)))
    for name, unit in RUN_METRICS:
        vals = [(r["run"], r[name]) for r in run_rows if r["completed"] and not math.isnan(r.get(name, NAN))]
        out.append(aggregate("RUN(completed runs)", "", NAN, "all", name, unit, vals, len(run_rows)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", default=str(HERE / "runs"))
    parser.add_argument("--out-dir", default=str(HERE / "analysis" / "angular"))
    args = parser.parse_args()

    run_dirs = sorted((d for d in Path(args.runs_root).iterdir()
                       if d.is_dir() and re.fullmatch(r"angular_run_\d+", d.name)),
                      key=lambda d: int(d.name.rsplit("_", 1)[1]))
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
              f"{run_row['n_transitions_expected']} transitions executed, "
              f"{run_row['n_translating_transitions']} translating while rotating")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trans_fields = ["run", "transition_id", "transition_index", "class", "scheduled_phase", "phase", "executed",
                    "w_prev_cmd_radps", "w_new_cmd_radps", "delta_w_cmd_radps", "t_cmd_s", "t_phase_end_s",
                    "phase_duration_s", "n_samples", "baseline_radps", "peak_radps"] + \
                   [n for n, _ in TRANSITION_METRICS] + ["delay_resolution_s", "flags", "notes"]
    run_fields = ["run", "outcome", "completed"] + [n for n, _ in RUN_METRICS] + \
                 ["t_linear_onset_s", "phase_at_linear_onset", "max_abs_cmd_linear_mps", "n_transitions_expected",
                  "n_transitions_executed", "n_valid_transitions", "n_translating_transitions", "n_cmd", "n_odom",
                  "clock_offset_s", "notes"]
    sum_fields = ["transition_id", "class", "cmd_radps", "subset",
                  "metric", "unit", "mean", "std", "min", "max", "n_valid_runs", "n_runs", "runs"]
    write_csv(out / "per_transition_metrics.csv", trans_fields, trans_rows)
    write_csv(out / "per_run_metrics.csv", run_fields, run_rows)
    write_csv(out / "summary.csv", sum_fields, summarize(trans_rows, run_rows))
    print(f"Wrote {out}/per_transition_metrics.csv, per_run_metrics.csv, summary.csv")


if __name__ == "__main__":
    main()
