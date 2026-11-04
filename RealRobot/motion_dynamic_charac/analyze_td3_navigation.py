#!/usr/bin/env python3
"""Actuator dynamics during real TD3 navigation (RealRobot/eval_randomization_dr), compared with the isolated
step-response experiments (runs/run_*, angular_run_*, combined_run_*). Standard library only, no plots.

What the navigation logs contain (AdaptiveFPS/scripts/real_eval_fixed.py / real_eval_adaptive.py and
real_episode_recorder.py):
  * TD3 ticks every 0.2 s (metadata td3_tick_times_s, same clock as episode_time_s). At each tick the command
    (v = a0 * 0.22 m/s, w = a1 * 2.0 rad/s, unclipped) is published by real_environment through a service call;
    the new command is already visible in log rows 1-5 ms after the tick, so it is applied from the tick time.
  * trajectory.csv rows every 0.1 s (timer): the last command and the LATEST CACHED odometry pose (x, y, yaw).
    No odometry stamps and no measured velocities are logged: a pose is between 0 and ~50 ms old (20 Hz odometry)
    plus the odometry transport latency when it is logged.

Method (chosen because differentiating 10 Hz poses with +/-25 ms timing uncertainty gives 25-50 % noise per
sample):
  * Measured increments over windows of WINDOW_ROWS log rows (0.2 s): forward displacement ds (sum of pose steps
    projected on the mid-step heading, signed, so reversing counts negative) and heading change dyaw (unwrapped).
  * Predicted increments: the commanded signal passed through a first-order-plus-dead-time model
    K * exp(-theta s) / (T s + 1), integrated exactly over the same window (fine grid FINE_DT).
  * Per episode and component, a grid search over theta (dead time) and T (time constant) with K fitted by least
    squares (no intercept) gives the effective dynamics; T = 0 gives the best pure delay ("effective lag").
  * Gains per command level: ratio of summed measured to summed predicted increments in windows whose command was
    quasi-steady (it changed by less than STEADY_TOL over the ticks that drive the window); with continuously
    changing TD3 commands these are rare, so all-window ratios (model-dependent) are reported too.
  * The same estimators are applied to the isolated experiments resampled like the navigation recorder (10 Hz rows,
    latest pose by arrival time), so both data sets go through identical measurement pipelines. The isolated
    analyzers' stamp-aligned step metrics are the reference for what the estimator should return.

  * Excluded everywhere: windows before first motion + STARTUP_MARGIN_S (first-motion dead time, analysed
    separately). Lag/gain fits and gains use only unsaturated windows (no driving command demands more than
    SAT_SAFE_MPS on a wheel, nominal geometry); saturation is then tested by refitting all windows with a wheel
    limit applied to the commands, 'clip' (each wheel limited) or 'scale' (both scaled, curvature kept).
  * Pivot check: 1 s windows where one wheel moves at < 25 % of its (lagged) command while the other moves
    normally, sustained >= PIV_HOLD_S; validated on the isolated angular runs where pivoting is known. Also
    wheel_direction_check: slow (0.04-0.12 m/s) wheels commanded backward vs forward (the isolated pivots stall the
    backward-turning wheel).
  * Sliding-window lag (3 s windows, delays up to 1.5 s) against time since first motion.
  * isolated_angular_clean: angular runs cut 1 s before the pivot onset found by analyze_angular_motion.py.

Outputs: analysis/td3_navigation/{dynamics_fits.csv, gain_by_command.csv, saturation_models.csv,
saturation_bins.csv, lag_windows.csv, lag_vs_time.csv, wheel_direction_check.csv, straight_bias.csv,
first_motion.csv, pivot_check.csv, calibration_isolated.csv}

Usage: python3 RealRobot/motion_dynamic_charac/analyze_td3_navigation.py
"""

import csv
import glob
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAV_ROOT = HERE.parent / "eval_randomization_dr"
OUT = HERE / "analysis" / "td3_navigation"

LOG_DT = 0.1                 # recorder interval
WINDOW_ROWS = 2              # increment window = 2 rows = 0.2 s
FINE_DT = 0.005              # model simulation step
THETAS = [i * 0.01 for i in range(0, 61)]        # dead time grid, s
TAUS = [i * 0.01 for i in range(0, 31)]          # time constant grid, s
STEADY_TOL = {"v": 0.01, "w": 0.1}               # quasi-steady command: change < this over the driving ticks
WHEEL_SEP_M = 0.235          # nominal Create 3 wheel separation (not measured here)
SAT_SAFE_MPS = 0.28          # lag/gain fits use windows whose commands demand at most this per wheel
SAT_GRID = [0.26 + 0.005 * i for i in range(17)]  # wheel-speed limits tested by the saturation models, m/s
STARTUP_MARGIN_S = 0.5       # windows before first motion + this are excluded (first-motion dead time)
PIV_ROWS = 10                # pivot check: 1 s windows ...
PIV_CMD_MPS = 0.04           # ... both wheels commanded at least this ...
PIV_STALL_FRAC = 0.25        # ... one wheel moving at < this fraction of its command ...
PIV_OTHER_FRAC = 0.5         # ... the other at >= this fraction ...
PIV_HOLD_S = 1.0             # ... for at least this long
SAT_FIT_MPS = 0.305          # wheel limit used where a saturated command is needed (best fit of saturation_models)
LONG_THETAS = [i * 0.02 for i in range(0, 76)]   # sliding-lag grid up to 1.5 s
MOTION_POS_M = 0.003         # first motion: displacement or ...
MOTION_YAW_RAD = 0.01        # ... rotation from the start pose beyond these
W_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.01]   # |w| bins, rad/s (0.1-0.3 holds +/-0.2)
V_BINS = [0.0, 0.05, 0.10, 0.15, 0.19, 0.215, 0.221]
NAN = float("nan")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def unwrap(yaw):
    out = [yaw[0]] if yaw else []
    for a, b in zip(yaw, yaw[1:]):
        out.append(out[-1] + wrap(b - a))
    return out


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---- Loading ------------------------------------------------------------------------------------------------------

def load_nav_episode(ep_dir):
    meta = json.loads((ep_dir / "metadata.json").read_text())
    rows = read_csv(ep_dir / "trajectory.csv")
    T = [float(r["episode_time_s"]) for r in rows]
    ticks = list(meta["td3_tick_times_s"])
    cmd_v, cmd_w = [], []
    for k, tk in enumerate(ticks):
        t_next = ticks[k + 1] if k + 1 < len(ticks) else tk + 0.2
        # the row ~0.1 s after the tick shows this tick's command (rows right at the tick may still show the old one)
        r = next((r for r, t in zip(rows, T) if tk + 0.05 <= t < t_next), None)
        r = r or next((r for r, t in zip(rows, T) if t > tk), rows[-1])
        cmd_v.append(float(r["cmd_linear"]))
        cmd_w.append(float(r["cmd_angular"]))
    return {"name": f"{ep_dir.parent.name}/{ep_dir.name}", "policy": ep_dir.parent.name, "kind": "td3",
            "outcome": meta.get("outcome_str", ""), "T": T, "x": [float(r["x"]) for r in rows],
            "y": [float(r["y"]) for r in rows], "yaw": unwrap([float(r["yaw"]) for r in rows]),
            "ticks": ticks, "cmd_v": cmd_v, "cmd_w": cmd_w, "t_end": T[-1]}


def load_isolated_as_log(run_dir, kind):
    """Isolated experiment resampled like the navigation recorder: rows every LOG_DT on the PC arrival clock with
    the latest odometry pose that had arrived; ticks = command publish times. Also keeps the odometry twist."""
    cmds = read_csv(run_dir / "commands.csv")
    odom = read_csv(run_dir / "odometry.csv")
    ta = [float(r["t_s"]) for r in odom]
    ticks = [float(r["t_s"]) for r in cmds]
    cmd_v = [float(r["cmd_linear"]) for r in cmds]
    cmd_w = [float(r["cmd_angular"]) for r in cmds]
    t_end = min(ta[-1], ticks[-1])
    T, x, y, yaw, j = [], [], [], [], -1
    t = ta[0] + 0.01
    yaw_u = unwrap([float(r["yaw"]) for r in odom])
    while t <= t_end:
        while j + 1 < len(ta) and ta[j + 1] <= t:
            j += 1
        if j >= 0:
            T.append(t); x.append(float(odom[j]["x"])); y.append(float(odom[j]["y"])); yaw.append(yaw_u[j])
        t += LOG_DT
    return {"name": run_dir.name, "policy": kind, "kind": kind, "outcome": "", "T": T, "x": x, "y": y, "yaw": yaw,
            "ticks": ticks, "cmd_v": cmd_v, "cmd_w": cmd_w, "t_end": T[-1],
            "odom_t": ta, "odom_v": [float(r["v_linear"]) for r in odom], "odom_w": [float(r["v_angular"]) for r in odom]}


# ---- Increments and model ----------------------------------------------------------------------------------------

def increments(ep, rows=WINDOW_ROWS):
    """[(t_i, t_j, ds, dyaw)] over windows of `rows` log rows (stride 1 row)."""
    T, x, y, yaw = ep["T"], ep["x"], ep["y"], ep["yaw"]
    step = [((x[k + 1] - x[k]) * math.cos((yaw[k] + yaw[k + 1]) / 2)
             + (y[k + 1] - y[k]) * math.sin((yaw[k] + yaw[k + 1]) / 2)) for k in range(len(T) - 1)]
    out = []
    for i in range(len(T) - rows):
        j = i + rows
        out.append((T[i], T[j], sum(step[i:j]), yaw[j] - yaw[i]))
    return out


def command_at(ep, comp, t):
    """Commanded value (published at the ticks, held until the next tick); 0 before the first tick."""
    ticks, c = ep["ticks"], ep["cmd_" + comp]
    lo, hi = 0, len(ticks)
    while lo < hi:
        mid = (lo + hi) // 2
        if ticks[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    return c[lo - 1] if lo > 0 else 0.0


def model_integral(ep, comp, tau):
    """(t0, F): F[n] = integral of the first-order-filtered command up to t0 + n * FINE_DT."""
    t0 = min(ep["ticks"][0], ep["T"][0]) - 1.0
    n = int((ep["t_end"] + 1.0 - t0) / FINE_DT) + 2
    a = 1.0 - math.exp(-FINE_DT / tau) if tau > 0 else 1.0
    yv, F, acc = 0.0, [0.0] * n, 0.0
    for k in range(n):
        c = command_at(ep, comp, t0 + (k + 0.5) * FINE_DT)
        yv += (c - yv) * a
        acc += yv * FINE_DT
        F[k] = acc
    return t0, F


def F_at(t0F, t):
    t0, F = t0F
    u = (t - t0) / FINE_DT - 1
    k = int(math.floor(u))
    if k < 0:
        return 0.0
    if k + 1 >= len(F):
        return F[-1]
    return F[k] + (u - k) * (F[k + 1] - F[k])


def fit_fopdt(ep, comp, incs, taus=TAUS, thetas=THETAS):
    """Grid search over (theta, tau); K by least squares. Returns dict with best FOPDT and best pure delay."""
    meas = [d[2] if comp == "v" else d[3] for d in incs]
    best, best_pure, surface = None, None, {}
    for tau in taus:
        t0F = model_integral(ep, comp, tau)
        for th in thetas:
            p = [F_at(t0F, tj - th) - F_at(t0F, ti - th) for ti, tj, _, _ in incs]
            spp = sum(q * q for q in p)
            if spp <= 0:
                continue
            K = sum(m * q for m, q in zip(meas, p)) / spp
            sse = sum((m - K * q) ** 2 for m, q in zip(meas, p))
            surface[(tau, th)] = sse
            if best is None or sse < best[0]:
                best = (sse, th, tau, K)
            if tau == 0 and (best_pure is None or sse < best_pure[0]):
                best_pure = (sse, th, K)
    n = len(meas)
    sst = sum(m * m for m in meas)
    # theta + tau along the SSE valley: for each tau, the best theta
    valley = []
    for tau in taus:
        cands = [(surface[(tau, th)], th) for th in thetas if (tau, th) in surface]
        if cands:
            s, th = min(cands)
            valley.append((tau, th, s))
    return {"theta_s": best[1], "tau_s": best[2], "K": best[3], "rms": math.sqrt(best[0] / n),
            "r2": 1 - best[0] / sst if sst > 0 else NAN, "lag_s": best_pure[1], "K_pure": best_pure[2],
            "rms_pure": math.sqrt(best_pure[0] / n), "valley": valley, "n": n}


# ---- Analyses ----------------------------------------------------------------------------------------------------

def wheels(v, w):
    return v + w * WHEEL_SEP_M / 2, v - w * WHEEL_SEP_M / 2


def body(r, l):
    return (r + l) / 2, (r - l) / WHEEL_SEP_M


def saturate(v, w, model, wmax):
    """Commanded (v, w) after a wheel-speed limit: 'clip' each wheel, or 'scale' both (curvature kept)."""
    r, l = wheels(v, w)
    if model == "clip":
        r, l = max(-wmax, min(wmax, r)), max(-wmax, min(wmax, l))
    elif model == "scale":
        k = min(1.0, wmax / max(abs(r), abs(l), 1e-12))
        r, l = r * k, l * k
    return body(r, l)


def with_saturation(ep, model, wmax):
    out = dict(ep)
    vw = [saturate(v, w, model, wmax) for v, w in zip(ep["cmd_v"], ep["cmd_w"])]
    out["cmd_v"], out["cmd_w"] = [a for a, _ in vw], [b for _, b in vw]
    return out


def demand(ep, a, b):
    """Largest commanded |wheel speed| among the commands active over [a, b]."""
    vals = [(command_at(ep, "v", a), command_at(ep, "w", a))]
    vals += [(v, w) for t, v, w in zip(ep["ticks"], ep["cmd_v"], ep["cmd_w"]) if a <= t <= b]
    return max(max(abs(x) for x in wheels(v, w)) for v, w in vals)


def first_motion(ep):
    """(delay, resolution, t_motion): first nonzero command -> first log row whose pose differs from the pose at the
    command beyond MOTION_POS_M / MOTION_YAW_RAD."""
    k0 = next((k for k, (v, w) in enumerate(zip(ep["cmd_v"], ep["cmd_w"])) if v != 0 or w != 0), None)
    if k0 is None:
        return NAN, NAN, NAN
    t_cmd = ep["ticks"][k0]
    i0 = max([i for i, t in enumerate(ep["T"]) if t <= t_cmd] or [0])
    x0, y0, a0 = ep["x"][i0], ep["y"][i0], ep["yaw"][i0]
    for i in range(i0, len(ep["T"])):
        if math.hypot(ep["x"][i] - x0, ep["y"][i] - y0) > MOTION_POS_M or abs(ep["yaw"][i] - a0) > MOTION_YAW_RAD:
            return ep["T"][i] - t_cmd, ep["T"][i] - ep["T"][i - 1] if i > 0 else NAN, ep["T"][i]
    return NAN, NAN, NAN


def fit_saturation(ep, incs, fits):
    """SSE (v and w, each normalised by its unsaturated-fit residual variance) of each saturation model and limit,
    with theta, tau, K of the unsaturated fits. Returns {(model, wmax): score}."""
    scores = {}
    for model, wmaxes in (("none", [NAN]), ("clip", SAT_GRID), ("scale", SAT_GRID)):
        for wmax in wmaxes:
            e = ep if model == "none" else with_saturation(ep, model, wmax)
            score = 0.0
            for comp in ("v", "w"):
                f = fits.get(comp)
                if f is None:
                    continue
                t0F = model_integral(e, comp, f["tau_s"])
                for ti, tj, ds, dy in incs:
                    m = ds if comp == "v" else dy
                    p = f["K"] * (F_at(t0F, tj - f["theta_s"]) - F_at(t0F, ti - f["theta_s"]))
                    score += (m - p) ** 2 / f["rms"] ** 2
            scores[(model, wmax)] = score
    return scores


def gains_by_command(ep, comp, incs, fit, bins):
    """Per bin of |command| (and sign): sums of measured and unit-gain predicted increments, all windows and
    quasi-steady windows."""
    t0F = model_integral(ep, comp, fit["tau_s"])
    out = {}
    for ti, tj, ds, dy in incs:
        m = ds if comp == "v" else dy
        p = F_at(t0F, tj - fit["theta_s"]) - F_at(t0F, ti - fit["theta_s"])
        a, b = ti - fit["theta_s"] - 3 * fit["tau_s"] - 0.2, tj - fit["theta_s"]
        vals = [command_at(ep, comp, a)] + [c for t, c in zip(ep["ticks"], ep["cmd_" + comp]) if a <= t <= b]
        steady, level = max(vals) - min(vals) < STEADY_TOL[comp], statistics.fmean(vals)
        if abs(p) < 1e-9:
            continue
        v_level = statistics.fmean([command_at(ep, "v", a)] + [c for t, c in zip(ep["ticks"], ep["cmd_v"]) if a <= t <= b])
        regime = "in_place" if abs(v_level) < 0.05 else "driving"
        sign = "+" if level >= 0 else "-"
        for lo, hi in zip(bins, bins[1:]):
            if lo <= abs(level) < hi:
                for sel in (("all", "steady") if steady else ("all",)):
                    for reg in ("any", regime):
                        acc = out.setdefault((sel, reg, sign, lo, hi), [0.0, 0.0, 0])
                        acc[0] += m
                        acc[1] += p
                        acc[2] += 1
                break
    return out


def pivot_windows(ep, theta, t_from):
    """1 s windows (stride one row): measured wheel speeds vs commanded (lagged by theta, averaged over the window).
    A window is a pivot candidate when one wheel commanded >= PIV_CMD_MPS moves at < PIV_STALL_FRAC of its command
    while the other moves at >= PIV_OTHER_FRAC of its command."""
    T, rows = ep["T"], PIV_ROWS
    step = [((ep["x"][k + 1] - ep["x"][k]) * math.cos((ep["yaw"][k] + ep["yaw"][k + 1]) / 2)
             + (ep["y"][k + 1] - ep["y"][k]) * math.sin((ep["yaw"][k] + ep["yaw"][k + 1]) / 2)) for k in range(len(T) - 1)]
    tF = {c: model_integral(ep, c, 0.0) for c in ("v", "w")}
    out = []
    for i in range(len(T) - rows):
        j = i + rows
        if T[i] < t_from:
            continue
        dt = T[j] - T[i]
        v, w = sum(step[i:j]) / dt, (ep["yaw"][j] - ep["yaw"][i]) / dt
        vc = (F_at(tF["v"], T[j] - theta) - F_at(tF["v"], T[i] - theta)) / dt
        wc = (F_at(tF["w"], T[j] - theta) - F_at(tF["w"], T[i] - theta)) / dt
        (r, l), (rc, lc) = wheels(v, w), wheels(vc, wc)
        flag = None
        for name, m, c, om, oc in (("left", l, lc, r, rc), ("right", r, rc, l, lc)):
            if abs(c) >= PIV_CMD_MPS and m * math.copysign(1, c) < PIV_STALL_FRAC * abs(c) \
                    and abs(oc) >= PIV_CMD_MPS and om * math.copysign(1, oc) >= PIV_OTHER_FRAC * abs(oc):
                flag = name
        out.append({"t": (T[i] + T[j]) / 2, "flag": flag, "r": r, "l": l, "rc": rc, "lc": lc})
    return out


def pivot_events(pw):
    """Runs of pivot-candidate windows lasting >= PIV_HOLD_S: [(t_start, t_end, wheel)]."""
    events, start, wheel, last = [], None, None, None
    for p in pw + [{"t": math.inf, "flag": None}]:
        if p["flag"] and p["flag"] == wheel and last is not None and p["t"] - last < 0.15:
            last = p["t"]
            continue
        if start is not None and last - start >= PIV_HOLD_S:
            events.append((start, last, wheel))
        start, wheel, last = (p["t"], p["flag"], p["t"]) if p["flag"] else (None, None, None)
    return events


def reconstruction_noise(ep, t_from):
    """Isolated only: 0.2 s pose-increment velocities vs the mean odometry twist over the same window."""
    ev, ew = [], []
    ot = ep["odom_t"]
    for ti, tj, ds, dy in increments(ep):
        if ti < t_from:
            continue
        sel = [k for k, t in enumerate(ot) if ti < t <= tj]
        if len(sel) < 2:
            continue
        ev.append(ds / (tj - ti) - statistics.fmean(ep["odom_v"][k] for k in sel))
        ew.append(dy / (tj - ti) - statistics.fmean(ep["odom_w"][k] for k in sel))
    return statistics.pstdev(ev) if ev else NAN, statistics.pstdev(ew) if ew else NAN


def fmt(x):
    if isinstance(x, float):
        return "NaN" if math.isnan(x) else f"{x:.6g}"
    return x


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in fields})




def sliding_lag(ep, t_motion, comp="w", win_s=3.0, stride_s=1.5):
    """Pure-delay fits of the (saturation-modelled) command in short windows: [(age, lag, K, r2)], age = window
    start - first motion. Lags up to 1.5 s so command-delivery stalls are visible."""
    F = model_integral(with_saturation(ep, "scale", SAT_FIT_MPS), comp, 0.0)
    incs = [d for d in increments(ep) if d[0] >= t_motion]
    out, t = [], t_motion
    while t + win_s <= ep["T"][-1]:
        win = [d for d in incs if t <= d[0] < t + win_s]
        meas = [d[2] if comp == "v" else d[3] for d in win]
        sst = sum(m * m for m in meas)
        best = None
        for th in LONG_THETAS:
            pr = [F_at(F, tj - th) - F_at(F, ti - th) for ti, tj, _, _ in win]
            spp = sum(q * q for q in pr)
            if spp <= 0:
                continue
            K = sum(m * q for m, q in zip(meas, pr)) / spp
            sse = sum((m - K * q) ** 2 for m, q in zip(meas, pr))
            if best is None or sse < best[0]:
                best = (sse, th, K)
        if best and sst > 0:
            out.append((t - t_motion, best[1], best[2], 1 - best[0] / sst))
        t += stride_s
    return out


def wheel_direction(ep, t_from, theta=0.15, tau=0.08, sat=True):
    """Slow wheels (|command| 0.04-0.12 m/s): sums of measured and commanded wheel motion, forward and backward."""
    e = with_saturation(ep, "scale", SAT_FIT_MPS) if sat else ep
    tF = {k: model_integral(e, k, tau) for k in ("v", "w")}
    acc = {"forward": [0.0, 0.0, 0], "backward": [0.0, 0.0, 0]}
    for ti, tj, ds, dy in increments(ep):
        if ti < t_from:
            continue
        dt = tj - ti
        vc = (F_at(tF["v"], tj - theta) - F_at(tF["v"], ti - theta)) / dt
        wc = (F_at(tF["w"], tj - theta) - F_at(tF["w"], ti - theta)) / dt
        for m, c in zip(wheels(ds / dt, dy / dt), wheels(vc, wc)):
            if 0.04 <= abs(c) <= 0.12:
                a = acc["backward" if c < 0 else "forward"]
                a[0] += m; a[1] += c; a[2] += 1
    return acc


def straight_bias(ep, incs, theta):
    """(sum dyaw, sum dt) over windows driven by straight commands: |w| <= 0.02 rad/s and v >= 0.1 m/s."""
    sy = st = 0.0
    for ti, tj, ds, dy in incs:
        a, b = ti - theta - 0.2, tj - theta
        cmds = [(command_at(ep, "v", a), command_at(ep, "w", a))]
        cmds += [(v, w) for t, v, w in zip(ep["ticks"], ep["cmd_v"], ep["cmd_w"]) if a <= t <= b]
        if all(abs(w) <= 0.02 and v >= 0.1 for v, w in cmds):
            sy += dy
            st += tj - ti
    return sy, st


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nav = [load_nav_episode(Path(d)) for d in sorted(glob.glob(str(NAV_ROOT / "*" / "episode_*")))]
    iso = []
    for pattern, kind in (("run_[0-9]*", "isolated_linear"), ("angular_run_[0-9]*", "isolated_angular"),
                          ("combined_run_[0-9]*", "isolated_combined")):
        for d in sorted((HERE / "runs").glob(pattern)):
            if len(read_csv(d / "odometry.csv")) > 100:            # run_4 aborted before moving
                iso.append(load_isolated_as_log(d, kind))
    # In-place rotation without the pivot regime: angular runs cut at the pivot onset found by the angular analyzer
    onset = {r["run"]: float(r["t_linear_onset_s"]) for r in read_csv(HERE / "analysis" / "angular" / "per_run_metrics.csv")}
    for d in sorted((HERE / "runs").glob("angular_run_[0-9]*")):
        ep = load_isolated_as_log(d, "isolated_angular_clean")
        t_cut = onset.get(d.name, NAN)
        if not math.isnan(t_cut):
            keep_n = sum(t < t_cut - 1.0 for t in ep["T"])
            for k in ("T", "x", "y", "yaw"):
                ep[k] = ep[k][:keep_n]
            ep["t_end"] = ep["T"][-1] if ep["T"] else 0.0
        if len(ep["T"]) > 100:
            iso.append(ep)
    print(f"{len(nav)} navigation episodes, {len(iso)} isolated runs")

    fit_rows, gain_acc, fm_rows, piv_rows, cal_rows, bias_acc, sat_scores, keep = [], {}, [], [], [], {}, {}, []
    lag_rows, dir_acc = [], {}
    for ep in nav + iso:
        delay, res, t_motion = first_motion(ep)
        fm_rows.append({"episode": ep["name"], "kind": ep["kind"], "first_motion_delay_s": delay, "resolution_s": res})
        if math.isnan(t_motion):
            continue
        t_from = t_motion + STARTUP_MARGIN_S
        incs_all = [d for d in increments(ep) if d[0] >= t_from]
        incs = [d for d in incs_all if demand(ep, d[0] - 0.8, d[1]) <= SAT_SAFE_MPS]
        fits = {}
        for comp in ("v", "w"):
            if not any(ep["cmd_" + comp]):
                continue
            f = fits[comp] = fit_fopdt(ep, comp, incs)
            row = {"episode": ep["name"], "kind": ep["kind"], "policy": ep["policy"], "component": comp,
                   "n_windows": f["n"], "n_windows_excluded_saturated": len(incs_all) - len(incs),
                   "theta_s": f["theta_s"], "tau_s": f["tau_s"], "theta_plus_tau_s": f["theta_s"] + f["tau_s"],
                   "gain_K": f["K"], "rms_resid": f["rms"], "r2": f["r2"], "pure_lag_s": f["lag_s"],
                   "pure_lag_gain": f["K_pure"], "rms_pure": f["rms_pure"]}
            sse_min = min(x[2] for x in f["valley"])
            for tau_q in (0.0, 0.1, 0.2):
                v = next((x for x in f["valley"] if abs(x[0] - tau_q) < 1e-9), None)
                if v:
                    row[f"valley_tau{tau_q:.1f}_theta_s"], row[f"valley_tau{tau_q:.1f}_sse_ratio"] = v[1], v[2] / sse_min
            fit_rows.append(row)
            for key, acc in gains_by_command(ep, comp, incs, f, V_BINS if comp == "v" else W_BINS).items():
                g = gain_acc.setdefault((ep["kind"], comp) + key, {"m": 0.0, "p": 0.0, "n": 0, "eps": []})
                g["m"] += acc[0]; g["p"] += acc[1]; g["n"] += acc[2]
                if acc[2] >= 5 and abs(acc[1]) > 1e-9:
                    g["eps"].append((ep["name"], acc[0] / acc[1]))
        theta = fits["w"]["lag_s"] if "w" in fits else fits["v"]["lag_s"]
        if "v" in fits:
            sy, sdt = straight_bias(ep, incs, theta)
            b = bias_acc.setdefault(ep["kind"], {"y": 0.0, "t": 0.0, "eps": []})
            b["y"] += sy; b["t"] += sdt
            if sdt >= 2.0:
                b["eps"].append(sy / sdt)
        events = pivot_events(pivot_windows(ep, theta, t_from))
        piv_rows.append({"episode": ep["name"], "kind": ep["kind"], "analysed_from_s": t_from,
                         "n_pivot_events": len(events), "pivot_time_s": sum(e[1] - e[0] for e in events),
                         "events": ";".join(f"{a:.1f}-{b:.1f}s:{w}" for a, b, w in events)})
        if ep["kind"] in ("td3", "isolated_combined", "isolated_angular_clean"):
            for age, lag, K, r2 in sliding_lag(ep, t_motion):
                lag_rows.append({"episode": ep["name"], "kind": ep["kind"], "age_s": age, "lag_s": lag, "K": K, "r2": r2})
        wd = wheel_direction(ep, t_from, sat=ep["kind"] == "td3")
        key = ep["kind"] if ep["kind"] == "td3" else ep["name"]
        for direction, (m, c, n) in wd.items():
            a = dir_acc.setdefault((key, ep["kind"], direction), [0.0, 0.0, 0])
            a[0] += m; a[1] += c; a[2] += n
        if ep["kind"] == "td3":
            sat_scores[ep["name"]] = fit_saturation(ep, incs_all, fits)
            keep.append((ep, incs_all, fits, theta))
        else:
            nv, nw = reconstruction_noise(ep, t_from)
            cal_rows.append({"run": ep["name"], "kind": ep["kind"], "recon_noise_v_mps": nv, "recon_noise_w_radps": nw})

    write(OUT / "dynamics_fits.csv", fit_rows,
          ["episode", "kind", "policy", "component", "n_windows", "n_windows_excluded_saturated", "theta_s", "tau_s",
           "theta_plus_tau_s", "gain_K", "rms_resid", "r2", "pure_lag_s", "pure_lag_gain", "rms_pure",
           "valley_tau0.0_theta_s", "valley_tau0.0_sse_ratio", "valley_tau0.1_theta_s", "valley_tau0.1_sse_ratio",
           "valley_tau0.2_theta_s", "valley_tau0.2_sse_ratio"])

    gain_rows = []
    for (kind, comp, sel, reg, sign, lo, hi), g in sorted(gain_acc.items()):
        per_ep = [x for _, x in g["eps"]]
        gain_rows.append({"kind": kind, "component": comp, "selection": sel, "regime": reg, "sign": sign,
                          "cmd_abs_lo": lo, "cmd_abs_hi": hi, "n_windows": g["n"],
                          "gain_pooled": g["m"] / g["p"] if abs(g["p"]) > 1e-9 else NAN, "n_episodes": len(per_ep),
                          "gain_episode_mean": statistics.fmean(per_ep) if per_ep else NAN,
                          "gain_episode_std": statistics.stdev(per_ep) if len(per_ep) >= 2 else NAN})
    write(OUT / "gain_by_command.csv", gain_rows, list(gain_rows[0].keys()))

    # Saturation models: summed normalised SSE over TD3 episodes; per-episode best as a spread measure
    total = {}
    for sc in sat_scores.values():
        for key, v in sc.items():
            total[key] = total.get(key, 0.0) + v
    none = total[("none", NAN)] if ("none", NAN) in total else next(v for k, v in total.items() if k[0] == "none")
    sat_rows = []
    for (model, wmax), v in sorted(total.items(), key=lambda kv: kv[1]):
        best_ep = [min((s for s in sc.items() if s[0][0] == model), key=lambda s: s[1])[0][1]
                   for sc in sat_scores.values()] if model != "none" else []
        sat_rows.append({"model": model, "wheel_limit_mps": wmax, "score": v, "score_vs_no_saturation": v / none,
                         "episode_best_limit_mean": statistics.fmean(best_ep) if best_ep else NAN,
                         "episode_best_limit_std": statistics.stdev(best_ep) if len(best_ep) >= 2 else NAN})
    write(OUT / "saturation_models.csv", sat_rows, list(sat_rows[0].keys()))
    best_model, best_wmax = sat_rows[0]["model"], sat_rows[0]["wheel_limit_mps"]

    # Saturation bins: commanded outer-wheel demand vs measured, and the best saturation model's prediction
    sbins = [0.0, 0.15, 0.2, 0.25, 0.28, 0.31, 0.34, 0.37, 0.40, 0.43, 0.46]
    acc = {}
    for ep, incs_all, fits, theta in keep:
        e_sat = with_saturation(ep, best_model, best_wmax)
        for ti, tj, ds, dy in incs_all:
            dt, tc = tj - ti, (ti + tj) / 2 - theta
            vc, wc = command_at(ep, "v", tc), command_at(ep, "w", tc)
            vs, ws = command_at(e_sat, "v", tc), command_at(e_sat, "w", tc)
            d = max(abs(x) for x in wheels(vc, wc))
            for lo, hi in zip(sbins, sbins[1:]):
                if lo <= d < hi:
                    a = acc.setdefault((lo, hi), {k: [] for k in ("d", "outer", "v", "w", "vc", "wc", "vs", "ws")})
                    v, w = ds / dt, dy / dt
                    for k, x in (("d", d), ("outer", max(abs(y) for y in wheels(v, w))), ("v", v), ("w", abs(w)),
                                 ("vc", vc), ("wc", abs(wc)), ("vs", vs), ("ws", abs(ws))):
                        a[k].append(x)
                    break
    sbin_rows = []
    for (lo, hi), a in sorted(acc.items()):
        if len(a["d"]) < 20:
            continue
        ratio = lambda m, c: sum(a[m]) / sum(a[c]) if abs(sum(a[c])) > 1e-9 else NAN
        sbin_rows.append({"cmd_wheel_demand_lo": lo, "cmd_wheel_demand_hi": hi, "n_windows": len(a["d"]),
                          "cmd_wheel_demand_mean": statistics.fmean(a["d"]),
                          "meas_outer_wheel_mean": statistics.fmean(a["outer"]),
                          "v_gain_vs_cmd": ratio("v", "vc"), "w_gain_vs_cmd": ratio("w", "wc"),
                          "v_gain_vs_sat_model": ratio("v", "vs"), "w_gain_vs_sat_model": ratio("w", "ws"),
                          "mean_cmd_v": statistics.fmean(a["vc"]), "mean_cmd_abs_w": statistics.fmean(a["wc"])})
    write(OUT / "saturation_bins.csv", sbin_rows, list(sbin_rows[0].keys()))

    bias_rows = [{"kind": k, "straight_time_s": b["t"], "w_bias_radps": b["y"] / b["t"] if b["t"] else NAN,
                  "n_episodes": len(b["eps"]), "w_bias_episode_std": statistics.stdev(b["eps"]) if len(b["eps"]) > 1 else NAN}
                 for k, b in sorted(bias_acc.items())]
    write(OUT / "straight_bias.csv", bias_rows, list(bias_rows[0].keys()))
    write(OUT / "lag_windows.csv", lag_rows, ["episode", "kind", "age_s", "lag_s", "K", "r2"])
    lag_sum = []
    for kind in sorted({r["kind"] for r in lag_rows}):
        for lo, hi in ((0, 1.5), (1.5, 3), (3, 6), (6, 12), (12, 1e9)):
            a = sorted(r["lag_s"] for r in lag_rows if r["kind"] == kind and lo <= r["age_s"] < hi and r["r2"] > 0.5)
            if len(a) >= 3:
                lag_sum.append({"kind": kind, "age_lo_s": lo, "age_hi_s": hi if hi < 1e9 else NAN, "n_windows": len(a),
                                "lag_median_s": statistics.median(a), "lag_p10_s": a[int(0.1 * (len(a) - 1))],
                                "lag_p90_s": a[int(0.9 * (len(a) - 1))], "lag_max_s": a[-1],
                                "frac_lag_gt_0.3s": sum(x > 0.3 for x in a) / len(a)})
    write(OUT / "lag_vs_time.csv", lag_sum, list(lag_sum[0].keys()))
    dir_rows = [{"source": k, "kind": kind, "wheel_direction": d, "n_windows": n,
                 "measured_over_commanded": m / c if abs(c) > 1e-9 else NAN}
                for (k, kind, d), (m, c, n) in sorted(dir_acc.items()) if n > 0]
    write(OUT / "wheel_direction_check.csv", dir_rows, list(dir_rows[0].keys()))
    write(OUT / "first_motion.csv", fm_rows, ["episode", "kind", "first_motion_delay_s", "resolution_s"])
    write(OUT / "pivot_check.csv", piv_rows, ["episode", "kind", "analysed_from_s", "n_pivot_events", "pivot_time_s",
                                              "events"])
    write(OUT / "calibration_isolated.csv", cal_rows, ["run", "kind", "recon_noise_v_mps", "recon_noise_w_radps"])
    print(f"Best saturation model: {best_model}, wheel limit {best_wmax:.3f} m/s. Wrote {OUT}")


if __name__ == "__main__":
    main()
