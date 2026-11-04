#!/usr/bin/env python3
"""Actuator behaviour in the final approach of TD3 navigation episodes vs the rest of each trajectory.

Uses the data, command timeline, saturation model and delay estimator of analyze_td3_navigation.py. Standard
library only, no plots.

Segments (successful episodes only; the others never approach the goal):
  final  last FINAL_PATH_M of travelled path (recorder distance_traveled_m), up to the end of the episode
         (sensitivity variant: straight-line goal distance <= FINAL_PATH_M)
  rest   from first motion + STARTUP_EXCLUDE_S (first-motion dead time and the elevated-delay period after it
         are excluded) to the start of the final segment
Metrics per segment, from the commands at the ticks inside it and the pose increments of its 0.2 s windows:
  cmd_v_mean, cmd_abs_w_mean                 commanded speeds
  sat_frac                                   fraction of ticks whose commanded wheel demand
                                             max(|v +/- w b/2|) exceeds the wheel limit (0.305 m/s)
  demand_mean                                mean commanded wheel demand, m/s
  dw_abs_mean, dw_big_frac, w_sign_flips_per_s  |change of w| per tick, ticks with |dw| > 0.5 rad/s,
                                             sign changes of w per second (|w| >= 0.05 rad/s only)
  dv_abs_mean                                |change of v| per tick
  lag_w_s, gain_w, r2_w                      pure-delay fit (0-1.5 s grid) of the heading increments to the
                                             saturation-modelled w command, gain by least squares
  lag_v_s, gain_v                            same for forward displacement (v saturated much of the time: weak)
  lag_w_chunk_median_s                       rest only: median pure delay over non-overlapping chunks of the same
                                             duration as the episode's final segment (controls for segment length)
Paired statistics over episodes: final - rest per episode; mean, std, median, number of episodes with
final > rest, and the Wilcoxon signed-rank p-value (normal approximation, two-sided).

Output: analysis/td3_final_approach/{per_episode.csv, summary.csv}

Usage: python3 RealRobot/motion_dynamic_charac/analyze_td3_final_approach.py
"""

import csv
import glob
import math
import statistics
from pathlib import Path

import analyze_td3_navigation as A

OUT = A.HERE / "analysis" / "td3_final_approach"
FINAL_PATH_M = 1.5
STARTUP_EXCLUDE_S = 3.0      # after first motion (dead time + ~1.5 s of elevated delay, see lag_vs_time.csv)
BIG_DW = 0.5                 # rad/s change between consecutive ticks
LAG_GRID = [i * 0.01 for i in range(0, 151)]
NAN = float("nan")
METRICS = ["duration_s", "cmd_v_mean", "cmd_abs_w_mean", "sat_frac", "demand_mean", "dw_abs_mean", "dw_big_frac",
           "w_sign_flips_per_s", "dv_abs_mean", "lag_w_s", "gain_w", "r2_w", "lag_v_s", "gain_v"]


def demand(v, w):
    return max(abs(x) for x in A.wheels(v, w))


def pure_delay_fit(F, incs, comp):
    meas = [d[2] if comp == "v" else d[3] for d in incs]
    sst = sum(m * m for m in meas)
    best = None
    for th in LAG_GRID:
        p = [A.F_at(F, tj - th) - A.F_at(F, ti - th) for ti, tj, _, _ in incs]
        spp = sum(q * q for q in p)
        if spp <= 0:
            continue
        K = sum(m * q for m, q in zip(meas, p)) / spp
        sse = sum((m - K * q) ** 2 for m, q in zip(meas, p))
        if best is None or sse < best[0]:
            best = (sse, th, K)
    if best is None or sst <= 0:
        return NAN, NAN, NAN
    return best[1], best[2], 1 - best[0] / sst


def segment_metrics(ep, Fs, incs_all, a, b):
    """Metrics of the segment [a, b)."""
    k = [i for i, t in enumerate(ep["ticks"]) if a <= t < b]
    v = [ep["cmd_v"][i] for i in k]
    w = [ep["cmd_w"][i] for i in k]
    dw = [abs(ep["cmd_w"][i] - ep["cmd_w"][i - 1]) for i in k if i > 0]
    dv = [abs(ep["cmd_v"][i] - ep["cmd_v"][i - 1]) for i in k if i > 0]
    signs = [math.copysign(1, x) for x in w if abs(x) >= 0.05]
    incs = [d for d in incs_all if a <= d[0] and d[1] <= b]
    m = {"duration_s": b - a, "n_ticks": len(k), "n_windows": len(incs),
         "cmd_v_mean": statistics.fmean(v), "cmd_abs_w_mean": statistics.fmean(abs(x) for x in w),
         "sat_frac": sum(demand(x, y) > A.SAT_FIT_MPS for x, y in zip(v, w)) / len(k),
         "demand_mean": statistics.fmean(demand(x, y) for x, y in zip(v, w)),
         "dw_abs_mean": statistics.fmean(dw), "dw_big_frac": sum(x > BIG_DW for x in dw) / len(dw),
         "w_sign_flips_per_s": sum(s1 != s2 for s1, s2 in zip(signs, signs[1:])) / (b - a),
         "dv_abs_mean": statistics.fmean(dv)}
    m["lag_w_s"], m["gain_w"], m["r2_w"] = pure_delay_fit(Fs["w"], incs, "w")
    m["lag_v_s"], m["gain_v"], _ = pure_delay_fit(Fs["v"], incs, "v")
    return m


def wilcoxon_p(diffs):
    """Two-sided Wilcoxon signed-rank test, normal approximation with tie-corrected variance (zeros dropped)."""
    d = [x for x in diffs if x != 0 and not math.isnan(x)]
    n = len(d)
    if n < 6:
        return NAN
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks, i = [0.0] * n, 0
    ties = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    z = (w_plus - mean) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(str(A.NAV_ROOT / "*" / "episode_*"))):
        ep = A.load_nav_episode(Path(d))
        if ep["outcome"] != "SUCCESS":
            continue
        traj = A.read_csv(Path(d) / "trajectory.csv")
        dist = [float(r["distance_traveled_m"]) for r in traj]
        gdist = [float(r["goal_distance"]) for r in traj]
        _, _, t_motion = A.first_motion(ep)
        t_rest = t_motion + STARTUP_EXCLUDE_S
        t_end = ep["T"][-1]
        e_sat = A.with_saturation(ep, "scale", A.SAT_FIT_MPS)
        Fs = {c: A.model_integral(e_sat, c, 0.0) for c in ("v", "w")}
        incs = A.increments(ep)
        for definition, t_b in (("path", next(t for t, x in zip(ep["T"], dist) if dist[-1] - x <= FINAL_PATH_M)),
                                ("goal_distance", next(t for t, g in zip(ep["T"], gdist) if g <= FINAL_PATH_M))):
            fin = segment_metrics(ep, Fs, incs, t_b, t_end + 0.01)
            rest = segment_metrics(ep, Fs, incs, t_rest, t_b)
            # rest delay in chunks as long as the final segment
            L, chunks, a = t_end - t_b, [], t_rest
            while a + L <= t_b:
                c = [x for x in incs if a <= x[0] and x[1] <= a + L]
                lag, _, r2 = pure_delay_fit(Fs["w"], c, "w")
                if r2 > 0.5:
                    chunks.append(lag)
                a += L
            row = {"episode": ep["name"], "definition": definition, "t_final_start_s": t_b, "t_rest_start_s": t_rest,
                   "rest_chunks": len(chunks),
                   "rest_lag_w_chunk_median_s": statistics.median(chunks) if chunks else NAN,
                   "rest_lag_w_chunk_p90_s": sorted(chunks)[int(0.9 * (len(chunks) - 1))] if chunks else NAN}
            for name in METRICS + ["n_ticks", "n_windows"]:
                row["final_" + name], row["rest_" + name] = fin[name], rest[name]
            rows.append(row)
        print(f"{ep['name']}: final from {rows[-2]['t_final_start_s']:.1f} s of {t_end:.1f} s")

    fields = ["episode", "definition", "t_final_start_s", "t_rest_start_s"] + \
             [f"{s}_{m}" for m in METRICS + ["n_ticks", "n_windows"] for s in ("final", "rest")] + \
             ["rest_chunks", "rest_lag_w_chunk_median_s", "rest_lag_w_chunk_p90_s"]
    write(OUT / "per_episode.csv", rows, fields)

    summ = []
    for definition in ("path", "goal_distance"):
        rs = [r for r in rows if r["definition"] == definition]
        pairs = [(m, "rest_" + m) for m in METRICS] + [("lag_w_s", "rest_lag_w_chunk_median_s")]
        for m, rest_col in pairs:
            f = [r["final_" + m] for r in rs]
            g = [r[rest_col] for r in rs]
            ok = [(x, y) for x, y in zip(f, g) if not (math.isnan(x) or math.isnan(y))]
            dif = [x - y for x, y in ok]
            summ.append({"definition": definition, "metric": m,
                         "rest_reference": "chunks of equal length" if rest_col.endswith("chunk_median_s") else "whole rest",
                         "n_episodes": len(ok), "final_mean": statistics.fmean(x for x, _ in ok),
                         "final_std": statistics.stdev(x for x, _ in ok), "rest_mean": statistics.fmean(y for _, y in ok),
                         "rest_std": statistics.stdev(y for _, y in ok), "diff_mean": statistics.fmean(dif),
                         "diff_std": statistics.stdev(dif), "diff_median": statistics.median(dif),
                         "n_final_greater": sum(x > 0 for x in dif), "wilcoxon_p": wilcoxon_p(dif)})
    write(OUT / "summary.csv", summ, list(summ[0].keys()))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
