"""Phase 1, real data: robustness of the rolling branching ratio to data-handling and model choices.

Variants: sweep-merge horizon, timestamp coarsening + jitter (the Filimonov-Sornette rounding effect),
window length, fast end of the decay grid, and excluding unsided executions. No bootstrap here; the
question is whether the n(t) curve moves, not its sampling error.
Writes results/phase1_robustness.csv, results/phase1_robustness_summary.csv and figures/phase1_robustness_*.png.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_fomc import figures
from hawkes_fomc.config import DEFAULT_SESSION, RAW_DIR, RESULTS_DIR, SEED
from hawkes_fomc.data import EventStream, coarsen_times, jitter_times, load_mbp10, trade_events
from hawkes_fomc.hawkes_classical import fit_fixed_decays
from hawkes_fomc.windows import FITTERS, rolling_windows, run_rolling

GRID_1MS = 1.0 / np.geomspace(1e-3, 10.0, 9)
FITTERS["mix_const_1ms"] = lambda t, h, t0, t1: fit_fixed_decays(t, h, t0, t1, GRID_1MS)
FITTERS["mix_spline_1ms"] = lambda t, h, t0, t1: fit_fixed_decays(t, h, t0, t1, GRID_1MS, "spline", 5)

BASE_FITTERS = ["mix_const", "mix_spline"]


def derived_stream(base: EventStream, times: np.ndarray, name: str) -> EventStream:
    return EventStream(name=name, times=times, session=base.session)


def variants(arr, session) -> list[tuple[str, str, EventStream, float, float, list[str]]]:
    """(family, label, stream, window length, stride, fitters)."""
    base = trade_events(arr, session)
    out = [("base", "merge 0 (identical ts)", base, 600.0, 300.0, BASE_FITTERS)]
    for merge in (1e-5, 1e-4, 1e-3):
        out.append(("merge", f"merge within {merge * 1e6:g} us", trade_events(arr, session, merge_within=merge), 600.0, 300.0, BASE_FITTERS))
    out.append(("merge", "sided executions only", trade_events(arr, session, include_unsided=False), 600.0, 300.0, BASE_FITTERS))
    for resolution in (1e-3, 1.0):
        for seed in range(3):
            rng = np.random.default_rng([SEED, int(resolution * 1e6), seed])
            times = jitter_times(coarsen_times(base.times, resolution), resolution, rng)
            out.append(("rounding", f"rounded to {resolution:g} s + jitter, seed {seed}", derived_stream(base, times, "rounded"), 600.0, 300.0, BASE_FITTERS))
    out.append(("window", "5-minute windows", base, 300.0, 150.0, BASE_FITTERS))
    out.append(("window", "20-minute windows", base, 1200.0, 600.0, BASE_FITTERS))
    out.append(("grid", "fastest component 1 ms", base, 600.0, 300.0, ["mix_const_1ms", "mix_spline_1ms"]))
    return out


def main(session=DEFAULT_SESSION) -> None:
    arr = load_mbp10(sorted(RAW_DIR.glob("*.dbn.zst"))[0])
    frames = []
    started = time.perf_counter()
    for family, label, stream, length, stride, fitters in variants(arr, session):
        windows = rolling_windows(session.open_s, session.close_s, length, stride)
        table = run_rolling(stream, windows, fitters, n_boot=0, seed=SEED)
        table.insert(0, "family", family)
        table.insert(1, "variant", label)
        table["fitter"] = table["fitter"].str.replace("_1ms", "", regex=False)
        frames.append(table)
        print(f"{label:45s} {len(table):4d} fits, {time.perf_counter() - started:5.0f}s", flush=True)
    table = pd.concat(frames, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "phase1_robustness.csv", index=False)

    base = table[table["family"] == "base"].set_index(["fitter", "t_mid"])["n"]
    rows = []
    for (family, label, fitter), sub in table.groupby(["family", "variant", "fitter"], sort=False):
        matched = sub.set_index("t_mid")["n"]
        common = matched.index.intersection(base.loc[fitter].index)
        delta = matched.loc[common] - base.loc[fitter].loc[common] if len(common) else pd.Series(dtype=float)
        rows.append({
            "family": family, "variant": label, "fitter": fitter, "windows": len(sub),
            "n_mean": sub["n"].mean(), "n_sd_across_windows": sub["n"].std(),
            "mean_delta_vs_base": delta.mean() if len(delta) else np.nan,
            "mean_abs_delta_vs_base": delta.abs().mean() if len(delta) else np.nan,
            "corr_with_base": np.corrcoef(matched.loc[common], base.loc[fitter].loc[common])[0, 1] if len(common) > 2 else np.nan,
            "ks_reject5": (sub["ks_p"] < 0.05).mean(), "lb_reject5": (sub["lb_p"] < 0.05).mean(),
            "dispersion_reject5": (sub["dispersion_p"] < 0.05).mean(),
            "heldout_gain_kernel_only": sub["heldout_gain_kernel_only_per_event"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "phase1_robustness_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(summary.round(3).to_string(index=False))

    for family in ("merge", "rounding", "window", "grid"):
        curves = []
        for label, sub in table[(table["family"].isin(["base", family])) & (table["fitter"] == "mix_spline")].groupby("variant", sort=False):
            sub = sub.sort_values("t_mid")
            curves.append({"name": label, "t": sub["t_mid"].to_numpy(), "n": sub["n"].to_numpy()})
        fig = figures.branching_ratio_figure(curves, session, f"Robustness, mixture kernel with spline baseline: {family}")
        print("figure:", figures.save(fig, f"phase1_robustness_{family}"))


if __name__ == "__main__":
    main()
