"""Phase 1, real data: rolling classical fits of the branching ratio through the day.

Writes results/phase1_rolling_<stream>_w<length>.csv and figures/phase1_rolling_<stream>_w<length>.png.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_fomc import figures
from hawkes_fomc.config import DEFAULT_SESSION, RAW_DIR, RESULTS_DIR, SEED, Session
from hawkes_fomc.data import load_mbp10, midprice_change_events, trade_events
from hawkes_fomc.windows import FITTERS, rolling_windows, run_rolling


def main(length: float, stride: float, stream_name: str, n_boot: int, fitters: list[str], session: Session = DEFAULT_SESSION) -> None:
    files = sorted(RAW_DIR.glob("*.dbn.zst"))
    arr = load_mbp10(files[0])
    stream = trade_events(arr, session) if stream_name == "trades" else midprice_change_events(arr, session)
    windows = rolling_windows(session.open_s, session.close_s, length, stride)
    started = time.perf_counter()
    table = run_rolling(stream, windows, fitters, n_boot, SEED)
    tag = f"{stream_name}_w{int(length)}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"phase1_rolling_{tag}.csv", index=False)
    print(f"{len(table)} window fits in {time.perf_counter() - started:.0f}s")
    summary = table.groupby("fitter").agg(
        n_mean=("n", "mean"), n_min=("n", "min"), n_max=("n", "max"), converged=("converged", "mean"),
        ks_reject5=("ks_p", lambda p: float(np.mean(p < 0.05))), gain=("gain_over_poisson_per_event", "mean"),
        heldout_gain=("heldout_gain_over_poisson_per_event", "mean"), boot_sd=("boot_sd", "mean"),
    )
    print(summary.round(4).to_string())
    curves = []
    for name in fitters:
        sub = table[table["fitter"] == name]
        curve = {"name": name, "t": sub["t_mid"].to_numpy(), "n": sub["n"].to_numpy(), "flagged": (sub["ks_p"] < 0.05).to_numpy()}
        if n_boot > 0:
            curve["lo"], curve["hi"] = sub["boot_lo"].to_numpy(), sub["boot_hi"].to_numpy()
        curves.append(curve)
    title = f"Branching ratio, {stream_name}, {int(length / 60)}-minute windows, {int(stride / 60)}-minute stride; bands: parametric bootstrap 95%"
    print("figure:", figures.save(figures.branching_ratio_figure(curves, session, title), f"phase1_rolling_{tag}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=float, default=600.0)
    p.add_argument("--stride", type=float, default=300.0)
    p.add_argument("--stream", choices=["trades", "midprice"], default="trades")
    p.add_argument("--boot", type=int, default=100)
    p.add_argument("--fitters", nargs="+", default=list(FITTERS))
    a = p.parse_args()
    main(a.length, a.stride, a.stream, a.boot, a.fitters)
