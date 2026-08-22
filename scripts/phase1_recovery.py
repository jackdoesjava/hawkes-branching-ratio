"""Phase 1: simulation-recovery study for the classical estimators.

Simulates Hawkes processes with known branching ratio at realistic per-window event counts and
checks that each estimator recovers it. Writes results/phase1_recovery_fits.csv (every fit),
results/phase1_recovery.csv (summary table) and figures/phase1_recovery.png.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_fomc import figures
from hawkes_fomc.config import RESULTS_DIR, SEED
from hawkes_fomc.hawkes_classical import DEFAULT_BETAS, fit_exp, fit_fixed_decays, ks_exp1, rescaled_intervals
from hawkes_fomc.simulate import HawkesParams, simulate_window

T0, T1 = 34200.0, 34800.0  # a 10-minute window at 09:30 ET in seconds-since-midnight units
N_TRUE = (0.3, 0.6, 0.9)
EVENTS = (500, 2000, 5000)
SEEDS = 50
BURN_IN = 300.0

KERNELS = {
    # single exponential, 0.5 s timescale
    "exp": lambda n: (np.array([n]), np.array([2.0])),
    # three on-grid components: 10 ms, 100 ms, 1 s
    "mixture": lambda n: (n * np.array([0.25, 0.5, 0.25]), np.array([100.0, 10.0, 1.0])),
    # single exponential at 0.3 s, deliberately off the fixed decay grid
    "exp_offgrid": lambda n: (np.array([n]), np.array([1.0 / 0.3])),
}

DECADE_BETAS = 1.0 / np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])

FITTERS = {
    "exp_true_beta": lambda t, h, betas: fit_fixed_decays(t, h, T0, T1, betas),
    "exp_profile": lambda t, h, betas: fit_exp(t, h, T0, T1),
    "grid_decade_const": lambda t, h, betas: fit_fixed_decays(t, h, T0, T1, DECADE_BETAS),
    "grid_default_const": lambda t, h, betas: fit_fixed_decays(t, h, T0, T1, DEFAULT_BETAS),
    "grid_default_spline2min": lambda t, h, betas: fit_fixed_decays(t, h, T0, T1, DEFAULT_BETAS, "spline", 5),
}


def run_cell(cell: tuple[str, float, int, int]) -> list[dict]:
    kernel, n_true, events, seed = cell
    alpha, betas = KERNELS[kernel](n_true)
    mu = events * (1 - n_true) / (T1 - T0)
    params = HawkesParams(mu=mu, alpha=alpha, betas=betas)
    history, times = simulate_window(params, T0, T1, BURN_IN, np.random.default_rng(SEED + seed))
    rows = []
    for name, fitter in FITTERS.items():
        if name == "exp_true_beta" and kernel == "mixture":
            continue
        start = time.perf_counter()
        fit = fitter(times, history, betas)
        stat, p = ks_exp1(rescaled_intervals(fit, times, history, T0, T1))
        rows.append({
            "kernel": kernel, "n_true": n_true, "events": events, "seed": seed, "fitter": name,
            "n_events": len(times), "n_hat": fit.n, "se_n": fit.se_n, "converged": fit.converged,
            "loglik": fit.loglik, "ks_stat": stat, "ks_p": p, "seconds": time.perf_counter() - start,
            "beta_hat": fit.betas[0] if name == "exp_profile" else np.nan,
        })
    return rows


def summarise(fits: pd.DataFrame) -> pd.DataFrame:
    fits = fits.assign(err=fits["n_hat"] - fits["n_true"], covered=(fits["n_hat"] - fits["n_true"]).abs() <= 1.96 * fits["se_n"])
    g = fits.groupby(["kernel", "fitter", "n_true", "events"], sort=False)
    out = g.agg(
        mean=("n_hat", "mean"), sd=("n_hat", "std"), bias=("err", "mean"), rmse=("err", lambda e: float(np.sqrt(np.mean(e**2)))),
        mean_se=("se_n", "mean"), coverage95=("covered", "mean"), ks_reject5=("ks_p", lambda p: float(np.mean(p < 0.05))),
        converged=("converged", "mean"), mean_events=("n_events", "mean"), seconds=("seconds", "mean"),
    ).reset_index()
    return out


def main() -> None:
    cells = [(k, n, e, s) for k, n, e in product(KERNELS, N_TRUE, EVENTS) for s in range(SEEDS)]
    started = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        rows = [r for batch in pool.map(run_cell, cells, chunksize=8) for r in batch]
    fits = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fits.to_csv(RESULTS_DIR / "phase1_recovery_fits.csv", index=False)
    table = summarise(fits)
    table.to_csv(RESULTS_DIR / "phase1_recovery.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 500)
    print(table.round(4).to_string(index=False))
    print(f"{len(fits)} fits in {time.perf_counter() - started:.0f}s")
    fig = figures.recovery_figure(table, "Simulation recovery of the branching ratio, 10-minute windows, 50 seeds per cell")
    print("figure:", figures.save(fig, "phase1_recovery"))


if __name__ == "__main__":
    main()
