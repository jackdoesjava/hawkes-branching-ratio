"""Phase 1: stress tests of the classical estimator beyond the clean recovery study.

Each scenario simulates 10-minute windows with ~2000 events and reports what the default fitters
return, so the README can state what the estimator does when its assumptions fail:
near-critical n, a power-law kernel whose mass mostly lies beyond the window, baseline bursts and
steps that a 2-minute spline cannot follow, a genuine slow component that a flexible baseline might
steal, and multi-execution sweeps that are or are not merged.
Writes results/phase1_stress.csv (every fit) and results/phase1_stress_summary.csv.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_fomc.config import RESULTS_DIR, SEED
from hawkes_fomc.data import trade_events
from hawkes_fomc.hawkes_classical import DEFAULT_BETAS, fit_fixed_decays, gof_tests, rescaled_intervals
from hawkes_fomc.simulate import HawkesParams, simulate_hawkes

T0, T1 = 34200.0, 34800.0
SPAN = T1 - T0
SEEDS = 30
TARGET = 2000


def fitters():
    return {
        "grid_const": lambda t, h: fit_fixed_decays(t, h, T0, T1, DEFAULT_BETAS),
        "grid_spline2min": lambda t, h: fit_fixed_decays(t, h, T0, T1, DEFAULT_BETAS, "spline", 5),
        "grid_spline1min": lambda t, h: fit_fixed_decays(t, h, T0, T1, DEFAULT_BETAS, "spline", 10),
    }


def power_law_cluster(mu: float, n: float, epsilon: float, tau0: float, s_max: float, t_start: float, t_end: float, rng):
    """Cluster simulation with phi(s) = n (eps/tau0) (1 + s/tau0)^-(1+eps), delays truncated at s_max."""
    parents = np.sort(rng.uniform(t_start, t_end, rng.poisson(mu * (t_end - t_start))))
    out = [parents]
    while len(parents):
        kids = rng.poisson(n, size=len(parents))
        parent_times = np.repeat(parents, kids)
        delays = tau0 * ((1.0 - rng.uniform(size=len(parent_times))) ** (-1.0 / epsilon) - 1.0)
        children = parent_times + delays
        children = children[(delays < s_max) & (children < t_end)]
        out.append(children)
        parents = children
    return np.unique(np.sort(np.concatenate(out)))


def split_into_sweeps(times: np.ndarray, rng, fraction: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Replace a fraction of events by 2-6 executions one nanosecond apart, mimicking unmerged ITCH sweeps."""
    multiplicity = rng.choice([2, 3, 4, 5, 6], size=len(times), p=[0.65, 0.20, 0.08, 0.04, 0.03])
    is_sweep = rng.uniform(size=len(times)) < fraction
    multiplicity = np.where(is_sweep, multiplicity, 1)
    expanded = np.repeat(times, multiplicity) + np.concatenate([np.arange(m) * 1e-9 for m in multiplicity])
    return np.sort(expanded), multiplicity


def run(cell: tuple[str, int]) -> list[dict]:
    scenario, seed = cell
    rng = np.random.default_rng([SEED, seed])
    rows = []
    burn = 300.0
    extra = {}
    if scenario.startswith("near_critical"):
        n = float(scenario.split("_")[-1])
        params = HawkesParams(mu=TARGET * (1 - n) / SPAN, alpha=np.array([n]), betas=np.array([2.0]))
        times = simulate_hawkes(params, T0 - 900.0, T1, rng)
        truth = {"n_true": n, "n_true_within_10s": n}
    elif scenario == "power_law_eps0.15":
        n, eps, tau0, s_max = 0.9, 0.15, 0.1, 3600.0
        mass_10s = 1 - (1 + 10.0 / tau0) ** (-eps)
        truncated_n = n * (1 - (1 + s_max / tau0) ** (-eps))
        times = power_law_cluster(TARGET * (1 - truncated_n) / SPAN, n, eps, tau0, s_max, T0 - s_max, T1, rng)
        truth = {"n_true": truncated_n, "n_true_within_10s": n * mass_10s}
    elif scenario in ("burst_mu", "step_mu", "smooth_mu"):
        n = 0.3
        base_rate = TARGET * (1 - n) / SPAN
        if scenario == "burst_mu":
            starts = np.array([T0 + 90.0, T0 + 270.0, T0 + 450.0])
            rate = lambda t: base_rate * (0.7 + 4.0 * np.any((t[:, None] >= starts) & (t[:, None] < starts + 30.0), axis=1))
            mu_max = base_rate * 4.7
        elif scenario == "step_mu":
            rate = lambda t: base_rate * np.where(t < T0 + 300.0, 0.6, 1.4)
            mu_max = base_rate * 1.4
        else:
            rate = lambda t: base_rate * (1.0 + 0.8 * np.sin(2 * np.pi * (t - T0) / 600.0))
            mu_max = base_rate * 1.8
        params = HawkesParams(mu=rate, alpha=np.array([n]), betas=np.array([2.0]), mu_max=mu_max)
        times = simulate_hawkes(params, T0 - burn, T1, rng)
        truth = {"n_true": n, "n_true_within_10s": n}
    elif scenario == "slow_component_const_mu":
        alpha, betas = np.array([0.6, 0.3]), np.array([2.0, 0.1])
        params = HawkesParams(mu=TARGET * 0.1 / SPAN, alpha=alpha, betas=betas)
        times = simulate_hawkes(params, T0 - 900.0, T1, rng)
        truth = {"n_true": 0.9, "n_true_within_10s": 0.6 + 0.3 * (1 - np.exp(-1.0))}
    elif scenario in ("sweeps_unmerged", "sweeps_merged"):
        n = 0.6
        params = HawkesParams(mu=TARGET * (1 - n) / SPAN, alpha=np.array([n]), betas=np.array([2.0]))
        clean = simulate_hawkes(params, T0 - burn, T1, rng)
        expanded, multiplicity = split_into_sweeps(clean, rng)
        if scenario == "sweeps_merged":
            # merge executions within 10 us, as the data layer does for real sweeps
            keep = np.ones(len(expanded), dtype=bool)
            keep[1:] = np.diff(expanded) > 1e-5
            times = expanded[keep]
        else:
            times = expanded
        truth = {"n_true": n, "n_true_within_10s": n}
        extra = {"executions_per_event": float(multiplicity.mean())}
    else:
        raise ValueError(scenario)
    history, window = times[times < T0], times[times >= T0]
    for name, fitter in fitters().items():
        fit = fitter(window, history)
        gof = gof_tests(rescaled_intervals(fit, window, history, T0, T1))
        rows.append({
            "scenario": scenario, "seed": seed, "fitter": name, "n_events": len(window), "n_hat": fit.n, "se_n": fit.se_n,
            "mu_mean_hat": fit.mu_mean, "converged": fit.converged, "condition_number": fit.condition_number,
            "alpha_slowest_two": float(fit.alpha[-2:].sum()), **truth, **extra, **gof,
        })
    return rows


SCENARIOS = [
    "near_critical_0.95", "near_critical_0.99", "power_law_eps0.15", "smooth_mu", "step_mu", "burst_mu",
    "slow_component_const_mu", "sweeps_unmerged", "sweeps_merged",
]


def main() -> None:
    cells = [(s, seed) for s in SCENARIOS for seed in range(SEEDS)]
    started = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        rows = [r for batch in pool.map(run, cells, chunksize=4) for r in batch]
    fits = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fits.to_csv(RESULTS_DIR / "phase1_stress.csv", index=False)
    fits["bias_total"] = fits["n_hat"] - fits["n_true"]
    fits["bias_within_10s"] = fits["n_hat"] - fits["n_true_within_10s"]
    summary = fits.groupby(["scenario", "fitter"], sort=False).agg(
        n_true=("n_true", "first"), n_true_within_10s=("n_true_within_10s", "first"), n_hat_mean=("n_hat", "mean"),
        n_hat_sd=("n_hat", "std"), bias_total=("bias_total", "mean"), bias_within_10s=("bias_within_10s", "mean"),
        mean_events=("n_events", "mean"), converged=("converged", "mean"), ks_reject5=("ks_p", lambda p: float(np.mean(p < 0.05))),
        lb_reject5=("lb_p", lambda p: float(np.mean(p < 0.05))), condition_median=("condition_number", "median"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "phase1_stress_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(summary.round(3).to_string(index=False))
    print(f"{len(fits)} fits in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
