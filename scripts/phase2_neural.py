"""Phase 2: the neural Hawkes model with an identifiable branching ratio.

Part A validates the estimator on simulated data with known n (exponential, on-grid mixture, a
power-law truth and a smoothly varying baseline), side by side with the classical fitters.
Part B fits rolling windows on the real data with constant and MLP baselines, scores held-out
log-likelihood against the classical fits, and bootstraps n with warm-started refits.
Writes results/phase2_recovery.csv, results/phase2_rolling_<stream>_w<length>.csv and figures/phase2_*.png.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_fomc import figures
from hawkes_fomc.config import DEFAULT_SESSION, RAW_DIR, RESULTS_DIR, SEED, Session
from hawkes_fomc.data import EventStream, load_mbp10, trade_events
from hawkes_fomc.hawkes_classical import DEFAULT_BETAS, fit_fixed_decays
from hawkes_fomc.hawkes_neural import (NeuralFit, SimulationTooLarge, fit_neural, gof_neural, loglik_neural,
                                       simulate_neural)
from hawkes_fomc.simulate import HawkesParams, simulate_hawkes
from hawkes_fomc.windows import Window, poisson_loglik, rolling_windows

T0, T1 = 34200.0, 34800.0
S_MAX = 60.0
STEPS = 400


def _single_thread() -> None:
    torch.set_num_threads(1)


# Part A ------------------------------------------------------------------------------------------
def simulate_scenario(scenario: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng([SEED, seed])
    target = 2000
    if scenario.startswith("exp_"):
        n = float(scenario.split("_")[1])
        params = HawkesParams(mu=target * (1 - n) / 600.0, alpha=np.array([n]), betas=np.array([2.0]))
        times, truth = simulate_hawkes(params, T0 - 300.0, T1, rng), {"n_true": n, "n_true_within_60s": n}
    elif scenario == "mixture_0.6":
        params = HawkesParams(mu=target * 0.4 / 600.0, alpha=0.6 * np.array([0.25, 0.5, 0.25]), betas=np.array([100.0, 10.0, 1.0]))
        times, truth = simulate_hawkes(params, T0 - 300.0, T1, rng), {"n_true": 0.6, "n_true_within_60s": 0.6}
    elif scenario == "power_law_0.9":
        n, eps, tau0, s_cut = 0.9, 0.15, 0.1, 3600.0
        truncated = n * (1 - (1 + s_cut / tau0) ** (-eps))
        parents = np.sort(rng.uniform(T0 - s_cut, T1, rng.poisson(target * (1 - truncated) / 600.0 * (T1 - T0 + s_cut))))
        out = [parents]
        while len(parents):
            kids = rng.poisson(n, size=len(parents))
            parent_times = np.repeat(parents, kids)
            delays = tau0 * ((1.0 - rng.uniform(size=len(parent_times))) ** (-1.0 / eps) - 1.0)
            children = parent_times + delays
            children = children[(delays < s_cut) & (children < T1)]
            out.append(children)
            parents = children
        times = np.unique(np.sort(np.concatenate(out)))
        truth = {"n_true": truncated, "n_true_within_60s": n * (1 - (1 + S_MAX / tau0) ** (-eps))}
    elif scenario == "smooth_mu_0.3":
        n, base = 0.3, target * 0.7 / 600.0
        rate = lambda t: base * (1.0 + 0.8 * np.sin(2 * np.pi * (t - T0) / 600.0))
        params = HawkesParams(mu=rate, alpha=np.array([n]), betas=np.array([2.0]), mu_max=1.8 * base)
        times, truth = simulate_hawkes(params, T0 - 300.0, T1, rng), {"n_true": n, "n_true_within_60s": n}
    else:
        raise ValueError(scenario)
    return times[times < T0], times[times >= T0], truth


def recovery_cell(cell: tuple[str, int]) -> list[dict]:
    _single_thread()
    scenario, seed = cell
    history, times, truth = simulate_scenario(scenario, seed)
    rows = []
    for name, fit in (
        ("classical_const", fit_fixed_decays(times, history, T0, T1, DEFAULT_BETAS)),
        ("classical_spline", fit_fixed_decays(times, history, T0, T1, DEFAULT_BETAS, "spline", 5)),
        ("neural_const", fit_neural(times, history, T0, T1, "const", S_MAX, STEPS, seed)),
        ("neural_mlp", fit_neural(times, history, T0, T1, "mlp", S_MAX, STEPS, seed)),
    ):
        rows.append({"scenario": scenario, "seed": seed, "fitter": name, "n_events": len(times), "n_hat": fit.n, "converged": fit.converged, "loglik": fit.loglik, **truth})
    return rows


def part_a(seeds: int) -> pd.DataFrame:
    scenarios = ["exp_0.3", "exp_0.6", "exp_0.9", "mixture_0.6", "power_law_0.9", "smooth_mu_0.3"]
    cells = [(s, k) for s in scenarios for k in range(seeds)]
    with ProcessPoolExecutor(initializer=_single_thread) as pool:
        rows = [r for batch in pool.map(recovery_cell, cells) for r in batch]
    fits = pd.DataFrame(rows)
    fits["bias"] = fits["n_hat"] - fits["n_true"]
    fits["bias_within_60s"] = fits["n_hat"] - fits["n_true_within_60s"]
    summary = fits.groupby(["scenario", "fitter"], sort=False).agg(
        n_true=("n_true", "first"), n_true_within_60s=("n_true_within_60s", "first"), mean=("n_hat", "mean"), sd=("n_hat", "std"),
        bias=("bias", "mean"), bias_within_60s=("bias_within_60s", "mean"), converged=("converged", "mean"), loglik=("loglik", "mean"),
    ).reset_index()
    fits.to_csv(RESULTS_DIR / "phase2_recovery_fits.csv", index=False)
    summary.to_csv(RESULTS_DIR / "phase2_recovery.csv", index=False)
    return summary


# Part B ------------------------------------------------------------------------------------------
def window_job(args: tuple) -> dict:
    _single_thread()
    stream, window, baseline, n_boot = args
    times, history = stream.between(window.t0, window.t1), stream.history(window.t0)
    started = time.perf_counter()
    fit = fit_neural(times, history, window.t0, window.t1, baseline, S_MAX, STEPS, SEED)
    gof = gof_neural(fit, times, history, window.t0, window.t1)
    poisson = poisson_loglik(times, window.t0, window.t1, len(times) / window.length)
    row = {
        "fitter": f"neural_{baseline}", "t0": window.t0, "t1": window.t1, "t_mid": window.mid, "n_events": len(times),
        "n": fit.n, "n_within_10s": float(np.interp(10.0, fit.grid_s, fit.big_phi)), "mu_mean": fit.mu_mean, "converged": fit.converged,
        "loglik_per_event": fit.loglik / max(len(times), 1), "gain_over_poisson_per_event": (fit.loglik - poisson) / max(len(times), 1),
        **gof, "fit_seconds": time.perf_counter() - started,
    }
    nxt = Window(window.t1, window.t1 + window.length)
    next_times = stream.between(nxt.t0, nxt.t1)
    if len(next_times) and nxt.t1 <= stream.times[-1]:
        next_history = stream.history(nxt.t0)
        carried = loglik_neural(fit, next_times, next_history, nxt.t0, nxt.t1)
        refit = loglik_neural(fit, next_times, next_history, nxt.t0, nxt.t1, refit_baseline=True)
        row["heldout_gain_over_poisson_per_event"] = (carried - poisson_loglik(next_times, nxt.t0, nxt.t1, len(times) / window.length)) / len(next_times)
        row["heldout_gain_kernel_only_per_event"] = (refit - poisson_loglik(next_times, nxt.t0, nxt.t1, len(next_times) / nxt.length)) / len(next_times)
        row["heldout_n_events"] = len(next_times)
    else:
        row["heldout_gain_over_poisson_per_event"] = row["heldout_gain_kernel_only_per_event"] = np.nan
        row["heldout_n_events"] = 0
    if n_boot > 0 and fit.n < 1.0:
        rng = np.random.default_rng([SEED, int(window.t0), baseline == "mlp"])
        boot, abandoned = [], 0
        budget = max(3 * max(len(times), 1), 5_000)
        # A refit costs O(N^2) because every event sums over the previous 60 s, so a fixed number of
        # replicates would spend hours on the closing windows and seconds on the quiet ones. Replicates
        # are scaled to hold compute roughly level across the day; boot_n records what each window got.
        replicates = int(np.clip(n_boot * (1500.0 / max(len(times), 1)) ** 2, 3, n_boot))
        for b in range(replicates):
            try:
                sim = simulate_neural(fit, window.t0 - 300.0, window.t1, rng, max_events=budget)
            except SimulationTooLarge:
                abandoned += 1
                continue
            sim_hist, sim_win = sim[sim < window.t0], sim[sim >= window.t0]
            if len(sim_win) < 10:
                continue
            refit_b = fit_neural(sim_win, sim_hist, window.t0, window.t1, baseline, S_MAX, STEPS, SEED + b,
                                 init_state=fit.state, polish=False)
            boot.append(refit_b.n)
        boot = np.array(boot)
        row.update({"boot_n": len(boot), "boot_abandoned": abandoned,
                    "boot_mean": boot.mean() if len(boot) else np.nan, "boot_sd": boot.std(ddof=1) if len(boot) > 1 else np.nan,
                    "boot_lo": np.percentile(boot, 2.5) if len(boot) else np.nan, "boot_hi": np.percentile(boot, 97.5) if len(boot) else np.nan})
    row["kernel_phi"] = " ".join(f"{v:.4e}" for v in np.interp(np.log(np.geomspace(1e-5, S_MAX, 14)), np.log(fit.grid_s), fit.phi))
    row["mu_grid"] = " ".join(f"{v:.4f}" for v in fit.mu_grid[::100])
    return row


def part_b(stream: EventStream, session: Session, length: float, stride: float, n_boot: int, tag: str) -> pd.DataFrame:
    windows = rolling_windows(session.open_s, session.close_s, length, stride)
    jobs = [(stream, w, baseline, n_boot) for baseline in ("const", "mlp") for w in windows]
    started = time.perf_counter()
    partial = RESULTS_DIR / f"phase2_rolling_{tag}.partial.csv"
    rows = []
    # Written window by window: these fits take minutes each, so a crash or an interrupt should not
    # throw away the whole run, and progress needs to be visible while it happens. A partial file
    # from an earlier attempt is picked up rather than refitted.
    if partial.exists():
        rows = pd.read_csv(partial).to_dict("records")
        seen = {(r["fitter"], round(r["t0"], 3)) for r in rows}
        jobs = [j for j in jobs if (f"neural_{j[2]}", round(j[1].t0, 3)) not in seen]
        print(f"resuming: {len(rows)} windows already done, {len(jobs)} to go", flush=True)
    with ProcessPoolExecutor(initializer=_single_thread) as pool:
        for done, row in enumerate(pool.map(window_job, jobs, chunksize=1), start=1):
            rows.append(row)
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"  [{done:3d}/{len(jobs)}] {row['fitter']:12s} {session.clock(row['t0'])[:5]} "
                  f"N={row['n_events']:5d} n={row['n']:.3f} ({row['fit_seconds']:.0f}s)", flush=True)
    table = pd.DataFrame(rows).sort_values(["fitter", "t0"]).reset_index(drop=True)
    table.to_csv(RESULTS_DIR / f"phase2_rolling_{tag}.csv", index=False)
    partial.unlink(missing_ok=True)
    print(f"{len(table)} neural window fits in {time.perf_counter() - started:.0f}s")
    return table


def heldout_comparison(neural: pd.DataFrame, classical: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([classical, neural], ignore_index=True)
    cols = ["gain_over_poisson_per_event", "heldout_gain_over_poisson_per_event", "heldout_gain_kernel_only_per_event", "n", "ks_p", "lb_p"]
    wide = both.pivot_table(index="t_mid", columns="fitter", values=cols)
    rows = []
    for a, b in (("neural_const", "mix_const"), ("neural_mlp", "mix_spline"), ("neural_const", "exp_const"), ("neural_mlp", "neural_const")):
        for col in ("heldout_gain_over_poisson_per_event", "heldout_gain_kernel_only_per_event", "gain_over_poisson_per_event"):
            d = (wide[(col, a)] - wide[(col, b)]).dropna()
            rows.append({"comparison": f"{a} minus {b}", "quantity": col, "windows": len(d), "mean_diff": d.mean(), "se_diff": d.std(ddof=1) / np.sqrt(len(d)),
                         "fraction_positive": (d > 0).mean()})
    return pd.DataFrame(rows)


def main(stream_name: str, length: float, stride: float, n_boot: int, seeds: int, reuse_fits: bool = False,
         skip_recovery: bool = False) -> None:
    session = DEFAULT_SESSION
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    tag = f"{stream_name}_w{int(length)}"
    if not reuse_fits and not skip_recovery:
        print("Part A: simulation recovery")
        print(part_a(seeds).round(3).to_string(index=False))

    print("Part B: rolling fits on real data")
    if reuse_fits:
        neural = pd.read_csv(RESULTS_DIR / f"phase2_rolling_{tag}.csv")
        print(f"reusing {len(neural)} saved neural window fits")
    else:
        arr = load_mbp10(sorted(RAW_DIR.glob("*.dbn.zst"))[0])
        neural = part_b(trade_events(arr, session), session, length, stride, n_boot, tag)
    print(neural.groupby("fitter").agg(n_mean=("n", "mean"), n10_mean=("n_within_10s", "mean"), converged=("converged", "mean"),
                                       ks_reject5=("ks_p", lambda p: float(np.mean(p < 0.05))), gain=("gain_over_poisson_per_event", "mean"),
                                       heldout=("heldout_gain_over_poisson_per_event", "mean"), heldout_kernel_only=("heldout_gain_kernel_only_per_event", "mean"),
                                       boot_sd=("boot_sd", "mean"), seconds=("fit_seconds", "mean")).round(4).to_string())
    classical_path = RESULTS_DIR / f"phase1_rolling_{tag}.csv"
    if classical_path.exists():
        classical = pd.read_csv(classical_path)
        comparison = heldout_comparison(neural, classical)
        comparison.to_csv(RESULTS_DIR / f"phase2_heldout_{tag}.csv", index=False)
        print(comparison.round(4).to_string(index=False))
        curves = []
        for name, sub in classical[classical["fitter"].isin(["mix_const", "mix_spline"])].groupby("fitter"):
            curves.append({"name": f"classical {name}", "t": sub["t_mid"].to_numpy(), "n": sub["n"].to_numpy(), "lo": sub["boot_lo"].to_numpy(), "hi": sub["boot_hi"].to_numpy()})
    else:
        curves = []
    for name, sub in neural.groupby("fitter"):
        curve = {"name": name, "t": sub["t_mid"].to_numpy(), "n": sub["n"].to_numpy(), "flagged": (sub["ks_p"] < 0.05).to_numpy()}
        if "boot_lo" in sub:
            curve["lo"], curve["hi"] = sub["boot_lo"].to_numpy(), sub["boot_hi"].to_numpy()
        curves.append(curve)
    fig = figures.branching_ratio_figure(curves, session, f"Branching ratio, classical versus neural kernels, {int(length / 60)}-minute windows; bands: parametric bootstrap 95%")
    print("figure:", figures.save(fig, f"phase2_rolling_{tag}"))
    print("figure:", figures.save(kernel_figure(neural, classical if classical_path.exists() else None, session), f"phase2_kernels_{tag}"))


def kernel_figure(neural: pd.DataFrame, classical: pd.DataFrame | None, session: Session):
    import matplotlib.pyplot as plt

    figures.style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    lags = np.geomspace(1e-5, S_MAX, 14)
    picks = ["10:00", "12:00", "14:00", "15:30"]
    for clock, color in zip(picks, figures.SERIES):
        t_mid = session.seconds(clock) + 300.0
        sub = neural[(neural["fitter"] == "neural_const") & (np.isclose(neural["t_mid"], t_mid))]
        if len(sub):
            phi = np.array([float(v) for v in sub.iloc[0]["kernel_phi"].split()])
            axes[0].loglog(lags, np.maximum(phi, 1e-12), color=color, label=f"neural, window at {clock}")
        if classical is not None:
            row = classical[(classical["fitter"] == "mix_const") & (np.isclose(classical["t_mid"], t_mid))]
            if len(row):
                alpha = np.array([float(v) for v in row.iloc[0]["alpha"].split()])
                phi_c = np.sum(alpha * DEFAULT_BETAS * np.exp(-np.outer(lags, DEFAULT_BETAS)), axis=1)
                axes[0].loglog(lags, np.maximum(phi_c, 1e-12), color=color, linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("lag s (seconds)")
    axes[0].set_ylabel("kernel phi(s)")
    axes[0].set_title("Learned kernels (solid) and classical fixed-grid kernels (dashed)", loc="left")
    axes[0].legend(fontsize=7)
    for clock, color in zip(picks, figures.SERIES):
        t_mid = session.seconds(clock) + 300.0
        sub = neural[(neural["fitter"] == "neural_mlp") & (np.isclose(neural["t_mid"], t_mid))]
        if len(sub):
            mu = np.array([float(v) for v in sub.iloc[0]["mu_grid"].split()])
            axes[1].plot(np.linspace(0, 10, len(mu)), mu, color=color, label=f"window at {clock}")
    axes[1].set_xlabel("minutes into window")
    axes[1].set_ylabel("learned baseline mu(t), events/s")
    axes[1].set_title("MLP baselines", loc="left")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stream", choices=["trades"], default="trades")
    p.add_argument("--length", type=float, default=600.0)
    p.add_argument("--stride", type=float, default=300.0)
    p.add_argument("--boot", type=int, default=20)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--reuse-fits", action="store_true", help="rebuild the comparison and figures from saved neural fits")
    p.add_argument("--skip-recovery", action="store_true", help="skip part A when its results are already on disk")
    a = p.parse_args()
    main(a.stream, a.length, a.stride, a.boot, a.seeds, a.reuse_fits, a.skip_recovery)
