from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import EventStream
from .hawkes_classical import (
    DEFAULT_BETAS,
    HawkesFit,
    baseline_basis,
    fit_exp,
    fit_fixed_decays,
    gof_tests,
    loglik,
    loglik_refit_baseline,
    rescaled_intervals,
)
from .simulate import HawkesParams, simulate_hawkes

EDGE_MARGIN = 15 * 60.0  # windows touching the first/last 15 minutes of the session are flagged, not dropped

Fitter = Callable[[np.ndarray, np.ndarray, float, float], HawkesFit]

FITTERS: dict[str, Fitter] = {
    "exp_const": lambda t, h, t0, t1: fit_exp(t, h, t0, t1),
    "exp_spline": lambda t, h, t0, t1: fit_exp(t, h, t0, t1, "spline", 5),
    "mix_const": lambda t, h, t0, t1: fit_fixed_decays(t, h, t0, t1, DEFAULT_BETAS),
    "mix_spline": lambda t, h, t0, t1: fit_fixed_decays(t, h, t0, t1, DEFAULT_BETAS, "spline", 5),
}


@dataclass(frozen=True)
class Window:
    t0: float
    t1: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.t0 + self.t1)

    @property
    def length(self) -> float:
        return self.t1 - self.t0


def rolling_windows(t_start: float, t_end: float, length: float, stride: float) -> list[Window]:
    starts = np.arange(t_start, t_end - length + 1e-9, stride)
    return [Window(float(s), float(s + length)) for s in starts]


def baseline_function(fit: HawkesFit) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """The fitted mu(t) as a callable on the fit window (time-averaged value outside it), with its maximum."""
    knots = baseline_basis(np.empty(0), fit.t0, fit.t1, fit.baseline, fit.n_pieces).knots

    def mu(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if fit.baseline == "spline":
            inside = np.interp(t, knots, fit.mu)  # hats are a partition of unity, so knot values are the coefficients
        else:
            piece = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, len(fit.mu) - 1)
            inside = fit.mu[piece]
        return np.where((t >= fit.t0) & (t < fit.t1), inside, fit.mu_mean)

    return mu, float(max(fit.mu.max(), fit.mu_mean))


def parametric_bootstrap(fit: HawkesFit, fitter: Fitter, n_boot: int, rng: np.random.Generator, burn_in: float = 300.0) -> np.ndarray:
    """Branching ratios refitted on n_boot simulations from the fitted model; NaN where a refit fails."""
    if fit.n >= 1.0:
        return np.full(n_boot, np.nan)
    mu, mu_max = baseline_function(fit)
    params = HawkesParams(mu=mu, alpha=fit.alpha, betas=fit.betas, mu_max=mu_max)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        times = simulate_hawkes(params, fit.t0 - burn_in, fit.t1, rng)
        history, window = times[times < fit.t0], times[times >= fit.t0]
        if len(window) < 10:
            continue
        refit = fitter(window, history, fit.t0, fit.t1)
        out[b] = refit.n if refit.converged else np.nan
    return out


def poisson_loglik(times: np.ndarray, t0: float, t1: float, rate: float) -> float:
    return float(len(times) * np.log(rate) - rate * (t1 - t0))


def fit_and_score(
    stream: EventStream, window: Window, name: str, n_boot: int, seed: int, next_length: float | None = None
) -> dict:
    """Fit one model on one window; score in-sample GOF, held-out gain over Poisson on the next window, bootstrap n."""
    fitter = FITTERS[name]
    times, history = stream.between(window.t0, window.t1), stream.history(window.t0)
    fit = fitter(times, history, window.t0, window.t1)
    tau = rescaled_intervals(fit, times, history, window.t0, window.t1)
    session_start, session_end = stream.session if stream.session else (-np.inf, np.inf)
    row = {
        "fitter": name, "t0": window.t0, "t1": window.t1, "t_mid": window.mid, "n_events": len(times),
        "n": fit.n, "se_n": fit.se_n, "mu_mean": fit.mu_mean, "converged": fit.converged,
        "condition_number": fit.condition_number,
        "beta_exp": fit.betas[0] if fit.kind == "exp" else np.nan,
        "alpha": " ".join(f"{a:.5f}" for a in fit.alpha),
        "timescales": " ".join(f"{1 / b:.2e}" for b in fit.betas),
        "edge_window": bool(window.t0 < session_start + EDGE_MARGIN or window.t1 > session_end - EDGE_MARGIN),
        "loglik_per_event": fit.loglik / max(len(times), 1),
        "gain_over_poisson_per_event": (fit.loglik - poisson_loglik(times, window.t0, window.t1, len(times) / window.length)) / max(len(times), 1),
        **gof_tests(tau),
    }
    next_length = window.length if next_length is None else next_length
    next_window = Window(window.t1, window.t1 + next_length)
    next_times = stream.between(next_window.t0, next_window.t1)
    # The stream spans the whole ITCH file, not just the session, so the last window would otherwise
    # be scored on post-close data where the arrival rate is an order of magnitude lower.
    if len(next_times) and next_window.t1 <= min(stream.times[-1], session_end):
        next_history = stream.history(next_window.t0)
        carried = loglik(fit, next_times, next_history, next_window.t0, next_window.t1)
        refit = loglik_refit_baseline(fit, next_times, next_history, next_window.t0, next_window.t1)
        poisson_carried = poisson_loglik(next_times, next_window.t0, next_window.t1, len(times) / window.length)
        poisson_refit = poisson_loglik(next_times, next_window.t0, next_window.t1, len(next_times) / next_length)
        row["heldout_gain_over_poisson_per_event"] = (carried - poisson_carried) / len(next_times)
        row["heldout_gain_kernel_only_per_event"] = (refit - poisson_refit) / len(next_times)
        row["heldout_n_events"] = len(next_times)
    else:
        row["heldout_gain_over_poisson_per_event"] = row["heldout_gain_kernel_only_per_event"] = np.nan
        row["heldout_n_events"] = 0
    if n_boot > 0:
        boot = parametric_bootstrap(fit, fitter, n_boot, np.random.default_rng([seed, int(window.t0), list(FITTERS).index(name)]))
        ok = boot[np.isfinite(boot)]
        row.update({
            "boot_n": len(ok), "boot_mean": ok.mean() if len(ok) else np.nan, "boot_sd": ok.std(ddof=1) if len(ok) > 1 else np.nan,
            "boot_lo": np.percentile(ok, 2.5) if len(ok) else np.nan, "boot_hi": np.percentile(ok, 97.5) if len(ok) else np.nan,
        })
    return row


def _job(args: tuple) -> dict:
    return fit_and_score(*args)


def run_rolling(stream: EventStream, windows: list[Window], fitters: list[str], n_boot: int, seed: int, workers: int | None = None) -> pd.DataFrame:
    jobs = [(stream, w, name, n_boot, seed) for name in fitters for w in windows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_job, jobs, chunksize=1))
    return pd.DataFrame(rows).sort_values(["fitter", "t0"]).reset_index(drop=True)
