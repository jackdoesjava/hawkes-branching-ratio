from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numba import njit
from scipy import optimize, stats

from .config import DEFAULT_TIMESCALES

DEFAULT_BETAS = 1.0 / np.asarray(DEFAULT_TIMESCALES)
BASELINE_KINDS = ("constant", "piecewise", "spline")
_LOWER = 1e-12


@njit(cache=True)
def _exp_recursion(times: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """s[i, k] = sum_{j < i} exp(-beta_k (t_i - t_j)), the O(N K) Ozaki recursion."""
    n, k = times.shape[0], betas.shape[0]
    s = np.zeros((n, k))
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        for j in range(k):
            s[i, j] = np.exp(-betas[j] * dt) * (s[i - 1, j] + 1.0)
    return s


@dataclass(frozen=True)
class Excitation:
    """Kernel contributions for a fixed decay grid, with pre-window history folded in.

    at_events[i, k]  = sum_{t_j < t_i} beta_k exp(-beta_k (t_i - t_j))
    integral[k]      = int_{t0}^{t1} sum_{t_j < t} beta_k exp(-beta_k (t - t_j)) dt
    cumulative[i, k] = the same integral taken from t0 to t_i
    Multiplying by alpha_k gives the intensity, the compensator, and the compensator at events.
    """

    betas: np.ndarray
    at_events: np.ndarray
    integral: np.ndarray
    cumulative: np.ndarray


def excitation(times: np.ndarray, history: np.ndarray, t0: float, t1: float, betas: np.ndarray) -> Excitation:
    betas = np.atleast_1d(np.asarray(betas, dtype=float))
    times = np.asarray(times, dtype=float)
    history = np.asarray(history, dtype=float)
    if len(times) and (times[0] < t0 or times[-1] >= t1):
        raise ValueError("window events must lie in [t0, t1)")
    if len(history) and history[-1] >= t0:
        raise ValueError("history events must lie before t0")
    # The recursion assumes strictly increasing times; ties or unsorted input silently produce a
    # wrong intensity rather than an error, so they are rejected here.
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("window events must be strictly increasing (aggregate ties before fitting)")
    if len(history) > 1 and not np.all(np.diff(history) > 0):
        raise ValueError("history events must be strictly increasing")
    history = history[history > t0 - 40.0 / betas.min()]  # exp(-40) is far below double precision noise
    h_state = np.exp(-np.outer(t0 - history, betas)).sum(axis=0)
    s_win = _exp_recursion(times, betas) if len(times) else np.zeros((0, len(betas)))
    decay_from_t0 = np.exp(-np.outer(times - t0, betas))
    at_events = betas * (s_win + h_state * decay_from_t0)
    integral = (1.0 - np.exp(-np.outer(t1 - times, betas))).sum(axis=0) + h_state * (1.0 - np.exp(-betas * (t1 - t0)))
    cumulative = (np.arange(len(times))[:, None] - s_win) + h_state * (1.0 - decay_from_t0)
    return Excitation(betas=betas, at_events=at_events, integral=integral, cumulative=cumulative)


@dataclass(frozen=True)
class Baseline:
    """Linear basis for mu(t) on [t0, t1]: values at events, window integrals, cumulative integrals."""

    kind: str
    knots: np.ndarray
    at_events: np.ndarray
    integral: np.ndarray
    cumulative: np.ndarray


def baseline_basis(times: np.ndarray, t0: float, t1: float, kind: str = "constant", n_pieces: int = 1) -> Baseline:
    if kind not in BASELINE_KINDS:
        raise ValueError(f"unknown baseline {kind!r}")
    if kind == "constant":
        n_pieces = 1
    times = np.asarray(times, dtype=float)
    knots = np.linspace(t0, t1, n_pieces + 1)
    width = (t1 - t0) / n_pieces
    if kind in ("constant", "piecewise"):
        piece = np.clip(np.searchsorted(knots, times, side="right") - 1, 0, n_pieces - 1)
        at_events = np.zeros((len(times), n_pieces))
        at_events[np.arange(len(times)), piece] = 1.0
        integral = np.full(n_pieces, width)
        cumulative = np.clip(times[:, None] - knots[None, :-1], 0.0, width)
        return Baseline(kind, knots, at_events, integral, cumulative)
    # Linear hats centred on each knot; the end hats are clipped to the window.
    at_events = np.clip(1.0 - np.abs(times[:, None] - knots[None, :]) / width, 0.0, None)
    integral = np.full(n_pieces + 1, width)
    integral[[0, -1]] = width / 2
    rise = np.clip(times[:, None] - knots[None, :-1], 0.0, width) ** 2 / (2 * width)
    fall = width / 2 - np.clip(knots[None, 1:] - times[:, None], 0.0, width) ** 2 / (2 * width)
    cumulative = np.zeros((len(times), n_pieces + 1))
    cumulative[:, 1:] += rise
    cumulative[:, :-1] += fall
    return Baseline(kind, knots, at_events, integral, cumulative)


@dataclass(frozen=True)
class HawkesFit:
    kind: str  # "exp_mixture" (fixed decays) or "exp" (single decay chosen by profile likelihood)
    betas: np.ndarray
    alpha: np.ndarray
    mu: np.ndarray
    baseline: str
    n_pieces: int
    t0: float
    t1: float
    n_events: int
    loglik: float
    converged: bool
    se_n: float  # observed-information SE of n, conditional on the decay grid; unreliable at boundary solutions
    cov: np.ndarray
    condition_number: float  # of the information matrix over free parameters; large means baseline/kernel collinearity

    @property
    def n(self) -> float:
        return float(np.sum(self.alpha))

    @property
    def mu_mean(self) -> float:
        integral = baseline_basis(np.empty(0), self.t0, self.t1, self.baseline, self.n_pieces).integral
        return float(self.mu @ integral) / (self.t1 - self.t0)

    def kernel(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)[..., None]
        return np.sum(self.alpha * self.betas * np.exp(-self.betas * s), axis=-1)


def _negloglik(theta: np.ndarray, design: np.ndarray, compensator: np.ndarray, offset: np.ndarray | float = 0.0) -> tuple[float, np.ndarray]:
    """Negative log-likelihood up to terms not involving theta; `offset` is a fixed intensity component."""
    lam = design @ theta + offset
    value = -(np.sum(np.log(lam)) - compensator @ theta)
    grad = -(design.T @ (1.0 / lam) - compensator)
    return value, grad


def _projected_newton_polish(theta: np.ndarray, design: np.ndarray, compensator: np.ndarray, offset: np.ndarray | float = 0.0, max_iter: int = 50):
    """Polish a near-optimal point to machine precision with Newton steps on the free coordinates.

    L-BFGS-B stops on relative function decrease, which leaves a weakly identified baseline
    (flat likelihood directions) short of a small gradient; the exact Hessian is tiny and cheap here.
    """
    theta = theta.copy()
    for _ in range(max_iter):
        lam = design @ theta + offset
        value = np.sum(np.log(lam)) - compensator @ theta
        grad = design.T @ (1.0 / lam) - compensator
        free = (theta > 10 * _LOWER) | (grad > 0)
        weighted = design[:, free] / lam[:, None]
        step = np.linalg.lstsq(weighted.T @ weighted, grad[free], rcond=None)[0]
        size = 1.0
        while size > 1e-12:
            candidate = theta.copy()
            candidate[free] = np.maximum(theta[free] + size * step, _LOWER)
            lam_candidate = design @ candidate + offset
            if np.all(lam_candidate > 0) and np.sum(np.log(lam_candidate)) - compensator @ candidate >= value:
                break
            size *= 0.5
        else:
            break
        moved = np.abs(candidate - theta).max()
        theta = candidate
        if moved <= 1e-12 * max(1.0, np.abs(theta).max()):
            break
    return theta


def _maximise_concave(design: np.ndarray, compensator: np.ndarray, theta0: np.ndarray, offset: np.ndarray | float = 0.0):
    """Maximise sum log(X theta + offset) - c.theta over theta >= 0; concave, so the optimum is global.

    Columns are rescaled to unit mean so L-BFGS-B sees a well-conditioned problem; the result is
    then polished with projected Newton steps and convergence judged by the projected gradient.
    Returns theta, the maximised value, the convergence flag, the observed-information covariance
    (zero rows/columns for parameters on the bound) and the information matrix's condition number.
    """
    # Column scale: the larger of the mean over events and the mean over time, so a fast component with
    # no close event pairs (all-zero column over events) is not rescaled into absurdity.
    scale = np.maximum(design.mean(axis=0), compensator / max(design.shape[0], 1))
    scale[scale <= 0] = 1.0
    res = optimize.minimize(
        _negloglik, theta0 * scale, args=(design / scale, compensator / scale, offset), jac=True, method="L-BFGS-B",
        bounds=[(_LOWER, None)] * len(theta0), options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
    )
    theta = _projected_newton_polish(res.x / scale, design, compensator, offset)
    lam = design @ theta + offset
    value = float(np.sum(np.log(lam)) - compensator @ theta)
    grad = design.T @ (1.0 / lam) - compensator
    at_bound = theta <= 10 * _LOWER
    # At the optimum sum_i X_ik / lambda_i = c_k for free parameters, so the gradient is judged relative to c_k.
    projected = np.where(at_bound, np.maximum(grad, 0.0), grad) / np.maximum(compensator, 1e-12)
    converged = bool(np.abs(projected).max() < 1e-6)
    free = ~at_bound
    weighted = design[:, free] / lam[:, None]
    information = weighted.T @ weighted
    cov = np.zeros((len(theta), len(theta)))
    cov[np.ix_(free, free)] = np.linalg.pinv(information)
    return theta, value, converged, cov, _condition_number(information)


def _condition_number(information: np.ndarray) -> float:
    """Condition number of the correlation-normalised information matrix; inf if a parameter is unidentified."""
    if information.size == 0:
        return np.nan
    diagonal = np.sqrt(np.diag(information))
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        return np.inf
    try:
        return float(np.linalg.cond(information / np.outer(diagonal, diagonal)))
    except np.linalg.LinAlgError:
        return np.inf


def fit_fixed_decays(
    times: np.ndarray,
    history: np.ndarray,
    t0: float,
    t1: float,
    betas: np.ndarray = DEFAULT_BETAS,
    baseline: str = "constant",
    n_pieces: int = 1,
) -> HawkesFit:
    """MLE for mu(t) + sum_k alpha_k beta_k e^{-beta_k s} with the decays fixed, so the problem is concave."""
    exc = excitation(times, history, t0, t1, betas)
    base = baseline_basis(times, t0, t1, baseline, n_pieces)
    if len(times) == 0:
        # With no events the likelihood is -integral(lambda), maximised at the origin: an honest
        # degenerate answer rather than the NaNs a log-of-nothing optimisation would produce.
        p, k = base.at_events.shape[1], len(exc.betas)
        return HawkesFit(
            kind="exp_mixture", betas=exc.betas, alpha=np.zeros(k), mu=np.zeros(p), baseline=base.kind,
            n_pieces=n_pieces, t0=t0, t1=t1, n_events=0, loglik=0.0, converged=True, se_n=np.nan,
            cov=np.zeros((p + k, p + k)), condition_number=np.nan,
        )
    design = np.hstack([base.at_events, exc.at_events])
    compensator = np.concatenate([base.integral, exc.integral])
    p, k = base.at_events.shape[1], len(exc.betas)
    rate = max(len(times), 1) / (t1 - t0)
    theta0 = np.concatenate([np.full(p, 0.5 * rate), np.full(k, 0.5 / k)])
    theta, value, converged, cov, condition = _maximise_concave(design, compensator, theta0)
    se_n = float(np.sqrt(max(cov[p:, p:].sum(), 0.0)))
    return HawkesFit(
        kind="exp_mixture", betas=exc.betas, alpha=theta[p:], mu=theta[:p], baseline=base.kind, n_pieces=n_pieces,
        t0=t0, t1=t1, n_events=len(times), loglik=value, converged=converged, se_n=se_n, cov=cov, condition_number=condition,
    )


def default_beta_grid(n: int = 29, fastest: float = 1e-5, slowest: float = 60.0) -> np.ndarray:
    """Search grid for the single-exponential decay, spanning the same timescales as DEFAULT_TIMESCALES.

    A grid starting at 1 ms pins the estimate at its own edge on real trade data, where the dominant
    trade-to-trade lag is tens of microseconds.
    """
    return 1.0 / np.geomspace(fastest, slowest, n)


def fit_exp(
    times: np.ndarray,
    history: np.ndarray,
    t0: float,
    t1: float,
    baseline: str = "constant",
    n_pieces: int = 1,
    beta_grid: np.ndarray | None = None,
) -> HawkesFit:
    """Single exponential with free decay: profile likelihood over a beta grid, then Brent refinement.

    The inner problem in (mu, alpha) is concave for each beta, so the profile is exact and the
    grid search avoids the local optima that plague joint optimisation over (alpha, beta).
    """
    grid = default_beta_grid() if beta_grid is None else np.asarray(beta_grid, dtype=float)

    def profile(log_beta: float) -> float:
        return -fit_fixed_decays(times, history, t0, t1, np.array([np.exp(log_beta)]), baseline, n_pieces).loglik

    log_grid = np.log(np.sort(grid))
    values = np.array([profile(lb) for lb in log_grid])
    best = int(np.argmin(values))
    lo, hi = log_grid[max(best - 1, 0)], log_grid[min(best + 1, len(log_grid) - 1)]
    log_beta = log_grid[best]
    if hi > lo:
        res = optimize.minimize_scalar(profile, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        if res.fun <= values[best]:
            log_beta = float(res.x)
    fit = fit_fixed_decays(times, history, t0, t1, np.array([np.exp(log_beta)]), baseline, n_pieces)
    # A decay pinned at an end of the search grid is not the profile maximum: the likelihood is still
    # rising as beta leaves the grid, so the fit is reported as not converged rather than silently kept.
    at_edge = best in (0, len(log_grid) - 1)
    return replace(fit, kind="exp", converged=fit.converged and not at_edge)


def loglik(fit: HawkesFit, times: np.ndarray, history: np.ndarray, t0: float, t1: float) -> float:
    """Log-likelihood of fitted parameters on an arbitrary window.

    On the window the fit was made on the fitted baseline is used; elsewhere its time average is
    carried forward, since a piecewise baseline has no meaning outside the window it was fitted on.
    """
    exc = excitation(times, history, t0, t1, fit.betas)
    if t0 == fit.t0 and t1 == fit.t1:
        base = baseline_basis(times, t0, t1, fit.baseline, fit.n_pieces)
        mu_events, mu_integral = base.at_events @ fit.mu, base.integral @ fit.mu
    else:
        mu_events, mu_integral = np.full(len(times), fit.mu_mean), fit.mu_mean * (t1 - t0)
    lam = mu_events + exc.at_events @ fit.alpha
    return float(np.sum(np.log(lam)) - mu_integral - exc.integral @ fit.alpha)


def rescaled_intervals(fit: HawkesFit, times: np.ndarray, history: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Compensator increments between consecutive events; i.i.d. Exp(1) if the model is right.

    Only defined on the window the fit was made on: a piecewise or spline baseline is a function of
    position within its own window, so applying it elsewhere would silently rescale the knots.
    """
    if (t0, t1) != (fit.t0, fit.t1):
        raise ValueError("rescaled intervals are only defined on the window the model was fitted on")
    exc = excitation(times, history, t0, t1, fit.betas)
    base = baseline_basis(times, t0, t1, fit.baseline, fit.n_pieces)
    compensator_at_events = base.cumulative @ fit.mu + exc.cumulative @ fit.alpha
    return np.diff(np.concatenate([[0.0], compensator_at_events]))


def ks_exp1(intervals: np.ndarray) -> tuple[float, float]:
    res = stats.kstest(intervals, "expon")
    return float(res.statistic), float(res.pvalue)


def gof_tests(intervals: np.ndarray, lags: int = 10) -> dict[str, float]:
    """Residual tests on rescaled intervals: KS against Exp(1), Ljung-Box autocorrelation, Engle-Russell excess dispersion.

    KS alone is both over-demanding at large N and blind to serial dependence, so all three are reported
    (Lallouache & Challet 2016). The dispersion statistic sqrt(N/8) (s^2 - 1) is N(0,1) under Exp(1).
    """
    n = len(intervals)
    if n < 3:
        return dict.fromkeys(("ks_stat", "ks_p", "lb_stat", "lb_p", "dispersion_z", "dispersion_p"), np.nan)
    ks_stat, ks_p = ks_exp1(intervals)
    centred = intervals - intervals.mean()
    denominator = np.sum(centred**2)
    autocorr = np.array([np.sum(centred[:-k] * centred[k:]) / denominator for k in range(1, lags + 1)])
    lb_stat = n * (n + 2) * np.sum(autocorr**2 / (n - np.arange(1, lags + 1)))
    lb_p = float(stats.chi2.sf(lb_stat, lags))
    dispersion_z = float(np.sqrt(n / 8.0) * (intervals.var(ddof=1) - 1.0))
    return {
        "ks_stat": ks_stat, "ks_p": ks_p, "lb_stat": float(lb_stat), "lb_p": lb_p,
        "dispersion_z": dispersion_z, "dispersion_p": float(2 * stats.norm.sf(abs(dispersion_z))),
    }


def loglik_refit_baseline(fit: HawkesFit, times: np.ndarray, history: np.ndarray, t0: float, t1: float) -> float:
    """Held-out log-likelihood with the kernel frozen and a constant baseline re-estimated on the new window.

    Scores the kernel alone: the baseline level is nuisance, and a piecewise baseline fitted on another
    window carries no information about this one.
    """
    exc = excitation(times, history, t0, t1, fit.betas)
    base = baseline_basis(times, t0, t1, "constant")
    offset = exc.at_events @ fit.alpha
    theta0 = np.array([max(len(times), 1) / (t1 - t0)])
    theta, value, _, _, _ = _maximise_concave(base.at_events, base.integral, theta0, offset)
    return float(value - exc.integral @ fit.alpha)
