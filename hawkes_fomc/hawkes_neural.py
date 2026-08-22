from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .hawkes_classical import _maximise_concave, gof_tests

torch.set_default_dtype(torch.float64)


def _mlp(width: int, depth: int = 2) -> nn.Sequential:
    layers, d = [], 1
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class InterpIndex:
    """Log-space interpolation indices and weights for a fixed set of lags."""

    s: torch.Tensor
    idx: torch.Tensor
    w: torch.Tensor
    below: torch.Tensor
    above: torch.Tensor


class NeuralHawkes(nn.Module):
    """lambda(t) = mu_psi(t) + sum_{t_i < t} phi_theta(t - t_i) with an identifiable branching ratio.

    The kernel is the tail integral phi(s) = int_s^S g(u) du of a non-negative g = softplus(MLP(log u)),
    so it is positive and non-increasing by construction and n = int_0^S phi(s) ds = int_0^S u g(u) du is
    well defined. g lives on a log-spaced grid (200 nodes per decade) and all integrals are trapezoid
    quadrature on that grid; against the closed-form exponential kernel this gives n to 1e-4 relative
    and the interpolated phi to 0.5% (tests/test_neural.py). The baseline is either a positive
    constant or softplus(MLP(time)).
    """

    def __init__(self, s_min: float = 1e-6, s_max: float = 60.0, nodes_per_decade: int = 200, width: int = 32, baseline: str = "mlp"):
        super().__init__()
        n_grid = int(round(nodes_per_decade * math.log10(s_max / s_min))) + 1
        self.register_buffer("log_u", torch.linspace(math.log(s_min), math.log(s_max), n_grid))
        self.register_buffer("u", self.log_u.exp())
        self.s_min, self.s_max, self.baseline_kind = s_min, s_max, baseline
        self.kernel_net = _mlp(width)
        self.log_scale = nn.Parameter(torch.tensor(math.log(0.5)))
        self.baseline_net = _mlp(16) if baseline == "mlp" else None
        self.baseline_raw = nn.Parameter(torch.tensor(0.0))
        self.log_rate = 0.0  # set by fit(): log of the window's event rate, centres the baseline parameterisation

    # kernel on the grid ------------------------------------------------------------------
    def g_grid(self) -> torch.Tensor:
        x = (self.log_u - self.log_u.mean()) / (self.log_u.std() + 1e-12)
        # With g = h(u) / u the branching ratio is int h d(log u): normalising by the log-range makes
        # exp(log_scale) the branching ratio of a log-uniform kernel, so training starts subcritical.
        log_range = float(self.log_u[-1] - self.log_u[0])
        return torch.nn.functional.softplus(self.kernel_net(x[:, None])[:, 0]) * self.log_scale.exp() / (self.u * log_range)

    def phi_grid(self) -> torch.Tensor:
        g = self.g_grid()
        segment = 0.5 * (g[1:] + g[:-1]) * (self.u[1:] - self.u[:-1])
        tail = torch.flip(torch.cumsum(torch.flip(segment, [0]), 0), [0])
        return torch.cat([tail, tail.new_zeros(1)])

    def big_phi_grid(self) -> torch.Tensor:
        """Phi(u_j) = int_0^{u_j} phi; phi is taken flat below s_min (a negligible 1e-6 s)."""
        phi = self.phi_grid()
        segment = 0.5 * (phi[1:] + phi[:-1]) * (self.u[1:] - self.u[:-1])
        return torch.cat([phi[:1] * self.u[:1], phi[0] * self.u[0] + torch.cumsum(segment, 0)])

    def branching_ratio(self) -> torch.Tensor:
        return self.big_phi_grid()[-1]

    def index(self, s: torch.Tensor) -> "InterpIndex":
        """Interpolation indices and weights in log s; they depend only on the lags, so callers cache them."""
        log_s = torch.log(s.clamp_min(1e-300))
        idx = torch.bucketize(log_s, self.log_u).clamp(1, len(self.log_u) - 1)
        lo, hi = self.log_u[idx - 1], self.log_u[idx]
        w = ((log_s - lo) / (hi - lo)).clamp(0.0, 1.0)
        return InterpIndex(s=s, idx=idx, w=w, below=s < self.s_min, above=s >= self.s_max)

    @staticmethod
    def _interp(values: torch.Tensor, ix: "InterpIndex", below: torch.Tensor, above: torch.Tensor) -> torch.Tensor:
        out = values[ix.idx - 1] * (1 - ix.w) + values[ix.idx] * ix.w
        out = torch.where(ix.below, below.expand_as(out), out)
        return torch.where(ix.above, above.expand_as(out), out)

    def phi_at(self, ix: "InterpIndex") -> torch.Tensor:
        grid = self.phi_grid()
        return self._interp(grid, ix, grid[0], grid.new_zeros(()))

    def big_phi_at(self, ix: "InterpIndex") -> torch.Tensor:
        grid = self.big_phi_grid()
        below = grid[0] / self.u[0] * ix.s.clamp(0.0, self.s_min)  # linear below s_min, exact for a flat phi
        return torch.where(ix.below, below, self._interp(grid, ix, grid[0], grid[-1]))

    def phi(self, s: torch.Tensor) -> torch.Tensor:
        return self.phi_at(self.index(s))

    def big_phi(self, x: torch.Tensor) -> torch.Tensor:
        return self.big_phi_at(self.index(x))

    # baseline ------------------------------------------------------------------------------
    def mu(self, tau: torch.Tensor) -> torch.Tensor:
        """Baseline as a function of normalised window time tau in [0, 1]; constant outside by the caller."""
        if self.baseline_net is None:
            return torch.nn.functional.softplus(self.baseline_raw + self.log_rate).expand_as(tau)
        return torch.nn.functional.softplus(self.baseline_net(2 * tau[:, None] - 1)[:, 0] + self.baseline_raw + self.log_rate)


@dataclass(frozen=True)
class WindowTensors:
    """Everything the log-likelihood needs, precomputed once per window."""

    t0: float
    t1: float
    n_events: int
    lags: InterpIndex  # (P,) t_i - t_j for pairs with 0 < lag < s_max, j any earlier event (history included)
    rows: torch.Tensor  # (P,) index i of the window event each lag excites
    comp_lo: InterpIndex  # (M,) clip(t0 - t_j, 0, s_max) for every event j before t1
    comp_hi: InterpIndex  # (M,) min(t1 - t_j, s_max)
    tau_events: torch.Tensor  # (N,) normalised time of window events
    tau_quad: torch.Tensor  # (Q,) quadrature nodes on [0, 1] for the baseline integral


def window_tensors(times: np.ndarray, history: np.ndarray, t0: float, t1: float, s_max: float, n_quad: int = 1201) -> WindowTensors:
    times = np.asarray(times, dtype=float)
    history = np.asarray(history, dtype=float)
    history = history[history > t0 - s_max]
    everything = np.concatenate([history, times])
    first = np.searchsorted(everything, times - s_max, side="right")
    last = np.searchsorted(everything, times, side="left")
    rows = np.repeat(np.arange(len(times)), last - first)
    cols = np.concatenate([np.arange(a, b) for a, b in zip(first, last)]) if len(times) else np.empty(0, dtype=int)
    lags = times[rows] - everything[cols]
    indexer = NeuralHawkes(s_max=s_max, baseline="const")  # only its grid is used
    return WindowTensors(
        t0=t0, t1=t1, n_events=len(times),
        lags=indexer.index(torch.as_tensor(lags)), rows=torch.as_tensor(rows),
        comp_lo=indexer.index(torch.as_tensor(np.clip(t0 - everything, 0.0, s_max))),
        comp_hi=indexer.index(torch.as_tensor(np.minimum(t1 - everything, s_max))),
        tau_events=torch.as_tensor((times - t0) / (t1 - t0)), tau_quad=torch.linspace(0.0, 1.0, n_quad),
    )


def log_likelihood(model: NeuralHawkes, w: WindowTensors) -> torch.Tensor:
    excitation = torch.zeros(w.n_events).index_add_(0, w.rows, model.phi_at(w.lags))
    intensity = model.mu(w.tau_events) + excitation
    mu_quad = model.mu(w.tau_quad)
    baseline_integral = torch.trapezoid(mu_quad, w.tau_quad) * (w.t1 - w.t0)
    kernel_integral = (model.big_phi_at(w.comp_hi) - model.big_phi_at(w.comp_lo)).sum()
    return torch.log(intensity).sum() - baseline_integral - kernel_integral


@dataclass(frozen=True)
class NeuralFit:
    n: float
    t0: float
    t1: float
    n_events: int
    loglik: float
    converged: bool
    baseline: str
    s_max: float
    grid_s: np.ndarray  # kernel support grid
    phi: np.ndarray  # kernel on the grid
    big_phi: np.ndarray  # its cumulative integral on the grid
    mu_grid: np.ndarray  # baseline on 1201 uniform nodes across the window
    state: dict  # model state_dict for refits and held-out scoring

    @property
    def mu_mean(self) -> float:
        return float(np.trapezoid(self.mu_grid, np.linspace(0.0, 1.0, len(self.mu_grid))))

    def model(self) -> NeuralHawkes:
        m = NeuralHawkes(s_max=self.s_max, baseline=self.baseline)
        m.load_state_dict({k: v for k, v in self.state.items() if not k.startswith("_")})
        m.log_rate = float(self.state["_log_rate"]) if "_log_rate" in self.state else 0.0
        return m


def fit_neural(
    times: np.ndarray,
    history: np.ndarray,
    t0: float,
    t1: float,
    baseline: str = "mlp",
    s_max: float = 60.0,
    steps: int = 800,
    seed: int = 0,
    init_state: dict | None = None,
    polish: bool = True,
) -> NeuralFit:
    """Maximum-likelihood training with Adam, then an L-BFGS polish.

    The polish costs about as much as the whole Adam run, so bootstrap refits (which are
    warm-started and only need a point estimate of n) turn it off.
    """
    torch.manual_seed(seed)
    model = NeuralHawkes(s_max=s_max, baseline=baseline)
    model.log_rate = math.log(max(len(times), 1) / (t1 - t0))
    if init_state is not None:
        model.load_state_dict({k: v for k, v in init_state.items() if not k.startswith("_")})
        steps = steps // 4  # warm starts need far fewer steps
    w = window_tensors(times, history, t0, t1, s_max)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps, eta_min=1e-4)
    scale = max(len(times), 1)
    history_loss = []
    for _ in range(steps):
        optimiser.zero_grad()
        loss = -log_likelihood(model, w) / scale
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimiser.step()
        scheduler.step()
        history_loss.append(loss.item())
    before = history_loss[-1] if history_loss else float("inf")
    if polish:
        # Full-batch problem with ~1e3 parameters, so a quasi-Newton pass converges where Adam plateaus.
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, tolerance_grad=1e-9,
                                  tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss = -log_likelihood(model, w) / scale
            loss.backward()
            return loss

        lbfgs.step(closure)
    with torch.no_grad():
        final = float(log_likelihood(model, w))
        converged = bool(np.isfinite(final)) and (-final / scale) <= before + 1e-9
        state = {k: v.clone() for k, v in model.state_dict().items()}
        state["_log_rate"] = torch.tensor(model.log_rate)
        return NeuralFit(
            n=float(model.branching_ratio()), t0=t0, t1=t1, n_events=len(times), loglik=final, converged=converged,
            baseline=baseline, s_max=s_max, grid_s=model.u.numpy().copy(), phi=model.phi_grid().numpy().copy(),
            big_phi=model.big_phi_grid().numpy().copy(), mu_grid=model.mu(torch.linspace(0, 1, 1201)).numpy().copy(), state=state,
        )


def _excitation_numpy(fit: NeuralFit, w: WindowTensors) -> tuple[np.ndarray, float]:
    model = fit.model()
    with torch.no_grad():
        excitation = torch.zeros(w.n_events).index_add_(0, w.rows, model.phi_at(w.lags)).numpy()
        kernel_integral = float((model.big_phi_at(w.comp_hi) - model.big_phi_at(w.comp_lo)).sum())
    return excitation, kernel_integral


def loglik_neural(fit: NeuralFit, times: np.ndarray, history: np.ndarray, t0: float, t1: float, refit_baseline: bool = False) -> float:
    """Held-out log-likelihood: the kernel is frozen; the baseline is carried as its window mean or re-estimated as a constant."""
    w = window_tensors(times, history, t0, t1, fit.s_max)
    excitation, kernel_integral = _excitation_numpy(fit, w)
    if refit_baseline:
        ones = np.ones((len(times), 1))
        _, value, _, _, _ = _maximise_concave(ones, np.array([t1 - t0]), np.array([max(len(times), 1) / (t1 - t0)]), excitation)
        return float(value - kernel_integral)
    mu = fit.mu_mean
    return float(np.sum(np.log(mu + excitation)) - mu * (t1 - t0) - kernel_integral)


def rescaled_intervals_neural(fit: NeuralFit, times: np.ndarray, history: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Compensator increments between consecutive events, with the kernel's finite support handled exactly."""
    model = fit.model()
    times = np.asarray(times, dtype=float)
    history = np.asarray(history, dtype=float)
    history = history[history > t0 - fit.s_max]
    everything = np.concatenate([history, times])
    n_total = fit.n
    with torch.no_grad():
        phi_t0 = model.big_phi(torch.as_tensor(np.clip(t0 - everything, 0.0, fit.s_max))).numpy()  # Phi(clip(t0 - t_j))
        tau = torch.linspace(0, 1, 1201)
        mu_cum = torch.cumulative_trapezoid(model.mu(tau), tau).numpy() * (t1 - t0)
    mu_cum = np.concatenate([[0.0], mu_cum])
    comp_mu = np.interp((times - t0) / (t1 - t0), np.linspace(0, 1, 1201), mu_cum)
    # events j whose whole kernel lies before t_i contribute n - Phi(clip(t0 - t_j)); nearer ones Phi(t_i - t_j) - Phi(clip(t0 - t_j))
    residual = n_total - phi_t0
    residual_cum = np.concatenate([[0.0], np.cumsum(residual)])
    far = np.searchsorted(everything, times - fit.s_max, side="right")
    near_end = np.searchsorted(everything, times, side="left")
    comp_kernel = residual_cum[far].copy()
    rows = np.repeat(np.arange(len(times)), near_end - far)
    cols = np.concatenate([np.arange(a, b) for a, b in zip(far, near_end)]) if len(times) else np.empty(0, dtype=int)
    with torch.no_grad():
        near = model.big_phi(torch.as_tensor(times[rows] - everything[cols])).numpy() - phi_t0[cols]
    np.add.at(comp_kernel, rows, near)
    return np.diff(np.concatenate([[0.0], comp_mu + comp_kernel]))


class SimulationTooLarge(RuntimeError):
    """A replicate whose cascade outgrew the budget; near-critical fits can generate these without bound."""


def simulate_neural(fit: NeuralFit, t_start: float, t_end: float, rng: np.random.Generator,
                    max_events: int | None = None) -> np.ndarray:
    """Cluster simulation from a fitted neural model: mu(t) by thinning, child delays by inverse CDF on the kernel grid.

    Cost is quadratic in the event count, so a replicate drawn from a fit with n near 1 can be
    arbitrarily more expensive than the window it came from. `max_events` abandons those instead of
    letting one bootstrap replicate run for hours; the caller counts how many were abandoned.
    """
    n = fit.n
    if n >= 1.0:
        raise ValueError("fitted model is supercritical; cannot simulate")
    tau_grid = np.linspace(0.0, 1.0, len(fit.mu_grid))
    mu_max = float(fit.mu_grid.max())

    def mu(t: np.ndarray) -> np.ndarray:
        inside = (t >= fit.t0) & (t < fit.t1)
        return np.where(inside, np.interp((t - fit.t0) / (fit.t1 - fit.t0), tau_grid, fit.mu_grid), fit.mu_mean)

    candidates = np.sort(rng.uniform(t_start, t_end, rng.poisson(max(mu_max, fit.mu_mean) * (t_end - t_start))))
    parents = candidates[rng.uniform(0.0, max(mu_max, fit.mu_mean), len(candidates)) < mu(candidates)]
    cdf = fit.big_phi / n
    generations = [parents]
    total = len(parents)
    while len(parents):
        kids = rng.poisson(n, size=len(parents))
        parent_times = np.repeat(parents, kids)
        delays = np.interp(rng.uniform(size=len(parent_times)), cdf, fit.grid_s)
        children = parent_times + delays
        children = children[children < t_end]
        generations.append(children)
        total += len(children)
        if max_events is not None and total > max_events:
            raise SimulationTooLarge(f"{total} events exceeds the budget of {max_events}")
        parents = children
    return np.unique(np.sort(np.concatenate(generations)))


def gof_neural(fit: NeuralFit, times: np.ndarray, history: np.ndarray, t0: float, t1: float) -> dict[str, float]:
    return gof_tests(rescaled_intervals_neural(fit, times, history, t0, t1))
