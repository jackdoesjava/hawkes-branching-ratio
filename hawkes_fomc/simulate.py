from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

RateFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class HawkesParams:
    """lambda(t) = mu(t) + sum_k alpha_k beta_k exp(-beta_k (t - t_i)); branching ratio n = sum_k alpha_k."""

    mu: float | RateFn
    alpha: np.ndarray
    betas: np.ndarray
    mu_max: float | None = None  # required upper bound when mu is a function

    @property
    def n(self) -> float:
        return float(np.sum(self.alpha))

    def kernel(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)[..., None]
        return np.sum(self.alpha * self.betas * np.exp(-self.betas * s), axis=-1)


def _homogeneous_poisson(rate: float, t_start: float, t_end: float, rng: np.random.Generator) -> np.ndarray:
    count = rng.poisson(rate * (t_end - t_start))
    return np.sort(rng.uniform(t_start, t_end, count))


def _immigrants(params: HawkesParams, t_start: float, t_end: float, rng: np.random.Generator) -> np.ndarray:
    if not callable(params.mu):
        return _homogeneous_poisson(float(params.mu), t_start, t_end, rng)
    if params.mu_max is None:
        raise ValueError("mu_max is required for a time-varying baseline")
    candidates = _homogeneous_poisson(params.mu_max, t_start, t_end, rng)
    rate = params.mu(candidates)
    if np.any(rate > params.mu_max + 1e-12):
        raise ValueError("mu(t) exceeds mu_max; thinning would be biased")
    return candidates[rng.uniform(0.0, params.mu_max, len(candidates)) < rate]


def simulate_hawkes(params: HawkesParams, t_start: float, t_end: float, rng: np.random.Generator) -> np.ndarray:
    """Exact cluster (branching) simulation on [t_start, t_end).

    Immigrants arrive at rate mu(t); every event has Poisson(n) children whose delays are drawn
    from the normalised kernel, a mixture of exponentials. The process starts empty at t_start,
    so callers wanting a stationary window should simulate from an earlier time and treat the
    early part as history.
    """
    n = params.n
    if n >= 1.0:
        raise ValueError("cluster simulation requires a subcritical process (n < 1)")
    parents = _immigrants(params, t_start, t_end, rng)
    generations = [parents]
    weights = params.alpha / n if n > 0 else None
    while len(parents) and n > 0:
        n_children = rng.poisson(n, size=len(parents))
        parent_times = np.repeat(parents, n_children)
        component = rng.choice(len(params.betas), size=len(parent_times), p=weights)
        children = parent_times + rng.exponential(1.0 / params.betas[component])
        children = children[children < t_end]
        generations.append(children)
        parents = children
    times = np.sort(np.concatenate(generations))
    return np.unique(times)  # continuous times tie with probability zero; unique guards the invariant


def simulate_window(
    params: HawkesParams, t0: float, t1: float, burn_in: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate on [t0 - burn_in, t1) and split into (history, window events)."""
    times = simulate_hawkes(params, t0 - burn_in, t1, rng)
    return times[times < t0], times[times >= t0]
