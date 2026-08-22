import numpy as np
import pytest

from hawkes_fomc.simulate import HawkesParams, simulate_hawkes, simulate_window


def test_times_are_strictly_increasing_and_in_range():
    params = HawkesParams(mu=2.0, alpha=np.array([0.3, 0.3]), betas=np.array([10.0, 0.5]))
    times = simulate_hawkes(params, 100.0, 700.0, np.random.default_rng(0))
    assert np.all(np.diff(times) > 0)
    assert times[0] >= 100.0 and times[-1] < 700.0


def test_mean_count_matches_branching_formula():
    # Stationary mean rate is mu / (1 - n); the empty start at t0 - burn_in is negligible after 60 s for a 0.5 s kernel.
    mu, n, horizon = 2.0, 0.6, 600.0
    params = HawkesParams(mu=mu, alpha=np.array([n]), betas=np.array([2.0]))
    counts = [len(simulate_window(params, 0.0, horizon, 60.0, np.random.default_rng(s))[1]) for s in range(40)]
    expected = mu * horizon / (1 - n)
    # Var(N) ~ mu T / (1-n)^3 for a Hawkes count; four standard errors of the 40-seed mean.
    tolerance = 4 * np.sqrt(mu * horizon / (1 - n) ** 3 / 40)
    assert abs(np.mean(counts) - expected) < tolerance


def test_time_varying_baseline_is_thinned_correctly():
    rate = lambda t: 2.0 + 1.5 * np.sin(2 * np.pi * t / 300.0)
    params = HawkesParams(mu=rate, alpha=np.array([0.0]), betas=np.array([1.0]), mu_max=3.5)
    times = np.concatenate([simulate_hawkes(params, 0.0, 600.0, np.random.default_rng(s)) for s in range(20)])
    counts, _ = np.histogram(times, bins=np.linspace(0, 600, 7))
    grid = np.linspace(0, 600, 6001)
    expected = 20 * np.array([np.trapezoid(rate(grid[(grid >= a) & (grid <= b)]), grid[(grid >= a) & (grid <= b)]) for a, b in zip(np.linspace(0, 600, 7)[:-1], np.linspace(0, 600, 7)[1:])])
    assert np.all(np.abs(counts - expected) < 4 * np.sqrt(expected))


def test_supercritical_is_rejected():
    with pytest.raises(ValueError):
        simulate_hawkes(HawkesParams(mu=1.0, alpha=np.array([1.0]), betas=np.array([1.0])), 0.0, 10.0, np.random.default_rng(0))
