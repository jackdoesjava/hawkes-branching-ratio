import numpy as np
import pytest
from scipy import integrate, optimize

from hawkes_fomc.config import DEFAULT_TIMESCALES
from hawkes_fomc.hawkes_classical import (
    DEFAULT_BETAS,
    _negloglik,
    baseline_basis,
    excitation,
    fit_exp,
    fit_fixed_decays,
    gof_tests,
    ks_exp1,
    loglik,
    loglik_refit_baseline,
    rescaled_intervals,
)
from hawkes_fomc.simulate import HawkesParams, simulate_window

T0, T1 = 1000.0, 1600.0  # a 10-minute window, offset from zero to catch absolute/relative time bugs


def _random_events(rng, n=200, n_hist=50):
    history = np.sort(rng.uniform(T0 - 30.0, T0, n_hist))
    times = np.sort(rng.uniform(T0, T1, n))
    return times, history


def test_excitation_matches_brute_force():
    rng = np.random.default_rng(1)
    times, history = _random_events(rng)
    betas = np.array([50.0, 2.0, 0.1])
    exc = excitation(times, history, T0, T1, betas)
    everything = np.concatenate([history, times])
    for i, t in enumerate(times):
        past = everything[everything < t]
        expected = np.sum(betas * np.exp(-np.outer(t - past, betas)), axis=0)
        assert np.allclose(exc.at_events[i], expected, rtol=1e-10, atol=1e-12)


def test_excitation_integrals_match_quadrature():
    rng = np.random.default_rng(2)
    times, history = _random_events(rng, n=40, n_hist=20)
    betas = np.array([5.0, 0.2])
    exc = excitation(times, history, T0, T1, betas)
    everything = np.concatenate([history, times])

    def intensity(t, beta):
        past = everything[everything < t]
        return np.sum(beta * np.exp(-beta * (t - past)))

    for k, beta in enumerate(betas):
        total = 0.0
        edges = np.concatenate([[T0], times, [T1]])
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            piece, _ = integrate.quad(intensity, a, b, args=(beta,), limit=200)
            total += piece
            if i < len(times):
                assert np.isclose(exc.cumulative[i, k], total, rtol=1e-7, atol=1e-9)
        assert np.isclose(exc.integral[k], total, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("kind,n_pieces", [("constant", 1), ("piecewise", 5), ("spline", 5)])
def test_baseline_integrals_match_quadrature(kind, n_pieces):
    probe = np.sort(np.random.default_rng(3).uniform(T0, T1, 30))
    grid = np.union1d(np.linspace(T0, T1, 60001), probe)  # the grid must contain each probe time for the cumulative check
    dense = baseline_basis(grid, T0, T1, kind, n_pieces)
    base = baseline_basis(probe, T0, T1, kind, n_pieces)
    step = 0.01  # trapezoid error on a step function is half a grid step per discontinuity
    for j in range(dense.at_events.shape[1]):
        assert np.isclose(np.trapezoid(dense.at_events[:, j], grid), base.integral[j], atol=2 * step)
        for i, t in enumerate(probe):
            mask = grid <= t
            assert np.isclose(np.trapezoid(dense.at_events[mask, j], grid[mask]), base.cumulative[i, j], atol=2 * step)


def test_spline_basis_is_a_partition_of_unity():
    probe = np.linspace(T0, T1, 1001)
    base = baseline_basis(probe, T0, T1, "spline", 7)
    assert np.allclose(base.at_events.sum(axis=1), 1.0)


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(4)
    times, history = _random_events(rng)
    exc = excitation(times, history, T0, T1, np.array([10.0, 0.5]))
    base = baseline_basis(times, T0, T1, "spline", 3)
    design = np.hstack([base.at_events, exc.at_events])
    comp = np.concatenate([base.integral, exc.integral])
    theta = np.array([0.3, 0.4, 0.2, 0.5, 0.1, 0.2])
    _, grad = _negloglik(theta, design, comp)
    numeric = optimize.approx_fprime(theta, lambda th: _negloglik(th, design, comp)[0], 1e-7)
    assert np.allclose(grad, numeric, rtol=1e-5, atol=1e-6)


def _recover(n_true, betas_true, alpha_true, target_events, seeds, fitter, burn_in=120.0):
    mu = target_events * (1 - n_true) / (T1 - T0)
    params = HawkesParams(mu=mu, alpha=np.asarray(alpha_true), betas=np.asarray(betas_true))
    fits = []
    for s in seeds:
        history, times = simulate_window(params, T0, T1, burn_in, np.random.default_rng(s))
        fits.append(fitter(times, history))
    return fits


@pytest.mark.parametrize("n_true", [0.3, 0.6, 0.9])
def test_recovers_branching_ratio_exponential_kernel(n_true):
    seeds = 40
    fits = _recover(n_true, [2.0], [n_true], 2000, range(seeds), lambda t, h: fit_fixed_decays(t, h, T0, T1, np.array([2.0])))
    estimates = np.array([f.n for f in fits])
    assert all(f.converged for f in fits)
    # tolerance is three standard errors of the seed mean, so a real bias cannot hide behind a loose constant
    standard_error = estimates.std(ddof=1) / np.sqrt(seeds)
    assert abs(estimates.mean() - n_true) < 3 * standard_error, f"bias {estimates.mean() - n_true:+.4f} vs 3 SE {3 * standard_error:.4f}"
    z = (estimates - n_true) / np.array([f.se_n for f in fits])
    assert 0.75 < z.std(ddof=1) < 1.35, f"asymptotic SE miscalibrated: z sd {z.std(ddof=1):.2f}"


def test_default_grid_bias_on_an_off_grid_kernel_stays_within_documented_bounds():
    # The production grid is fixed, so a true timescale between grid points is the normal case; the
    # README quotes this bias, and the test fails if it grows.
    seeds = 40
    fits = _recover(0.6, [1 / 0.5], [0.6], 2000, range(seeds), lambda t, h: fit_fixed_decays(t, h, T0, T1, DEFAULT_BETAS))
    estimates = np.array([f.n for f in fits])
    bias = estimates.mean() - 0.6
    assert 0.0 < bias < 0.05, f"off-grid bias {bias:+.4f} outside the documented 0 to +0.05 band"


@pytest.mark.parametrize("n_true", [0.3, 0.9])
def test_profile_likelihood_recovers_decay_and_n(n_true):
    fits = _recover(n_true, [2.0], [n_true], 2000, range(10), lambda t, h: fit_exp(t, h, T0, T1))
    n_hat = np.array([f.n for f in fits])
    beta_hat = np.array([f.betas[0] for f in fits])
    assert abs(n_hat.mean() - n_true) < 0.04
    assert abs(np.log(beta_hat).mean() - np.log(2.0)) < 0.3


def test_recovers_branching_ratio_mixture_kernel():
    betas = np.array([100.0, 10.0, 1.0])
    alpha = np.array([0.15, 0.25, 0.20])
    grid = np.array([1000.0, 100.0, 10.0, 1.0, 0.1])
    fits = _recover(0.6, betas, alpha, 3000, range(20), lambda t, h: fit_fixed_decays(t, h, T0, T1, grid), burn_in=200.0)
    estimates = np.array([f.n for f in fits])
    assert abs(estimates.mean() - 0.6) < 0.03, f"mean bias {estimates.mean() - 0.6:.3f}"


def test_rescaled_intervals_are_exp1_under_the_truth():
    n_true, beta = 0.7, 2.0
    rejections, pvals = 0, []
    for s in range(30):
        mu = 2000 * (1 - n_true) / (T1 - T0)
        params = HawkesParams(mu=mu, alpha=np.array([n_true]), betas=np.array([beta]))
        history, times = simulate_window(params, T0, T1, 120.0, np.random.default_rng(100 + s))
        fit = fit_fixed_decays(times, history, T0, T1, np.array([beta]))
        tau = rescaled_intervals(fit, times, history, T0, T1)
        assert np.isclose(tau.mean(), 1.0, atol=0.1)
        stat, p = ks_exp1(tau)
        pvals.append(p)
        rejections += p < 0.05
    assert rejections <= 5, f"{rejections}/30 windows rejected at 5%"


def test_nonstationary_baseline_inflates_constant_mu_fit_and_flexible_mu_corrects_it():
    # Hardiman-Bouchaud: slow variation in mu(t) is absorbed by the slowest kernel component under a
    # constant-mu fit. True n = 0.3 with a 0.5 s kernel; mu(t) swings by a factor of 19 over a 5-minute period.
    n_true, seeds = 0.3, 15
    rate = lambda t: 3.0 * (1.0 + 0.9 * np.sin(2 * np.pi * (t - T0) / 300.0))
    params = HawkesParams(mu=rate, alpha=np.array([n_true]), betas=np.array([2.0]), mu_max=5.7)
    grid = np.array([1000.0, 100.0, 10.0, 1.0, 0.1])
    const, flex = [], []
    for s in range(seeds):
        history, times = simulate_window(params, T0, T1, 120.0, np.random.default_rng(200 + s))
        const.append(fit_fixed_decays(times, history, T0, T1, grid, "constant").n)
        flex.append(fit_fixed_decays(times, history, T0, T1, grid, "spline", 10).n)
    const, flex = np.array(const), np.array(flex)
    # the constant-mu fit does not merely inflate, it runs to the critical boundary
    assert const.mean() > 0.9, f"constant-mu fit reached {const.mean():.3f}, expected near-critical"
    assert abs(flex.mean() - n_true) < 3 * flex.std(ddof=1) / np.sqrt(seeds) + 0.02, f"flexible-mu fit gave {flex.mean():.3f}"


def test_flexible_baseline_absorbs_a_slow_kernel_component():
    # The converse failure mode: with a genuinely slow component and a constant true baseline, a
    # dense spline steals the slow mass. This is why n is always reported under both baselines.
    alpha, betas = np.array([0.3, 0.3]), np.array([10.0, 0.1])  # 0.1 s and 10 s, n = 0.6
    params = HawkesParams(mu=2000 * 0.4 / (T1 - T0), alpha=alpha, betas=betas)
    data = [simulate_window(params, T0, T1, 1200.0, np.random.default_rng(400 + s)) for s in range(10)]
    slow = np.asarray(DEFAULT_TIMESCALES) > 1.0
    mass = {}
    for kind, pieces in (("constant", 1), ("spline", 5), ("spline", 20)):
        fits = [fit_fixed_decays(t, h, T0, T1, DEFAULT_BETAS, kind, pieces) for h, t in data]
        mass[(kind, pieces)] = np.mean([f.alpha[slow].sum() for f in fits])
    assert mass[("constant", 1)] > mass[("spline", 5)] > mass[("spline", 20)]
    assert mass[("constant", 1)] < 0.3, "even a constant baseline under-recovers a 10 s component in 10 minutes"
    assert mass[("spline", 20)] < 0.05, "30 s knots should leave the slow component no room at all"


def test_history_removes_cold_start_bias_for_slow_kernel():
    n_true, beta = 0.9, 0.1  # 10 s timescale, clusters span ~100 s, so a cold start matters in a 10-minute window
    mu = 2000 * (1 - n_true) / (T1 - T0)
    params = HawkesParams(mu=mu, alpha=np.array([n_true]), betas=np.array([beta]))
    with_hist, cold = [], []
    for s in range(15):
        history, times = simulate_window(params, T0, T1, 600.0, np.random.default_rng(300 + s))
        with_hist.append(fit_fixed_decays(times, history, T0, T1, np.array([beta])).n)
        cold.append(fit_fixed_decays(times, np.empty(0), T0, T1, np.array([beta])).n)
    assert abs(np.mean(with_hist) - n_true) < abs(np.mean(cold) - n_true)


def test_loglik_on_own_window_equals_fit_loglik():
    rng = np.random.default_rng(5)
    times, history = _random_events(rng)
    fit = fit_fixed_decays(times, history, T0, T1, np.array([5.0, 0.5]), "piecewise", 3)
    assert np.isclose(loglik(fit, times, history, T0, T1), fit.loglik)


def test_rejects_events_outside_window():
    with pytest.raises(ValueError):
        excitation(np.array([T0 - 1.0, T0 + 1.0]), np.empty(0), T0, T1, np.array([1.0]))
    with pytest.raises(ValueError):
        excitation(np.array([T0 + 1.0]), np.array([T0 + 0.5]), T0, T1, np.array([1.0]))


def test_rejects_unsorted_or_tied_event_times():
    # the recursion assumes strict ordering; without a check it returns a wrong intensity silently
    with pytest.raises(ValueError, match="strictly increasing"):
        excitation(np.array([T0 + 1.0, T0 + 3.0, T0 + 2.0]), np.empty(0), T0, T1, np.array([1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        excitation(np.array([T0 + 1.0, T0 + 1.0]), np.empty(0), T0, T1, np.array([1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        excitation(np.array([T0 + 1.0]), np.array([T0 - 1.0, T0 - 2.0]), T0, T1, np.array([1.0]))


def test_empty_window_gives_a_defined_degenerate_fit():
    fit = fit_fixed_decays(np.empty(0), np.empty(0), T0, T1, DEFAULT_BETAS)
    assert fit.n == 0.0 and fit.loglik == 0.0 and fit.converged
    assert np.all(np.isfinite(fit.alpha)) and np.all(np.isfinite(fit.mu))
    assert np.isnan(gof_tests(np.empty(0))["ks_p"])


def test_profile_likelihood_flags_a_decay_pinned_at_the_grid_edge():
    # a true timescale below the search grid cannot be resolved, and the fit must say so
    params = HawkesParams(mu=2000 * 0.4 / (T1 - T0), alpha=np.array([0.6]), betas=np.array([1 / 2e-6]))
    history, times = simulate_window(params, T0, T1, 60.0, np.random.default_rng(500))
    coarse = fit_exp(times, history, T0, T1, beta_grid=1.0 / np.geomspace(1e-3, 60.0, 25))
    assert not coarse.converged and np.isclose(1 / coarse.betas[0], 1e-3, rtol=1e-6)
    inside = fit_exp(times, history, T0, T1)  # the production grid reaches 10 us
    assert inside.betas[0] > coarse.betas[0]


def test_rescaled_intervals_refuse_a_foreign_window():
    params = HawkesParams(mu=2.0, alpha=np.array([0.5]), betas=np.array([2.0]))
    history, times = simulate_window(params, T0, T1, 120.0, np.random.default_rng(9))
    fit = fit_fixed_decays(times, history, T0, T1, np.array([2.0]), "spline", 5)
    with pytest.raises(ValueError, match="window the model was fitted on"):
        rescaled_intervals(fit, times, history, T0, T1 + 1.0)


def test_gof_tests_accept_exp1_and_reject_dependence():
    rng = np.random.default_rng(6)
    rejections = np.zeros(3)
    for _ in range(40):
        g = gof_tests(rng.exponential(1.0, 2000))
        rejections += np.array([g["ks_p"] < 0.05, g["lb_p"] < 0.05, g["dispersion_p"] < 0.05])
    assert np.all(rejections <= 6), f"rejections at 5% over 40 Exp(1) samples: {rejections}"
    # a positively autocorrelated, overdispersed series must be caught by the Ljung-Box and dispersion tests
    z = rng.standard_normal(2000)
    ar = np.empty(2000)
    ar[0] = z[0]
    for i in range(1, 2000):
        ar[i] = 0.6 * ar[i - 1] + z[i]
    bad = np.exp(ar - ar.var() / 2)  # lognormal, mean ~1, dependent and overdispersed
    g = gof_tests(bad)
    assert g["lb_p"] < 1e-6 and g["dispersion_p"] < 1e-6


def test_refit_baseline_heldout_equals_brute_force_maximum_over_mu():
    params = HawkesParams(mu=1.0, alpha=np.array([0.5]), betas=np.array([2.0]))
    history, times = simulate_window(params, T0, T1, 120.0, np.random.default_rng(7))
    fit = fit_fixed_decays(times, history, T0, T1, np.array([2.0]), "piecewise", 3)
    t2, t3 = T1, T1 + 600.0
    history2, times2 = simulate_window(HawkesParams(mu=2.0, alpha=np.array([0.5]), betas=np.array([2.0])), t2, t3, 120.0, np.random.default_rng(8))
    refit = loglik_refit_baseline(fit, times2, history2, t2, t3)
    exc = excitation(times2, history2, t2, t3, fit.betas)
    at_events, integral = exc.at_events @ fit.alpha, exc.integral @ fit.alpha
    # independent solution of the first-order condition sum 1/(mu + excitation) = t3 - t2
    score = lambda m: np.sum(1.0 / (m + at_events)) - (t3 - t2)
    mu_star = optimize.brentq(score, 1e-9, 1e4, xtol=1e-12, rtol=1e-14)
    reference = np.sum(np.log(mu_star + at_events)) - mu_star * (t3 - t2) - integral
    assert abs(refit - reference) < 1e-8 * abs(reference)
    assert refit >= loglik(fit, times2, history2, t2, t3) - 1e-9  # carrying mu forward is one feasible point
