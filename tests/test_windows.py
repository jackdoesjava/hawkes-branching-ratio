import numpy as np

from hawkes_fomc.data import EventStream
from hawkes_fomc.hawkes_classical import fit_fixed_decays
from hawkes_fomc.simulate import HawkesParams, simulate_hawkes
from hawkes_fomc.windows import FITTERS, Window, baseline_function, fit_and_score, parametric_bootstrap, rolling_windows


def test_rolling_windows_cover_the_session_without_overrun():
    windows = rolling_windows(34200.0, 57600.0, 600.0, 300.0)
    assert windows[0].t0 == 34200.0 and windows[-1].t1 <= 57600.0
    assert len(windows) == 77
    assert all(np.isclose(b.t0 - a.t0, 300.0) for a, b in zip(windows[:-1], windows[1:]))


def test_baseline_function_reproduces_fitted_spline_and_piecewise_baselines():
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(1000.0, 1600.0, 3000))
    for kind, pieces in (("spline", 5), ("piecewise", 4), ("constant", 1)):
        fit = fit_fixed_decays(times, np.empty(0), 1000.0, 1600.0, np.array([2.0]), kind, pieces)
        mu, mu_max = baseline_function(fit)
        grid = np.linspace(1000.0, 1599.999, 2001)
        values = mu(grid)
        assert values.max() <= mu_max + 1e-12
        assert np.isclose(np.trapezoid(values, grid), fit.mu_mean * 600.0, rtol=2e-3)
        assert np.isclose(mu(np.array([999.0]))[0], fit.mu_mean)  # outside the window: time average


def test_parametric_bootstrap_brackets_the_truth():
    params = HawkesParams(mu=1.5, alpha=np.array([0.5]), betas=np.array([2.0]))
    times = simulate_hawkes(params, 700.0, 1600.0, np.random.default_rng(1))
    history, window = times[times < 1000.0], times[times >= 1000.0]
    fitter = lambda t, h, t0, t1: fit_fixed_decays(t, h, t0, t1, np.array([2.0]))
    fit = fitter(window, history, 1000.0, 1600.0)
    boot = parametric_bootstrap(fit, fitter, 40, np.random.default_rng(2))
    assert np.isfinite(boot).sum() >= 38
    lo, hi = np.percentile(boot[np.isfinite(boot)], [2.5, 97.5])
    assert lo < 0.5 < hi
    assert np.isclose(boot[np.isfinite(boot)].std(), fit.se_n, rtol=0.6)


def test_fit_and_score_reports_heldout_and_edge_flags():
    params = HawkesParams(mu=2.0, alpha=np.array([0.4]), betas=np.array([5.0]))
    times = simulate_hawkes(params, 33600.0, 36000.0, np.random.default_rng(3))
    stream = EventStream(name="sim", times=times, session=(34200.0, 57600.0))
    row = fit_and_score(stream, Window(34200.0, 34800.0), "mix_const", n_boot=0, seed=0)
    assert row["edge_window"] is True and row["heldout_n_events"] > 0
    assert np.isfinite(row["heldout_gain_kernel_only_per_event"])
    assert row["heldout_gain_kernel_only_per_event"] >= row["heldout_gain_over_poisson_per_event"] - 0.05
    row2 = fit_and_score(stream, Window(35100.0, 35700.0), "mix_const", n_boot=0, seed=0)
    assert row2["edge_window"] is False
    assert set(FITTERS) >= {"exp_const", "exp_spline", "mix_const", "mix_spline"}
