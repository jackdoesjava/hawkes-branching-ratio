import numpy as np
import torch

from hawkes_fomc.hawkes_classical import baseline_basis, excitation, fit_fixed_decays, rescaled_intervals
from hawkes_fomc.hawkes_neural import NeuralHawkes, fit_neural, log_likelihood, loglik_neural, rescaled_intervals_neural, window_tensors
from hawkes_fomc.simulate import HawkesParams, simulate_window

T0, T1 = 1000.0, 1600.0
N_TRUE, BETA, S_MAX = 0.6, 2.0, 60.0


def _exponential_model(mu: float) -> NeuralHawkes:
    """A NeuralHawkes whose g on the grid is that of phi(s) = n beta exp(-beta s), i.e. g = n beta^2 exp(-beta s)."""
    model = NeuralHawkes(s_max=S_MAX, baseline="const")
    u = model.u
    model.g_grid = lambda: N_TRUE * BETA**2 * torch.exp(-BETA * u)
    model.log_rate = 0.0
    with torch.no_grad():
        model.baseline_raw.fill_(float(np.log(np.expm1(mu))))  # softplus^-1
    return model


def test_quadrature_matches_closed_form_exponential_kernel():
    model = _exponential_model(1.0)
    s = model.u.numpy()
    phi_exact = N_TRUE * BETA * (np.exp(-BETA * s) - np.exp(-BETA * S_MAX))
    assert np.allclose(model.phi_grid().numpy(), phi_exact, rtol=2e-3, atol=1e-9)
    big_phi_exact = N_TRUE * (1 - np.exp(-BETA * s)) - N_TRUE * BETA * s * np.exp(-BETA * S_MAX)
    assert np.allclose(model.big_phi_grid().numpy(), big_phi_exact, rtol=2e-3, atol=1e-6)
    n_exact = N_TRUE * (1 - np.exp(-BETA * S_MAX)) - N_TRUE * BETA * S_MAX * np.exp(-BETA * S_MAX)
    assert abs(float(model.branching_ratio()) - n_exact) < 1e-3 * n_exact
    probe = torch.tensor([1e-7, 3e-4, 0.01, 0.37, 2.5, 59.0, 80.0])
    assert np.allclose(model.phi(probe).numpy(), np.where(probe.numpy() >= S_MAX, 0.0, N_TRUE * BETA * (np.exp(-BETA * np.maximum(probe.numpy(), 1e-6)) - np.exp(-BETA * S_MAX))), rtol=3e-3, atol=1e-9)


def test_loglik_and_compensator_match_classical_for_exponential_kernel():
    mu = 1.2
    params = HawkesParams(mu=mu, alpha=np.array([N_TRUE]), betas=np.array([BETA]))
    history, times = simulate_window(params, T0, T1, 120.0, np.random.default_rng(11))
    model = _exponential_model(mu)
    w = window_tensors(times, history, T0, T1, S_MAX)
    with torch.no_grad():
        neural = float(log_likelihood(model, w))
    exc = excitation(times, history, T0, T1, np.array([BETA]))
    base = baseline_basis(times, T0, T1, "constant")
    classical = np.sum(np.log(mu + exc.at_events[:, 0] * N_TRUE)) - mu * (T1 - T0) - N_TRUE * exc.integral[0]
    assert abs(neural - classical) < 2e-3 * abs(classical) + 0.05


def test_rescaled_intervals_match_classical():
    mu = 1.2
    params = HawkesParams(mu=mu, alpha=np.array([N_TRUE]), betas=np.array([BETA]))
    history, times = simulate_window(params, T0, T1, 120.0, np.random.default_rng(12))
    fit = fit_fixed_decays(times, history, T0, T1, np.array([BETA]))  # any parameters would do; use the MLE
    model = _exponential_model(1.0)
    alpha_hat, mu_hat = float(fit.alpha[0]), float(fit.mu[0])
    model.g_grid = lambda: alpha_hat * BETA**2 * torch.exp(-BETA * model.u)
    with torch.no_grad():
        model.baseline_raw.fill_(float(np.log(np.expm1(mu_hat))))
    from hawkes_fomc.hawkes_neural import NeuralFit

    state = {k: v.clone() for k, v in model.state_dict().items()}
    neural_fit = NeuralFit(
        n=float(model.branching_ratio()), t0=T0, t1=T1, n_events=len(times), loglik=0.0, converged=True, baseline="const",
        s_max=S_MAX, grid_s=model.u.numpy(), phi=model.phi_grid().detach().numpy(), big_phi=model.big_phi_grid().detach().numpy(),
        mu_grid=np.full(1201, mu_hat), state=state,
    )
    # model() would rebuild from the state dict and lose the patched g; evaluate with the patched model directly
    import hawkes_fomc.hawkes_neural as hn

    original = hn.NeuralFit.model
    hn.NeuralFit.model = lambda self: model
    try:
        tau_neural = rescaled_intervals_neural(neural_fit, times, history, T0, T1)
    finally:
        hn.NeuralFit.model = original
    tau_classical = rescaled_intervals(fit, times, history, T0, T1)
    assert np.allclose(tau_neural, tau_classical, rtol=5e-3, atol=2e-3)
    assert abs(tau_neural.sum() - tau_classical.sum()) < 1e-3 * tau_classical.sum()


def test_fit_recovers_branching_ratio_on_simulated_exponential():
    params = HawkesParams(mu=2000 * (1 - N_TRUE) / (T1 - T0), alpha=np.array([N_TRUE]), betas=np.array([BETA]))
    history, times = simulate_window(params, T0, T1, 120.0, np.random.default_rng(13))
    fit = fit_neural(times, history, T0, T1, baseline="const", steps=400, seed=0)
    assert abs(fit.n - N_TRUE) < 0.08, fit.n
    classical = fit_fixed_decays(times, history, T0, T1, np.array([BETA]))
    assert fit.loglik > classical.loglik - 0.01 * abs(classical.loglik)  # flexible kernel cannot be much worse than the truth's family
    held = loglik_neural(fit, times, history, T0, T1)
    assert abs(held - fit.loglik) < 1e-6 * abs(fit.loglik) + 1e-6
