# Hawkes branching ratio on intraday tick data

Estimating the branching ratio of the trade arrival process in rolling windows through a single
trading day, using both a classical sum-of-exponentials Hawkes model and a neural kernel.

The original question was whether market endogeneity rises or falls when genuine exogenous
information arrives. An FOMC statement is about the cleanest test available: it lands at a known
second, it is exogenous, and it is big. So the plan was to estimate n(t) through an FOMC day and see
what happens at 14:00 and 14:30 ET.

## The data is not what I thought it was

The file I have is SPY on 2021-01-28. The January 2021 FOMC statement came out on the 27th, so this
is the day after. There is no scheduled Fed communication at 14:00 or 14:30, and the per-minute
counts confirm it: the ten minutes around 14:00 hold 984 trade events against a session median near
1600. No burst at either mark.

I ran the whole pipeline anyway, as a placebo. Every 14:00 comparison here measures what the method
reports when nothing happens, which is the number a real announcement result would have to beat. It
turns out not to be zero, and that is the main result below.

The code is date-parameterised (`hawkes_fomc.config.Session`), so it runs unchanged on a real FOMC
file.

## Data handling

Databento XNAS.ITCH MBP-10, nanosecond exchange timestamps. 4,576,851 book records over
04:00-19:00 ET, 4,006,587 inside the regular session.

The event stream is trades, one event per aggressive order rather than per fill. ITCH reports an
order that sweeps several resting orders as several executions sharing one timestamp, and counting
each fill separately invents self-excitation at zero lag. Merging by timestamp takes 108,133
executions down to 78,020 events, with up to 43 fills collapsing into one order. Opposite aggressor
sides are never merged even inside one nanosecond, since those are two orders; if that ever happens
they get separated by one nanosecond instead. On this day it never does. Mid-price changes (633,313
in the session) are available as a second stream.

The estimate is a branching ratio for Nasdaq trades over timescales up to a few seconds. Trades on
other venues that trigger Nasdaq trades show up as exogenous arrivals, so this is not a market-wide
reflexivity number.

## Model

    lambda(t) = mu(t) + sum_k alpha_k * beta_k * exp(-beta_k * (t - t_i))

summed over all past events including ones before the window starts, so windows are not started
cold. The decays are fixed on a half-decade grid from 10 microseconds to 10 seconds (13 terms),
which is the usual sum-of-exponentials stand-in for a power-law kernel. Fixing the decays makes the
log-likelihood concave in the remaining parameters, so the MLE is a global maximum rather than
whatever a joint optimiser happened to find. n is the total kernel mass, sum of alpha_k.

mu(t) is constant, piecewise constant, or a linear spline with 2-minute knots. All three are linear
bases so concavity survives. I report constant and flexible fits side by side throughout, for
reasons in the next section.

Fitting is L-BFGS-B with an analytic gradient, followed by a projected Newton polish. The polish
matters: L-BFGS-B stops on relative function decrease and leaves a weakly identified baseline well
short of a small gradient, which shows up as spurious non-convergence.

A single exponential with a free decay is also fitted, by profile likelihood over a decay grid with
Brent refinement. If the profile optimum lands on an end of the grid the fit is reported as not
converged, since the likelihood is still rising as beta leaves the grid.

Per window I compute time-rescaled residuals and test them three ways (KS against Exp(1),
Ljung-Box, Engle-Russell excess dispersion), parametric bootstrap intervals for n, held-out
log-likelihood on the next non-overlapping window, and the condition number of the information
matrix as a collinearity check.

Phase 2 swaps the fixed grid for a learned kernel while keeping n well defined:

    phi(s) = integral from s to S of g(u) du,   g = softplus(MLP(log u)) / u

so phi is positive and decreasing by construction and n = integral of u*g(u) is computable. All
integrals are trapezoid quadrature on a log grid at 200 nodes per decade, which reproduces the
closed-form exponential kernel to about 1e-4 relative on n. Training is Adam then an L-BFGS polish.

## What the simulations say

All of this is in `results/` and regenerates with `python run_all.py`.

The estimator works where the model is right. Across 6300 fits on simulated data with known n, the
fixed-grid MLE recovers n to within about 0.01 for exponential and on-grid mixture kernels at
n = 0.3, 0.6 and 0.9, with 95% interval coverage between 0.86 and 0.98.

The failures are more interesting.

**Baseline and slow kernel are the same thing.** I originally had a comment in `config.py` claiming
the 10-second component was fast enough compared to the 2-minute baseline knots that the two could
not compete. That was wrong. Given a true 10-second component carrying mass 0.30, the recovered slow
mass is 0.22 under a constant baseline, 0.11 under 120-second knots, and 0.00 under 30-second knots.
Even with the true decays known, a 10-second component in a 10-minute window is biased by -0.056,
and needs roughly 40 minutes of data before it comes within 0.015. So the gap between the constant
and spline estimates is not a nuisance to be resolved, it is the genuine ambiguity between slow
self-excitation and a drifting exogenous rate. Both numbers appear in every result.

**Bursts get through.** A smoothly varying exogenous rate pushes a constant-baseline fit to 0.96
when the truth is 0.30, and the spline baseline pulls it back to 0.32. A mid-window step is fixed
too. But 30-second bursts are not: both baselines return about 0.86 against a truth of 0.30, because
2-minute pieces cannot follow them. An announcement is a burst, so this is the biggest threat to any
FOMC-day claim and nothing at this resolution defends against it.

**Precision comes from window length, not event count.** With the true decay known, the SD of n at
n = 0.3 is 0.042, 0.032 and 0.035 for 500, 2000 and 5000 events in a 10-minute window. Ten times the
events buys about 20%. The information is in how many kernel relaxation times fit in the window,
roughly SD ~ 1/sqrt(T*beta). Two consequences: a busy post-announcement window is estimated no
better than a quiet one, and differences below about 0.03 to 0.05 are noise no matter how many
trades are in them. Period comparisons are weighted by bootstrap variance, never by event count.

**Grid coarseness biases n upward, and it does not wash out.** A decade grid biases n up by about
+0.06 when the true timescale falls between grid points; half-decade spacing halves that. This is a
non-negativity effect, since unused components can only pick up non-negative noise. It is not a
small-sample problem: at n = 0.3 the bias is +0.025, +0.038 and +0.028 at 500, 2000 and 5000 events.
A spline baseline removes most of it. The bootstrap reproduces it, so it could be corrected, but I
report it instead because it is the same size as the differences this day produces.

**Unmerged sweeps inflate n by about +0.11** and the KS test catches them in every window. Merging
fixes both.

**Misspecification is asymmetric.** A single exponential fitted to a multi-scale kernel understates n
by 0.08 to 0.17 and fails KS in every window past about 2000 events. A power-law truth with most of
its mass beyond the window is recovered as its within-window mass, which is the right answer to a
different question.

**Near criticality the MLE is biased low**, -0.005 at n = 0.95 and -0.038 at n = 0.99.

**n is a bad diagnostic of timestamp quality.** Rounding the real trade times to one second and
re-jittering uniformly inside each second destroys the whole microstructure, and n moves from 0.680
to 0.691. Everything else falls apart: the kernel abandons the 10 microsecond to 10 second range and
puts 0.505 of its mass on a single 0.3-second component (the jitter scale itself), and the held-out
gain over Poisson drops from 1.42 to 0.26 nats per event. This is the Filimonov-Sornette rounding
artefact from the inside. A stable n across timestamp treatments is not evidence the timestamps are
fine.

## Results on the day

See `docs/findings.md` for the write-up. Short version: comparing 14:00-14:30 against the half hour
before it gives a rise of about +0.20, and comparing the same window against the whole morning gives
about +0.06, which is nothing. The two disagree because 13:30-14:00 is the bottom of the daily
activity trough, so anchoring there manufactures a 14:00 spike on a day when nothing happened. With
three non-overlapping windows per 30-minute period there is essentially no power anyway: a
permutation test cannot reach p < 0.05 at any effect size.

## Things that are handled

- Sweep ties are merged rather than jittered. Jitter is implemented, but only for the
  timestamp-rounding robustness check.
- Pre-window events enter the excitation sum as history, so no cold start.
- Nested baselines, always reported together.
- The decay grid reaches 10 microseconds because a 1 ms fastest component fails the KS test on the
  lower tail of the rescaled intervals without changing n. Same for the single-exponential search
  grid, which used to pin at its own 1 ms edge in 40% of windows and report convergence anyway.
- Three residual tests, since KS alone is over-demanding at large N and blind to serial dependence.
- Windows touching the first or last 15 minutes are flagged, and the Phase 3 tests use only
  non-overlapping windows since a 5-minute stride on 10-minute windows shares half the data.
- Unsorted or tied event times raise instead of silently producing a wrong intensity, and an empty
  window returns a defined degenerate fit instead of NaNs.

`docs/literature_notes.md` has my notes on the papers these issues come from.

## Layout

```
hawkes_fomc/
  config.py            paths, ET session clock, decay grid, seed
  data.py              DBN loading, event streams, sweep aggregation, jitter
  simulate.py          cluster simulation with mixture-of-exponential kernels
  hawkes_classical.py  excitation recursion, baselines, concave MLE, profile fit, GOF
  hawkes_neural.py     monotone learned kernel with a computable branching ratio
  windows.py           rolling windows, bootstrap, held-out scoring
  figures.py           ET-axis plots with the 14:00 and 14:30 marks
scripts/
  phase0_audit.py      data audit
  phase1_recovery.py   simulation recovery
  phase1_stress.py     near-critical, power-law, bursts, sweeps
  phase1_rolling.py    rolling classical fits with bootstrap
  phase1_robustness.py merge horizon, rounding, window length, grid
  phase2_neural.py     neural validation and rolling fits
  phase3_compare.py    period comparison
tests/
```

## Running it

```
pip install -r requirements.txt
python run_all.py
```

You need the raw Databento file in `data/raw/`. It is not in the repo (129 MB, and it is licensed
data), but the job manifest and metadata are, so the same query can be reissued. Seeds are fixed in
`hawkes_fomc.config`.
