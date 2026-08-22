# What I found

## The day

The file is SPY on 2021-01-28. I picked it believing it was an FOMC day. It is not: the January 2021
statement came out on the 27th, so this is the day after. There is nothing scheduled at 14:00 or
14:30, and the event counts agree — the ten minutes around 14:00 hold 984 trade events against a
session median near 1600, and the only real bursts in the day are the open and the close.

Rather than throw it away I ran the whole thing as a placebo. If the method reports a 14:00 effect
on a day with no 14:00 event, that number is the floor any real announcement result has to clear.

## The rolling estimate

Fitting 10-minute windows on a 5-minute stride, 77 windows across the session:

| fit | mean n | range | KS rejected |
|---|---|---|---|
| mixture kernel, constant baseline | 0.68 | 0.47 to 0.94 | 10% of windows |
| mixture kernel, spline baseline | 0.61 | 0.31 to 0.79 | 9% of windows |

The two differ by roughly 0.07 all day. That gap is the constant-versus-flexible baseline choice, and
per the simulations it is not resolvable from one window: a slow kernel component and a drifting
baseline look the same. The single-exponential fits sit much lower, around 0.35, which is what the
simulations predict for a single exponential fitted to something genuinely multi-scale.

Both mixture fits track each other closely through the day, and outside the closing auction neither
exceeds 0.83; the 15:50-16:00 window is the exception, where the constant-baseline fit reaches 0.94.
The curve drifts with the activity profile: high in the first hour, sagging through midday, lowest
just before 14:00, then climbing into the close.

## The placebo fires

Splitting the day into periods and comparing on non-overlapping windows only:

| period | mix, constant mu | mix, spline mu |
|---|---|---|
| morning 09:45-13:30 | 0.68 | 0.61 |
| pre 13:30-14:00 | 0.54 | 0.44 |
| statement 14:00-14:30 | 0.74 | 0.60 |
| press conf 14:30-15:30 | 0.72 | 0.66 |
| close 15:30-16:00 | 0.80 | 0.66 |

Comparing 14:00-14:30 against the half hour immediately before it gives +0.205 with a constant
baseline and +0.165 with a spline. Comparing the same 14:00-14:30 window against the whole morning
gives +0.06 and -0.01. So the answer flips depending on which control you pick, on a day when
nothing happened at 14:00.

The reason is that 13:30-14:00 is the bottom of the daily activity trough, and n sags with it. Any
study that uses the pre-announcement window as its control is measuring the recovery from the
lunchtime lull and calling it an announcement effect.

## How significant is it really

Not very, and the tests disagree in a way worth spelling out. For the statement-versus-pre
comparison with a constant baseline:

- Welch t-test: p = 0.018
- permutation test: p = 0.098
- inverse-variance z-test weighted by bootstrap SEs: p < 0.001

The weighted test looks decisive because it only counts bootstrap uncertainty and treats the
variation between windows as zero, which is exactly the variation that matters. The permutation test
is the honest one, and with three non-overlapping windows on each side its smallest possible
two-sided p-value is 0.1, so it cannot reject at 5% no matter how big the effect is. Thirty-minute
event periods give three independent windows. That is not enough to claim significance from.

## Robustness

Across window length, kernel family, baseline, merge horizon, jitter seed and decay grid, the sign
of the statement-versus-morning contrast flips between specifications: the constant-baseline fits
make it positive, the spline-baseline fits make it negative, and neither is close to significant.
The press-conference contrasts hold their sign more often but are no more significant. Given the
effect is smaller than the spread across specifications, that is the expected outcome rather than a
defect.

Two specific checks are worth recording.

Window length does not matter much: the shape of the day, trough included, is the same at 5, 10 and
20 minutes.

Timestamp rounding matters enormously, but not in n. Rounding trade times to a full second and
re-jittering uniformly inside each second wrecks the microstructure completely, and n only moves from
0.680 to 0.691. Meanwhile the kernel abandons the 10 microsecond to 10 second range and dumps mass
0.505 out of 0.691, nearly three quarters of it, onto a single 0.3-second component, which is just
the jitter scale, and the held-out kernel-only gain over a Poisson model falls from 1.42 to 0.26 nats
per event. This is the Filimonov-Sornette rounding
artefact happening in front of me. The lesson is that n being stable under a timestamp treatment says
nothing about whether the timestamps survived it.

## What I would need to answer the original question

A real FOMC file, obviously. Beyond that:

Control days. One day cannot separate the intraday profile from anything else, and the intraday
profile is what produced the fake result above. Several ordinary days would let the profile be
estimated and removed.

A baseline that can follow a burst. The stress tests say a 30-second burst in the exogenous rate
gives n = 0.86 when the truth is 0.30, under both constant and 2-minute-spline baselines. An
announcement is a burst. Until the baseline can track something that fast, an announcement-day spike
in n cannot be distinguished from the announcement simply raising the arrival rate for half a minute.

More venues. This is Nasdaq only, and SPY trades everywhere. Trades on other venues that trigger
Nasdaq trades enter the model as exogenous arrivals, so this n is a venue-level quantity.
