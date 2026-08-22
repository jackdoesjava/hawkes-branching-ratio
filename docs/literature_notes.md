# Notes on the Hawkes branching-ratio literature

**Scope.** Univariate Hawkes process on 5–20-minute windows (≈500–5,000 events) of equity trades. The estimand is n = ∫φ, the mean number of direct children per event, equal under stationarity to the endogenous fraction of events (Hawkes & Oakes 1974). Page and volume references are from the published versions unless an arXiv copy is noted.

---

## 1. Filimonov & Sornette (2012, 2015)

**Filimonov & Sornette (2012), *Phys. Rev. E* 85, 056108.** E-mini S&P 500 mid-quote changes, 1998–2010; exponential kernel; MLE on rolling 10-, 20- and 30-minute windows with a 5-minute step; one-second exchange timestamps, events redistributed uniformly within the second. They report n rising from ≈0.3 (1998) to 0.7–0.8 after 2007, ≈0.95 at the 2010 flash crash. The companion preprint Filimonov, Wheatley & Sornette (arXiv:1306.2245, App. B; venue not verified) studies the finite-sample MLE: for n < 0.9 the 90% quantile of |n̂ − n| is below 0.1 beyond ≈200–400 events; for n ≳ 0.9 a large, length-dependent *negative* bias appears; at small n the non-negativity constraint induces a *positive* bias.

**Filimonov & Sornette (2015), *Quantitative Finance* 15(8), 1293–1314.** Verified findings:

- *Outliers.* With power-law kernels "just 0.17% of outliers are sufficient to lead to the spurious conclusion that the system is at or close to critical"; 1% of milder outliers add ≈0.28 to n̂. Exponential kernels stay within one s.d. of the truth.
- *Short-time regularization.* Three power laws with identical tails but different short-time forms give different n̂; fitting the wrong form biases n̂ upward by ≳0.2 for n > 0.3 in one direction and ≲0.07 in the other.
- *Edge effects.* A power law with τ₀ = 1 s, ε = 0.15 needs 1.1×10⁷ s to reach 95% of its mass; cold-started 30-minute fits are biased downward for ε ≤ 0.2 and simulations need 10⁸–10⁹ s burn-ins. Exponential transients are short.
- *Multimodal likelihood.* One two-month sample gives a global optimum at n̂ = 0.075 and a local one at 1.105; "exhaustive search of the absolute maximum" is required.
- *Regime shifts (Sec. 4.4).* Concatenating two Hawkes samples with n₁ = 0.5, n₂ = 0.2 gives n̂ = 1; same n but baselines differing by 40% gives n̂ = 0.9, by 60% gives 1. Independent *Poisson* days with the E-mini's empirical daily rates, fitted as one Hawkes process, yield n̂ "hovering around the critical value 1" for 14 years, and residual analysis does not reject. "In the presence of non-stationarity, a calibration using the Hawkes model with any kernel (including short-memory exponential kernel) will result in significant biases."
- *Timestamps (Sec. 4.2).* Vendor millisecond stamps are often packet-arrival times over second-resolution exchange times. Randomizing within 1 ms when the true bundling interval is 1 s gives a spurious n̂ ≈ 0.856 for pure Poisson and for Hawkes with n = 0.5 alike. Remedy: use the exchange's own resolution, or show robustness across randomization intervals.
- *Windows.* They defend 10–30-minute regular-hours windows against two-month concatenations, since daily activity varies by factors of 2–5.

---

## 2. Hardiman, Bercot & Bouchaud (2013); Hardiman & Bouchaud (2014)

**HBB (2013), *Eur. Phys. J. B* 86, 442.** Same E-mini data, 1998–2011, millisecond stamps randomized within the millisecond. Kernel: power law as a sum of M = 15 exponentials, ξᵢ = τ₀·5ⁱ, weights (1/ξᵢ)^(1+ε); MLE on two-month concatenations of regular hours. They find ε ≈ 0.15 below ~10³ s, a crossover to τ^(−1.45) at 10³–10⁶ s, and n ≈ 1 throughout. They *did* control the intraday profile: λ(t) = (1/w(t))[μ + ∫w(s)φ(t−s)dN(s)], w(t) the reciprocal of the empirical hourly rate, because "without appropriate detrending, the fitting procedure would interpret the slow variation in event rate during the day as arising from the shape of the Hawkes kernel". Their critique of FS2012: a simulated critical power law fitted with exponentials on 30-minute windows "reveals a sub-critical branching ratio which increases steadily over time", since short windows "only pick up the short-term (< 30 min) reflexivity". The small positive μ they obtain is kernel mass beyond the window.

The two camps agree on the mechanism: long windows with constant μ let non-stationarity masquerade as long memory; short windows with short-memory kernels truncate real long memory. A 10-minute exponential-mixture fit estimates kernel mass below its slowest component, not total n.

**Hardiman & Bouchaud (2014), *Phys. Rev. E* 90, 062807.** Likelihood-free estimator n ≈ 1 − √(mean/variance) of bin counts in windows W ≫ kernel support. For finite W it "systematically underestimates"; for a near-critical power law 1 − ñ(W) ∼ W^(−ε), and W = 10 s reproduces FS's numbers, "so their procedure must also underestimate". Counts are deseasonalized by a daily profile. A useful cross-check, but more baseline-sensitive than MLE: Potiron et al. (§5) find errors above 0.8 under a J-shaped baseline with n = 0.

---

## 3. Lallouache & Challet (2016), *Quantitative Finance* 16(1), 1–11

EBS EUR/USD, Q1 2012, 0.1 s slices (uniform jitter within slice plus volume-based trade reconstruction). Three simultaneous tests at p > 0.05 on time-rescaled residuals: Kolmogorov–Smirnov, Ljung–Box from lag 2, Engle–Russell excess dispersion. Two exponentials pass KS in every hour when fitted hourly; three exponentials pass all three tests on single days; for two consecutive days "no kernel can pass the three tests at the same time". Timescales ≈0.15 s, 10.6 s, 178 s; n grows with window and kernel richness: hourly 0.41 (one exponential) → 0.64 (two/three); daily 0.48–0.85; two days 0.80–0.88. Key sentences: "without [the timestamp correction] no fit ever passes a Kolmogorov–Smirnov test" and "the use of power law-like kernels mechanically increases the apparent endogeneity factor". Multi-day failure is attributed to "the non-stationarities of both exogeneity and endogeneity" (the lunch lull changes the kernel, not only μ). Jitter inflates the smallest fitted timescale by ≈15% and affects only KS. Implication: passing KS on 2,000 events says little about n's interpretation; failing it usually reflects timestamp artefacts or within-window non-stationarity.

---

## 4. Bacry, Mastromatteo & Muzy (2015), *Market Microstructure and Liquidity* 1(1), 1550005

MLE (Ogata 1978) costs O(M²) for general kernels and O(M) for (sums of) exponentials, "one of the major reasons why exponential kernels are so commonly used"; EM on the branching representation (Veen & Schoenberg 2008, *JASA* 103, 614–624; Lewis & Mohler 2011) converges slowly for slow kernels; Wiener–Hopf estimation (Bacry & Muzy 2016, *IEEE Trans. Inf. Theory* 62(4), 2184–2202) identifies φ *given* the mean intensity and conditional-expectation function, but does not separate a deterministic μ(t) from φ. The review distinguishes exogenous non-stationarity (μ(t)) from endogenous (‖φ‖ ≥ 1; Jaisson & Rosenbaum 2015, *Ann. Appl. Probab.* 25(2), 600–631), records exponential fits of order flow with n ≈ 0.9–0.95, and says "the power-law nature of Hawkes kernels remains a solid empirical fact which, at least, calls to question all the approaches based on exponential Hawkes models".

---

## 5. Time-varying μ(t), announcements, endo–exo

**Wheatley, Wehrli & Sornette (2019), *Quantitative Finance* 19(7), 1165–1178** (abstract). Endo–exo identification is "plagued by spurious strong-and-long memory due to improper treatment of trends, shocks and shifts"; EM with BIC selects the flexibility of a deterministic background; criticality is "strongly reject[ed]" univariately and bivariately.

**Wehrli, Wheatley & Sornette (2021), *Quantitative Finance* 21(5), 729–752** (abstract). EUR/USD and E-mini; piecewise-constant or adaptive-logspline immigration intensity chosen by information criteria: "the estimated branching ratio depends little upon window size and is usually far from criticality"; "the (positive) bias incurred by keeping the immigration intensity constant is small for time scales up to two hours, but can become as high as 0.3 for windows spanning days"; n has its own intraday seasonality.

**Omi, Hirata & Aihara (2017), *Phys. Rev. E* 96, 012303.** Bayesian log-baseline on many basis functions; on Nikkei 225 mini it beats constant or slowly varying baselines, most on sessions with post-announcement moves. **Chen & Hall (2013, *J. Appl. Probab.* 50(4), 1006–1024; 2016, *JCGS* 25(1), 209–224)** give MLE asymptotics with a time-varying background and note that a fully non-parametric background plus kernel is not identifiable from one path.

**Rambaldi, Pennesi & Lillo (2015), *Phys. Rev. E* 91, 012819.** EBS (three pairs, 2012, 100 ms), 723 announcements; an exogenous exponential kernel fires at announcement time beside a power-law (15-exponential) kernel. Without the news term n ≈ 0.86–0.91; with it "for high impact news n takes lower values", and the no-news model is ~10⁻⁸ as probable. **Rambaldi, Filimonov & Lillo (2018), *Phys. Rev. E* 97, 032318** show intensity bursts are frequent and mostly *not* news-related, so constant μ fails on ordinary days too.

**Potiron, Scaillet, Volkov & Yu (2025, unpublished working paper).** Itô-semimartingale baseline with jumps. With a null kernel and J-shaped baseline the constant-baseline exponential MLE has mean absolute error ≈0.95 and ≈0.85 (near-critical n̂ with n = 0); with a true exponential kernel and bursty baseline ≈0.07. On E-mini, n ≈ 0.7–0.8 "while alternative methods are positively biased".

---

## 6. Simultaneous executions from one aggressive order

On Nasdaq an aggressive order sweeping several resting orders produces one ITCH *Order Executed* (or *Executed With Price* / non-cross *Trade*) message per resting order, all stamped with the matching-engine nanosecond time (TotalView-ITCH 5.0). Databento's `ts_event` is that timestamp and `side` on trade records is the aggressor side (docs and `dbn` enums; confirm on your build). Counting each execution injects zero-duration clusters that are mechanically, not behaviourally, self-excited: the fastest component absorbs them, and exact ties give zero rescaled durations that fail KS by construction (LC2016; FS2015's 0.856 artefact is the same effect in reverse).

The literature's rule is order-level aggregation. Rambaldi, Bacry & Lillo (2017, *Quantitative Finance* 17(7), 999–1020; EUREX, microsecond stamps): "we treat multiple orders that happen at the same time and on the same side of the order book as a single event (for instance a market order that hits two limit orders present in the book at the same price is regarded as a single trade)"; opposite-side coincidences are < 0.2%. The same paper carries sweep volume as a mark. Schneider & Weber (2023, *Phys. Rev. E* 108, 015303) show uniform redistribution of binned events is inferior to intensity-based reconstruction: merge, do not jitter.

---

## 7. Uncertainty and finite-sample bias

Asymptotic normality with inverse-Fisher covariance: Ogata (1978, *Ann. Inst. Stat. Math.* 30, 243–261); exponential recursion, gradient and Hessian: Ozaki (1979, *ibid.* 31, 145–155); ergodic QMLE theory for exponential Hawkes: Clinet & Yoshida (2017, *Stoch. Proc. Appl.* 127(6), 1800–1839). None is reliable at N ~ 10³ with n near 1 or components on the boundary α_k = 0.

**Cavaliere, Lu, Rahbek & Stærk-Østergaard (2023), *J. Econometrics* 235(1), 133–165.** Monte Carlo with n ∈ {0.2, 0.5, 0.8}, unit mean intensity, spans T ∈ {50, 100, 200}: "the asymptotic CIs suffer from the problem of undercoverage for almost all models and sample spans", worst for the branching ratio (86.5% at nominal 95% for n = 0.8, T = 50); up to 27% of small-sample fits fail their sanity check (n̂ > 1 or indefinite Hessian). Their fixed-intensity bootstrap (FIB; intensity path held fixed, events redrawn) and recursive bootstrap restore coverage, FIB being faster and more robust. Rule: for N ≲ 5,000 report bootstrap intervals, never information-matrix SEs alone, and distrust any asymptotic SE for n̂ > 0.85.

---

## 8. Fixed log-spaced decays: convexity, identifiability, baseline vs slow kernel

With fixed β_k, λ(t) = Σ_j θ_j g_j(t) with g_j ≥ 0, so ℓ(θ) = Σ_i log(θ·g(t_i)) − θ·∫g is concave (strictly iff the {g(t_i)} span the parameter space). This underlies the `HawkesSumExpKern` learners in *tick* (Bacry, Bompaire, Deegan, Gaïffas & Poulsen 2018, *JMLR*) and the fixed-decay framework of Bacry, Bompaire, Gaïffas & Muzy (2020, *JMLR* 21(50)). Bochud & Challet (2007, *Quantitative Finance* 7(6), 585–589) give optimal exponential approximations of power laws (≈four exponentials per three decades per citing papers), so a factor-10 grid is adequate.

**Identifiable timescales in 10 minutes.** A component at τ_k is identified by lag structure at that scale, with ≈T/τ_k independent looks: 60 for the 10 s component; the 1 ms component rests on sub-millisecond durations. Anything with τ_k ≳ T/10 ≈ 60 s is indistinguishable from baseline drift. The 1 ms α is contaminated by sweep residuals and same-parent child-order slicing (statistical, not economic, reflexivity).

**The precise problem.** The slow component's excitation E_s(t) = α_s β_s Σ e^{−β_s(t−t_i)} sums over ≈33 parents per e-fold here (0.3 s mean spacing vs τ_s = 10 s), so E_s(t) ≈ α_s × (event rate smoothed over τ_s), a smoothed copy of λ. A baseline basis function with support comparable to τ_s reproduces the same function on the event set; g_s and g_baseline are nearly collinear, the observed information has a near-null direction along (α_s, −μ_j), and the likelihood has a ridge: the *sum* of baseline plus slow excitation is sharply determined, the split is not. Constant μ resolves the ridge toward α_s (FS2015's regime-shift inflation); an over-flexible μ(t) resolves it toward μ (HBB's "excess base intensity" deflation).

**Resolutions.** (a) Keep τ_s ≪ knot spacing Δ: 10 s/120 s ≈ 1/12 is acceptable, but verify per window via the score correlation between α_s and each μ_j and the profile of ℓ over α_s. (b) Compare nested baselines (constant ⊂ piecewise ⊂ spline) by BIC and a clearly specified held-out likelihood, reporting n̂ under each. (c) Simulate 20–40 s bursts, which 2-minute pieces cannot absorb, and report how far n̂ moves. (d) Report α_k by component and cumulative mass by timescale, and call the headline number n_{≤10 s}.

---

## 9. What this means for my setup

**Data.**

1. *Venue fragmentation.* SPY is NYSE Arca-listed and trades everywhere; Cboe reports off-exchange share above 50% of US volume in 2025, and Nasdaq's share of SPY on this day is unknown (I found no published figure; measure it). Trades on other venues that trigger Nasdaq trades enter as immigrants, deflating n̂ and loading μ(t) with cross-venue clustering. Report the share, label the result "Nasdaq-venue n", and re-run on a consolidated feed as robustness.
2. *One day is anecdotal.* Include at least a quiet day, an FOMC/CPI day and a rebalance day; Wehrli et al.'s "small bias below two hours" does not cover announcement bursts, which is the case my μ(t) targets.
3. *Aggregation.* Identical-`ts_event` merging matches Rambaldi–Bacry–Lillo; add their same-side condition, exclude opening/closing cross prints, and run a merge-horizon ladder (0, 10 μs, 100 μs, 1 ms) as FS2015 do for randomization intervals. Prefer defining aggressor orders from MBO records (identical `ts_event`, contiguous sequence numbers); use MBP-10 only for the mid-price stream. Report residual exact ties and break them before KS.
4. *Mid-price changes (~635k/day).* For a one-cent-tick ETF these are dominated by sub-millisecond queue depletion/refill; they need their own grid (likely a 100 μs component) and aggregation rule.
5. *Session edges.* Drop or flag windows touching the first and last 15 minutes.

**Model and estimation.**

6. *State the estimand.* With 10 s as the slowest decay the model cannot hold mass at 1–10 minutes; HBB's critique applies verbatim. Show in simulation what an ε = 0.15 power-law truth does to your grid (mass leaks into μ and α_10s in a baseline-dependent split).
7. *Concavity* holds for piecewise and spline baselines too; L-BFGS-B with bounds is right. But α_k = 0 boundary solutions occur with probability near ½ for unused decades, so Hessian SEs are invalid and n̂'s distribution is a mixture; print the information matrix's condition number per window.
8. *Burn-in history* removes the cold-start deflation for fast components (two minutes suffices for the 10 s component). The bootstrap must replicate it, and history events must be excluded from the likelihood sum and the KS residuals.
9. *Single exponential via profile likelihood.* Report the profile: with two coexisting timescales it is flat or bimodal and β̂ can jump between windows, producing spurious jumps in n̂ (FS2015's point (iv)).
10. *Held-out likelihood.* A piecewise/spline μ(t) fitted on window k has no value on window k+1; specify the rule (re-fit μ with kernel frozen, or carry the last piece), noting the former scores only the kernel. Add in-sample BIC as Wheatley/Wehrli do.
11. *Goodness of fit.* KS alone is "excessively demanding" (LC2016) yet failed to reject regime-switching Poisson fitted as critical Hawkes (FS2015). Report KS, Ljung–Box and excess dispersion on time-rescaled residuals (Ogata 1988, *JASA* 83, 9–27; Brown et al. 2002, *Neural Comput.* 14(2), 325–346); with a 5-minute stride, adjacent p-values are not independent.
12. *Uncertainty.* Use FIB as the primary bootstrap (it does not require n̂ < 1 to simulate), handle n̂ ≥ 1 windows explicitly, report asymmetric percentile intervals, and expect undercoverage below N ≈ 1,000.
13. *Overlap.* Adjacent n̂ share half their data; tests for intraday variation in n need non-overlapping windows.

**Simulation validation.**

14. Add n ∈ {0.95, 0.99} (FWS2013 negative-bias regime; Cavaliere sanity-check regime), N = 500, and a misspecified truth (HBB's 15-exponential power law, ε = 0.15, τ₀ = 0.1 s, long burn-in); report recovered n_{≤10 s} against true total and true sub-10 s mass.
15. Split the constant-μ inflation test into a smooth U-shape (fixable), a mid-window step (fixable only if a knot lands near it), and 20–40 s bursts (not fixable; quantify). Run the converse: true n = 0.9, constant μ, spline baseline, to show it does not steal genuine slow excitation.
16. Simulate sweeps: split events into k-fold same-timestamp executions with an empirical k distribution; show un-merged fits inflate α_1ms and n̂ and merging restores them.

---

## 10. Checklist

1. Document timestamp provenance and exact-tie counts before/after aggregation.
2. Aggregate to aggressor orders (same `ts_event`, same side) with a merge-horizon ladder.
3. Exclude crosses and the first/last 15 minutes; report Nasdaq's share of SPY volume.
4. Fit nested baselines; report n̂ under each with BIC and a specified held-out rule.
5. Report α_k by component; headline n = n_{≤10 s}.
6. Per window: information-matrix condition number and α_10s–baseline score correlation.
7. FIB bootstrap intervals; flag n̂ ≥ 0.9 as biased downward with unreliable asymptotic SEs.
8. KS + Ljung–Box + excess dispersion on rescaled residuals; no pooling across overlapping windows.
9. Cross-check with the Hardiman–Bouchaud mean–variance estimator.
10. Simulations: near-critical n, small N, power-law truth, regime step, short bursts, synthetic sweeps.
