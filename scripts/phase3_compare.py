"""Phase 3: does endogeneity drop, spike, or spike-then-drop around the announcement?

Compares the branching ratio across intraday periods for every fitted model class, on
non-overlapping windows only (adjacent 5-minute-stride windows share half their data). Three tests
per comparison: Welch t on window-level n, a permutation test of period labels, and a z-test
that weights each window by its parametric-bootstrap variance. A robustness table records the
sign of each effect across window lengths, kernel families, baselines, merge horizons and grid.
On a day without an announcement these are placebo comparisons and calibrate the null.
Writes results/phase3_periods.csv, results/phase3_tests.csv, results/phase3_signs.csv, figures/phase3_periods.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from math import comb
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_fomc import figures
from hawkes_fomc.config import DEFAULT_SESSION, RESULTS_DIR, SEED, Session

PERIODS = {
    "morning": ("09:45", "13:30"),
    "pre": ("13:30", "14:00"),
    "statement": ("14:00", "14:30"),
    "press_conference": ("14:30", "15:30"),
    "close": ("15:30", "16:00"),
}
COMPARISONS = [("statement", "morning"), ("press_conference", "morning"), ("statement", "pre"), ("press_conference", "statement"), ("close", "morning")]


def label_periods(table: pd.DataFrame, session: Session) -> pd.DataFrame:
    table = table.copy()
    table["period"] = None
    for name, (a, b) in PERIODS.items():
        inside = (table["t_mid"] >= session.seconds(a)) & (table["t_mid"] < session.seconds(b))
        table.loc[inside, "period"] = name
    return table[table["period"].notna()]


def non_overlapping(table: pd.DataFrame) -> pd.DataFrame:
    """Keep windows spaced a full window length apart, so no two share data.

    The anchor has to be per window-length: this table concatenates 5-, 10- and 20-minute runs, and
    measuring every offset from one global minimum t0 silently keeps overlapping windows.
    """
    lengths = (table["t1"] - table["t0"]).round(6)
    keep = pd.Series(False, index=table.index)
    for length, group in table.groupby(lengths):
        keep.loc[group.index] = np.isclose(np.mod(group["t0"] - group["t0"].min(), length), 0.0)
    return table[keep]


def permutation_p(x: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None, n_perm: int = 20000) -> float:
    """Two-sided label-permutation p-value.

    Seeded per call so the result does not depend on how many comparisons ran before it. The
    tolerance on the comparison matters: with three values a side, a permutation that merely reorders
    one group differs from the observed statistic only by float rounding, and a bare >= drops it.
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    pooled = np.concatenate([x, y])
    observed = abs(x.mean() - y.mean())
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        count += abs(pooled[: len(x)].mean() - pooled[len(x):].mean()) >= observed - 1e-12
    return (count + 1) / (n_perm + 1)


def activity_control(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Repeat the period contrasts on n residualised against log event rate.

    Trading activity has a pronounced intraday profile, and an announcement raises activity by
    construction. If a period difference in n survives only while activity is uncontrolled, it is a
    restatement of the volume profile rather than a statement about reflexivity.
    """
    rows = []
    for fitter, sub in table.groupby("fitter", sort=False):
        sub = sub[sub["n_events"] > 0]
        if len(sub) < 8:
            continue
        x = np.log(sub["n_events"].to_numpy() / (sub["t1"] - sub["t0"]).to_numpy())
        y = sub["n"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        residual = pd.Series(y - (slope * x + intercept), index=sub.index)
        groups = {p: residual[sub["period"] == p].to_numpy() for p in sub["period"].unique()}
        correlation = np.corrcoef(x, y)[0, 1]
        for a, b in COMPARISONS:
            if a not in groups or b not in groups or len(groups[a]) < 2 or len(groups[b]) < 2:
                continue
            rows.append({
                "fitter": fitter, "comparison": f"{a} - {b}", "corr_n_lograte": correlation, "slope": slope,
                "raw_diff": table[(table.fitter == fitter) & (table.period == a)]["n"].mean() - table[(table.fitter == fitter) & (table.period == b)]["n"].mean(),
                "activity_adjusted_diff": groups[a].mean() - groups[b].mean(),
                "welch_p": stats.ttest_ind(groups[a], groups[b], equal_var=False).pvalue,
                "perm_p": permutation_p(groups[a].copy(), groups[b].copy(), rng),
            })
    return pd.DataFrame(rows)


def compare(table: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods, tests = [], []
    for fitter, sub in table.groupby("fitter", sort=False):
        groups = {p: g for p, g in sub.groupby("period")}
        for p, g in groups.items():
            periods.append({"fitter": fitter, "period": p, "windows": len(g), "n_mean": g["n"].mean(), "n_sd": g["n"].std(ddof=1) if len(g) > 1 else np.nan,
                            "n_min": g["n"].min(), "n_max": g["n"].max(), "boot_sd_mean": g["boot_sd"].mean() if "boot_sd" in g else np.nan})
        for a, b in COMPARISONS:
            if a not in groups or b not in groups or len(groups[a]) < 2 or len(groups[b]) < 2:
                continue
            x, y = groups[a]["n"].to_numpy(), groups[b]["n"].to_numpy()
            welch = stats.ttest_ind(x, y, equal_var=False)
            # smallest two-sided p a label permutation can produce: with a handful of non-overlapping
            # windows per period this floor is often above 0.05, so "not significant" may mean "no power"
            floor = 2.0 / comb(len(x) + len(y), len(x))
            row = {"fitter": fitter, "comparison": f"{a} - {b}", "windows_a": len(x), "windows_b": len(y), "diff": x.mean() - y.mean(),
                   "welch_t": welch.statistic, "welch_p": welch.pvalue, "perm_p": permutation_p(x.copy(), y.copy(), rng),
                   "perm_p_floor": floor, "mannwhitney_p": stats.mannwhitneyu(x, y, alternative="two-sided").pvalue}
            if "boot_sd" in groups[a] and groups[a]["boot_sd"].notna().all() and groups[b]["boot_sd"].notna().all():
                # inverse-variance weighted means; the bootstrap SD captures estimation noise, the across-window SD adds real variation
                wa, wb = 1 / groups[a]["boot_sd"] ** 2, 1 / groups[b]["boot_sd"] ** 2
                ma, mb = np.sum(wa * x) / wa.sum(), np.sum(wb * y) / wb.sum()
                se = np.sqrt(1 / wa.sum() + 1 / wb.sum())
                row.update({"weighted_diff": ma - mb, "weighted_z": (ma - mb) / se, "weighted_p": 2 * stats.norm.sf(abs((ma - mb) / se))})
            tests.append(row)
    return pd.DataFrame(periods), pd.DataFrame(tests)


def load_all(session: Session) -> dict[str, pd.DataFrame]:
    sources = {}
    for path in sorted(RESULTS_DIR.glob("phase1_rolling_trades_w*.csv")) + sorted(RESULTS_DIR.glob("phase2_rolling_trades_w*.csv")):
        sources[path.stem] = pd.read_csv(path)
    robust = RESULTS_DIR / "phase1_robustness.csv"
    if robust.exists():
        table = pd.read_csv(robust)
        for (family, variant), sub in table.groupby(["family", "variant"]):
            if family == "rounding" and "seed 0" not in variant:
                continue
            sources[f"robustness: {variant}"] = sub.assign(fitter=sub["fitter"] + f" [{variant}]")
    return sources


def main(session: Session = DEFAULT_SESSION) -> None:
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    main_sources = {k: v for k, v in load_all(session).items() if not k.startswith("robustness")}
    combined = pd.concat(main_sources.values(), ignore_index=True)
    combined = non_overlapping(label_periods(combined, session))
    periods, tests = compare(combined, rng)
    periods.to_csv(RESULTS_DIR / "phase3_periods.csv", index=False)
    tests.to_csv(RESULTS_DIR / "phase3_tests.csv", index=False)
    print("Period means on non-overlapping windows")
    print(periods.round(3).to_string(index=False))
    print("\nTests")
    print(tests.round(4).to_string(index=False))
    underpowered = tests[tests["perm_p_floor"] > 0.05]["comparison"].unique()
    if len(underpowered):
        print(f"\nComparisons where a permutation test cannot reach p<0.05 at any effect size: {sorted(underpowered)}")
        print("The weighted z-test ignores between-window variation and will look far more significant than it is.")

    adjusted = activity_control(combined, rng)
    adjusted.to_csv(RESULTS_DIR / "phase3_activity_adjusted.csv", index=False)
    print("\nPeriod contrasts after removing the log-activity trend")
    print(adjusted.round(4).to_string(index=False))

    signs = []
    for source, table in load_all(session).items():
        labelled = non_overlapping(label_periods(table, session))
        _, t = compare(labelled, rng)
        for _, row in t.iterrows():
            if row["comparison"] in ("statement - morning", "press_conference - morning", "press_conference - statement"):
                signs.append({"source": source, "fitter": row["fitter"], "comparison": row["comparison"], "diff": row["diff"], "sign": int(np.sign(row["diff"])), "perm_p": row["perm_p"]})
    signs = pd.DataFrame(signs)
    signs.to_csv(RESULTS_DIR / "phase3_signs.csv", index=False)
    print("\nSign of the effect across specifications")
    print(signs.round(4).to_string(index=False))
    flips = signs.groupby("comparison")["sign"].agg(lambda s: sorted(set(s)))
    print("\nsigns present per comparison:", flips.to_dict())

    import matplotlib.pyplot as plt

    figures.style()
    order = list(PERIODS)
    fitters = list(dict.fromkeys(periods["fitter"]))
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, (fitter, color) in enumerate(zip(fitters, figures.SERIES * 2)):
        sub = periods[periods["fitter"] == fitter].set_index("period").reindex(order)
        x = np.arange(len(order)) + (k - len(fitters) / 2) * 0.09
        err = sub["n_sd"] / np.sqrt(sub["windows"])
        ax.errorbar(x, sub["n_mean"], yerr=err, color=color, marker="o", markersize=4, linestyle="none", capsize=2, label=fitter)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{p}\n{PERIODS[p][0]}-{PERIODS[p][1]}" for p in order])
    ax.set_ylabel("branching ratio n, period mean +/- SE over windows")
    ax.set_title(f"Branching ratio by intraday period, non-overlapping 10-minute windows, {session.date}", loc="left")
    ax.legend(ncol=3, fontsize=7)
    print("figure:", figures.save(fig, "phase3_periods"))


if __name__ == "__main__":
    main()
