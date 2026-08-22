from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .config import FIGURES_DIR, Session

# Categorical slots in the validated order (adjacent pairs pass CVD separation); light surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK_SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6,
    })


def et_axis(ax: plt.Axes, session: Session, t_start: float | None = None, t_end: float | None = None, label_marks: bool = True) -> None:
    """Clock-time x axis in ET with the announcement marks drawn."""
    t_start = session.open_s if t_start is None else t_start
    t_end = session.close_s if t_end is None else t_end
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_locator(MultipleLocator(3600 if t_end - t_start > 8 * 3600 else 1800))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda s, _: session.clock(s)[:5]))
    ax.set_xlabel(f"ET, {session.date}")
    ymax = ax.get_ylim()[1]
    for hhmm, name in session.marks:
        t = session.seconds(hhmm)
        ax.axvline(t, color=INK_SECONDARY, linewidth=0.9, linestyle="--", zorder=0.5)
        if label_marks:
            ax.text(t, ymax, f" {hhmm} {name}", color=INK_SECONDARY, fontsize=7, va="top", ha="left", rotation=90)


def save(fig: plt.Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def per_minute_figure(panels: list[tuple[str, np.ndarray, np.ndarray]], session: Session, title: str) -> plt.Figure:
    """Stacked panels of events per minute, one per stream: (label, minute_start_seconds, counts)."""
    style()
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.2 * len(panels) + 0.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (label, edges, counts), color in zip(axes, panels, SERIES):
        ax.step(edges, counts, where="post", color=color, linewidth=1.2)
        ax.fill_between(edges, counts, step="post", color=color, alpha=0.15, linewidth=0)
        ax.set_ylabel(f"{label}\nper minute")
        ax.set_ylim(0, counts.max() * 1.12)
        et_axis(ax, session, edges[0], edges[-1] + 60.0, label_marks=ax is axes[0])
    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[0].set_title(title, loc="left")
    fig.align_ylabels(axes)
    return fig


def branching_ratio_figure(
    curves: list[dict],
    session: Session,
    title: str,
    ylabel: str = "branching ratio n",
) -> plt.Figure:
    """n(t) curves with confidence bands. Each curve: {name, t, n, lo, hi, flagged (bool array, optional)}."""
    style()
    fig, ax = plt.subplots(figsize=(11, 4))
    for curve, color in zip(curves, SERIES):
        t, n = np.asarray(curve["t"]), np.asarray(curve["n"])
        if "lo" in curve and "hi" in curve:
            ax.fill_between(t, curve["lo"], curve["hi"], color=color, alpha=0.18, linewidth=0)
        ax.plot(t, n, color=color, label=curve["name"])
        flagged = np.asarray(curve.get("flagged", np.zeros(len(t), dtype=bool)))
        if flagged.any():
            ax.plot(t[flagged], n[flagged], linestyle="none", marker="x", color=color, markersize=5, label=f"{curve['name']}: KS reject (5%)")
    ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.05, ax.get_ylim()[1]))
    et_axis(ax, session)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", ncol=2)
    return fig


def recovery_figure(table: pd.DataFrame, title: str) -> plt.Figure:
    """Simulation recovery: bias of n-hat with +/-1 SD bars versus true n; rows are true kernels, columns fitters."""
    style()
    kernels = list(dict.fromkeys(table["kernel"]))
    fitters = list(dict.fromkeys(table["fitter"]))
    sizes = sorted(table["events"].unique())
    fig, axes = plt.subplots(len(kernels), len(fitters), figsize=(2.6 * len(fitters), 2.3 * len(kernels)), sharex=True, sharey=True, squeeze=False)
    for i, kernel in enumerate(kernels):
        for j, fitter in enumerate(fitters):
            ax = axes[i, j]
            sub = table[(table["kernel"] == kernel) & (table["fitter"] == fitter)]
            ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=":")
            for size, color, shift in zip(sizes, SERIES, np.linspace(-0.03, 0.03, len(sizes))):
                s = sub[sub["events"] == size].sort_values("n_true")
                ax.errorbar(s["n_true"] + shift, s["bias"], yerr=s["sd"], color=color, marker="o", markersize=3.5, linestyle="none", capsize=2, label=f"{size} events")
            if i == 0:
                ax.set_title(fitter, loc="left", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"truth: {kernel}\nbias of n-hat")
            if i == len(kernels) - 1:
                ax.set_xlabel("true n")
            ax.set_xlim(0.2, 1.0)
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig
