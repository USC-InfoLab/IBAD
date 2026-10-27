"""Figures for the experiments.

Each draws its paper figure in the paper's layout, sizes, fonts and colours:
the ε sweep (Figure 4) and the template alignment (Figure 5a, 5b).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
from scipy.stats import gaussian_kde

from src.detectors import detector_class

# Fixed method -> colour, so a method keeps its colour in every panel and run.
# IBAD is black (the subject); the sequence baselines take saturated hues, the
# θ-space ones muted hues, with HistMean grey as the plain persistence floor.
METHOD_COLOURS = {
    "ibad": "black",
    "tod4traj": "#F58518",
    "bestad": "#54A24B",
    "icad": "#B279A2",
    "trajgen": "#E45756",
    "histmean": "#BAB0AC",
    "ghmm": "#72B7B2",
    "iforest": "#EECA3B",
    "ocsvm": "#9D7660",
    "deepsvdd": "#FF9DA6",
}
FALLBACK_COLOURS = ("#1B9E77", "#7570B3", "#D95F02", "#66A61E", "#E7298A")
DATASET_NAMES = {"nhts": "NHTS", "geolife": "GeoLife", "numosim": "NUMOSIM"}
# The alignment figures are set in a serif face with Computer Modern maths.
ALIGNMENT_STYLE = {"font.family": "serif", "font.size": 8, "axes.linewidth": 0.8,
                   "mathtext.fontset": "cm"}


def _save(fig, out_stem: Path | None) -> None:
    """Write the figure as PNG and PDF next to each other, when a path is given."""
    if out_stem is None:
        return
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")


def _colours(methods: list[str]) -> dict[str, str]:
    """Method -> colour: the fixed table, cycling the fallbacks for anything unlisted."""
    colours, spare = {}, 0
    for method in methods:
        if method in METHOD_COLOURS:
            colours[method] = METHOD_COLOURS[method]
        else:
            colours[method] = FALLBACK_COLOURS[spare % len(FALLBACK_COLOURS)]
            spare += 1
    return colours


def _place_end_labels(ax, points: list[tuple[str, float, float, str]],
                      fontsize: float = 6.8) -> None:
    """Write each method's name just past the end of its own curve, instead of a legend.

    Labels that would overlap are pushed apart vertically, but only among curves
    ending at the same x: a curve that stops early is labelled where it ends.
    """
    low, high = ax.get_ylim()
    gap = 0.075 * (high - low)
    by_x: dict[float, list[int]] = {}
    for i, (_, x, _, _) in enumerate(points):
        by_x.setdefault(x, []).append(i)

    ys = [p[2] for p in points]
    for idxs in by_x.values():
        order = sorted(idxs, key=lambda i: -ys[i])
        for prev, cur in zip(order, order[1:]):
            if ys[prev] - ys[cur] < gap:
                ys[cur] = ys[prev] - gap
        top, bottom = ys[order[0]], ys[order[-1]]
        if bottom < low:                   # the stack slid off the bottom
            for i in idxs:
                ys[i] += low - bottom
        elif top > high:
            for i in idxs:
                ys[i] -= top - high

    for (name, x, _, colour), y in zip(points, ys):
        ax.annotate(name, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=fontsize, color=colour,
                    fontweight="bold" if name == "IBAD" else "normal",
                    annotation_clip=False)


def threshold_figure(tables: dict[str, pd.DataFrame], methods: dict[str, list[str]],
                     policy: str, out_stem: Path | None = None,
                     row_height: float = 1.2, share_y: bool = False):
    """AUROC | AP against ε, one row per dataset, mean ± 1 sd over seeds.

    ``methods`` maps each dataset to the methods its row draws, in label order;
    the rows follow its order. A method with no results is left out (with a
    note), not drawn empty. Each row is keyed by names written at the ends of
    its curves in the AP panel rather than by a legend.

    ε sits at evenly spaced positions labelled with its value: the grid is
    uneven (0, 0.05, 0.1, 0.25), and a linear axis would crowd the low values.
    """
    datasets = [d for d in methods if d in tables]
    drawn = {}
    for dataset in datasets:
        have = set(tables[dataset].method)
        missing = [m for m in methods[dataset] if m not in have]
        if missing:
            names = ", ".join(detector_class(m).display_name for m in missing)
            print(f"NB: {dataset}: no results for {names} — not drawn.")
        drawn[dataset] = [m for m in methods[dataset] if m in have]

    # Colours over the union, so a method in both rows keeps one colour.
    union = list(dict.fromkeys(m for d in datasets for m in drawn[d]))
    colour = _colours(["ibad"] * ("ibad" in union) + [m for m in union if m != "ibad"])
    eps_values = sorted({e for d in datasets for e in tables[d].eps.unique()})
    position = {e: i for i, e in enumerate(eps_values)}

    fig, axes = plt.subplots(len(datasets), 2, figsize=(6.2, row_height * len(datasets) + 0.55),
                             squeeze=False, sharex=True, sharey="col" if share_y else False)
    for row, dataset in enumerate(datasets):
        frame = tables[dataset][tables[dataset].policy == policy]
        for col, metric in enumerate(("auroc", "ap")):
            ax = axes[row][col]
            ends = []
            for method in reversed(drawn[dataset]):          # IBAD drawn last, on top
                stats = frame[frame.method == method].groupby("eps")[metric].agg(["mean", "std"])
                is_ibad = method == "ibad"
                xs = [position[e] for e in stats.index]
                ax.errorbar(xs, stats["mean"], yerr=stats["std"].fillna(0.0),
                            marker="o", markersize=3.0 if is_ibad else 2.2,
                            linewidth=1.8 if is_ibad else 1.0, elinewidth=0.55, capsize=1.2,
                            alpha=1.0 if is_ibad else 0.85, color=colour[method],
                            zorder=5 if is_ibad else 3)
                ends.append((detector_class(method).display_name, float(xs[-1]),
                             float(stats["mean"].iloc[-1]), colour[method]))
            ax.set_xticks([position[e] for e in eps_values])
            ax.set_xticklabels([f"{e:g}" for e in eps_values])
            ax.tick_params(labelsize=6.8, length=2.5, pad=1.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.margins(x=0.04)
            ax.grid(alpha=0.22, linestyle=":", linewidth=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if metric == "auroc":
                low, high = ax.get_ylim()
                if low <= 0.5 <= high:             # chance, only when already in view
                    ax.axhline(0.5, color="0.6", linewidth=0.7, linestyle=":", zorder=0)
            else:
                _place_end_labels(ax, ends)        # only the right panel has margin
            if row == 0:
                ax.set_title(metric.upper(), fontsize=8, pad=3)
        axes[row][0].set_ylabel(dataset.upper(), fontsize=8, fontweight="bold", labelpad=2)

    fig.supxlabel(r"splice threshold  $\epsilon$", fontsize=8, y=0.015)
    fig.tight_layout(pad=0.35, w_pad=0.7, h_pad=0.45, rect=(0, 0.03, 1, 1))
    _save(fig, out_stem)
    return fig


def _names(result: dict) -> tuple[str, str]:
    return tuple(DATASET_NAMES.get(result[k], result[k]) for k in ("source", "target"))


def alignment_pairs_figure(result: dict, out_stem: Path | None = None):
    """Cosine similarity of each matched (source, target) template pair, best on top."""
    source, target = _names(result)
    pairs = result["pairs"]
    order = np.argsort(pairs["cosine"].to_numpy())
    sims = pairs["cosine"].to_numpy()[order]
    ypos = np.arange(len(sims))

    with plt.rc_context(ALIGNMENT_STYLE):
        fig, ax = plt.subplots(figsize=(4.0, 0.38 * len(sims) + 1.0), dpi=100)
        colours = plt.cm.YlGn(0.25 + 0.6 * (sims - sims.min()) / (np.ptp(sims) + 1e-12))
        ax.barh(ypos, sims, height=0.62, color=colours, edgecolor="#3a3a3a", linewidth=0.5,
                zorder=3)
        for y, s in zip(ypos, sims):
            ax.text(s + 0.015, y, f"{s:.2f}", va="center", ha="left", fontsize=6.5, color="#222")
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"T{a}  $\\leftrightarrow$  T{b}" for a, b in zip(
            pairs["source_template"].to_numpy()[order], pairs["target_template"].to_numpy()[order])],
            fontsize=6.5)
        ax.set_xlim(0, min(1.0, sims.max() + 0.14))
        ax.set_xlabel(f"{source} $\\leftrightarrow$ {target} pairwise template similarity.",
                      fontsize=7, labelpad=2)
        ax.tick_params(axis="x", labelsize=6.5, pad=1.5)
        ax.grid(axis="x", ls="--", alpha=0.3, lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.5)
        _save(fig, out_stem)
    return fig


def alignment_null_figure(result: dict, out_stem: Path | None = None):
    """The permutation null of the mean matched similarity against the observed value.

    The x-axis is logarithmic, so the null's spread and the far-off observed
    value fit one panel; the null's top 5% is shaded as the critical region.
    """
    source, target = _names(result)
    null, observed = result["null"], result["mean_matched_cosine"]
    chance = float(null.mean())
    critical = np.percentile(null, 95)
    x_lo = max(null.min() * 0.85, 1e-3)
    x_hi = max(null.max(), observed) * 1.15

    with plt.rc_context(ALIGNMENT_STYLE):
        fig, ax = plt.subplots(figsize=(4.0, 3.1), dpi=100)
        bins = np.logspace(np.log10(null.min()), np.log10(null.max()), 41)
        _, edges, patches = ax.hist(null, bins=bins, density=True, color="#9fb2d8",
                                    edgecolor="white", linewidth=0.4, alpha=0.9, zorder=2)
        for patch, left in zip(patches, edges[:-1]):
            if left >= critical:
                patch.set_facecolor("#d98a8a")
        grid = np.logspace(np.log10(x_lo), np.log10(x_hi), 400)
        ax.plot(grid, gaussian_kde(null)(grid), color="#2f3e6e", lw=1.5, zorder=3)
        ax.axvspan(critical, x_hi, color="#d98a8a", alpha=0.12, zorder=1)
        ax.axvline(chance, color="#2f3e6e", lw=1.1, ls="-.", alpha=0.7, zorder=3)
        ax.axvline(observed, color="#b3122b", lw=1.9, ls="--", zorder=5)

        y_hi = ax.get_ylim()[1]
        ax.set_ylim(0, y_hi * 1.32)
        ax.text(critical + (x_hi - critical) * 0.08, y_hi * 0.32,
                "Top 5% (critical region, $p<0.05$)",
                color="#b3122b", fontsize=6, va="center", ha="left")
        ax.annotate("Chance level\n(random alignment)",
                    xy=(chance, y_hi * 1.02), xytext=(chance * 2.0, y_hi * 1.2),
                    ha="left", va="center", fontsize=6, color="#2f3e6e",
                    arrowprops=dict(arrowstyle="->", color="#2f3e6e", lw=1.0, relpos=(0.0, 0.5)))
        ax.annotate(f"Observed\nalignment = {observed:.3f}",
                    xy=(observed, y_hi * 0.55), xytext=(observed * 0.55, y_hi * 0.9),
                    ha="center", va="center", fontsize=6.5, color="#b3122b", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#b3122b", lw=1.3))

        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=6.5, pad=1.5)
        ax.set_xlabel("Mean matched template similarity (log scale)", fontsize=7, labelpad=2)
        ax.set_ylabel("Density", fontsize=7, labelpad=2)
        ax.set_title(f"Aggregate alignment vs. chance: {source} to {target}",
                     fontsize=8, fontweight="bold", pad=6)
        ax.grid(axis="y", ls="--", alpha=0.3, lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.5)
        _save(fig, out_stem)
    return fig


def mean_sd(values: np.ndarray) -> str:
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"
