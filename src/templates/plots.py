"""Figures for inspecting discovered templates."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm

from src.features.documents import Documents

# Activities that appear across datasets are pinned to one colour so the same
# behaviour reads the same way in every figure. NUMOSIM bundles are pinned to
# the colour of the activity that dominates them.
SHARED_COLORS = {
    "Home": "C0",
    "DropOff+Home+Visit": "C0",
    "Education": "C1",
    "DropOff+School+Work": "C1",
    "Work": "tab:red",
    "DropOff+Work": "tab:red",
    "Shopping": "navy",
    "Recreation": "tab:green",
    "Recreation+Work": "tab:green",
    "Transportation": "tab:purple",
    "Religious": "tab:purple",
    "Restaurant": "tab:olive",
    "EatOut+Work": "tab:olive",
    "Excercise": "tab:pink",
    "Leisure": "tab:pink",
}


def _save(fig, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")


def activity_colors(vocabularies: list[list[str]]) -> dict[str, object]:
    """Build one colour per activity across every dataset vocabulary."""
    every_label = sorted({label for vocab in vocabularies for label in vocab})
    palette = plt.get_cmap("tab20")
    colors = {label: palette(i % 20) for i, label in enumerate(every_label)}
    colors.update(
        {label: c for label, c in SHARED_COLORS.items() if label in colors}
    )
    return colors


def plot_perplexity_elbow(
    sweep: pd.DataFrame, title: str, output_path: str | Path | None = None
):
    """Perplexity against K — the elbow guides the choice of K."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(sweep["k"], sweep["perplexity"], "o-", color="steelblue")
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.6)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_topic_heatmaps(
    phi: np.ndarray,
    documents: Documents,
    title: str,
    output_path: str | Path | None = None,
    gamma: float = 0.55,
    sort_activities: bool = False,
    cmap: str = "YlOrRd",
    max_label_chars: int = 22,
):
    """One heatmap per topic: activities (rows) against time of day (columns).

    Each panel is normalised by its own peak, so panels show shape rather than
    relative topic mass. ``sort_activities`` orders rows by peak time, which
    makes patterns read as diagonals when the vocabulary is large.
    """
    n_slots, n_acts = documents.n_slots, documents.n_acts
    k = phi.shape[0]

    schedule = np.clip(phi.reshape(k, n_slots, n_acts), 0, None)
    peak = schedule.max(axis=(1, 2), keepdims=True)
    schedule = np.divide(
        schedule, peak, out=np.zeros_like(schedule), where=peak > 0
    )

    labels = list(documents.labels)
    if sort_activities and n_acts > 1:
        mass = schedule.sum(axis=0)
        totals = mass.sum(axis=0)
        centroid = np.where(
            totals > 0,
            (mass * np.arange(n_slots)[:, None]).sum(axis=0) / np.maximum(totals, 1e-12),
            n_slots + 1,          # activities with no mass sink to the bottom
        )
        order = np.argsort(centroid)
        schedule = schedule[:, :, order]
        labels = [labels[i] for i in order]

    labels = [
        s if len(s) <= max_label_chars else s[: max_label_chars - 1] + "…"
        for s in labels
    ]

    ncols = min(k, 3)
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 6.2, nrows * max(3.8, 0.22 * n_acts + 1.4)),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()

    hours = list(range(0, 25, 4))
    tick_positions = [h * n_slots / 24 - 0.5 for h in hours]
    label_size = max(6, min(10, int(220 / max(n_acts, 1))))

    image = None
    for topic in range(k):
        ax = axes[topic]
        image = ax.imshow(
            schedule[topic].T,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            norm=PowerNorm(gamma=gamma, vmin=0, vmax=1),
            interpolation="nearest",
            extent=(-0.5, n_slots - 0.5, -0.5, n_acts - 0.5),
        )
        ax.set_yticks(range(n_acts))
        ax.set_yticklabels(labels, fontsize=label_size)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{h:02d}:00" for h in hours], fontsize=9)
        ax.set_xlabel("Hour of day", fontsize=9)
        ax.set_title(f"Template {topic}", fontsize=11, fontweight="bold")
        ax.tick_params(axis="both", length=2)
        for x in tick_positions[1:-1]:
            ax.axvline(x, color="white", linewidth=0.4, alpha=0.45)

    for unused in range(k, len(axes)):
        axes[unused].set_visible(False)

    bar = fig.colorbar(image, ax=axes.tolist(), shrink=0.75, pad=0.015, aspect=30)
    bar.set_label("Relative intensity (per template)", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    _save(fig, output_path)
    return fig


def plot_topic_sparklines(
    phi: np.ndarray,
    documents: Documents,
    title: str,
    output_path: str | Path | None = None,
    colors: dict | None = None,
    top_activities: int = 3,
    min_mass: float = 0.05,
):
    """Paper-ready view: the few activities that define each template over time."""
    k = phi.shape[0]
    n_slots, n_acts = documents.n_slots, documents.n_acts

    profiles = phi.reshape(k, n_slots, n_acts)
    profiles = profiles / profiles.sum(axis=(1, 2), keepdims=True)

    fig, axes = plt.subplots(1, k, figsize=(22, 4.0), sharey=True, squeeze=False)
    axes = axes.ravel()
    slots = np.arange(n_slots)

    for topic in range(k):
        ax = axes[topic]
        mass = profiles[topic].sum(axis=0)
        ranked = np.argsort(mass)[-top_activities:][::-1]

        labelled = False
        for activity in ranked:
            if mass[activity] <= min_mass:
                continue
            name = documents.labels[activity]
            ax.plot(
                slots,
                profiles[topic, :, activity],
                label=name,
                lw=2.0,
                color=(colors or {}).get(name),
            )
            labelled = True

        hours = [0, 4, 8, 12, 16, 20]
        ax.set_xticks([int(h / 24 * n_slots) for h in hours] + [n_slots - 1])
        ax.set_xticklabels(
            [f"{h:02d}:00" for h in hours] + ["23:59"],
            fontsize=10,
            rotation=45,
            ha="right",
        )
        ax.set_title(f"Template {topic}", fontsize=12, fontweight="bold", pad=4)
        ax.grid(True, linestyle=":", alpha=0.6, color="gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if labelled:
            ax.legend(loc="best", fontsize=10, framealpha=0.9, handlelength=1.2)

    axes[0].set_ylabel("PMF", fontsize=14)
    fig.suptitle(title, fontsize=16, fontweight="semibold", y=1.05)
    fig.subplots_adjust(wspace=0.15, bottom=0.2, left=0.06, right=0.98, top=0.85)
    _save(fig, output_path)
    return fig


def plot_membership(
    theta: np.ndarray, title: str, output_path: str | Path | None = None
):
    """Distribution of membership entropy and of the top topic weight."""
    k = theta.shape[1]
    max_prob = theta.max(axis=1)
    entropy = -np.sum(theta * np.log(theta + 1e-12), axis=1)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6))

    left.hist(entropy, bins=50, color="steelblue", density=True)
    left.axvline(np.log(k), color="red", ls="--", label=f"log({k}) = fully mixed")
    left.set_xlabel("Membership entropy")
    left.set_ylabel("Density")
    left.set_title(f"Entropy (mean {entropy.mean():.2f})")
    left.legend(fontsize=8)

    right.hist(max_prob, bins=50, color="darkorange", density=True)
    right.axvline(0.9, color="red", ls="--", label="0.9 threshold")
    right.set_xlabel("Top template weight")
    right.set_title(f"Top weight ({(max_prob > 0.9).mean():.1%} of days > 0.9)")
    right.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save(fig, output_path)
    return fig
