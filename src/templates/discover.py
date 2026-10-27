"""Module 1 — end-to-end template discovery for one dataset.

Run from the repo root:
    python -m src.templates.discover --dataset geolife
    python -m src.templates.discover --dataset geolife --sweep
"""

import argparse
from pathlib import Path

import numpy as np

from src.features import build_documents
from src.features.documents import Documents
from src.templates import artifacts, diagnostics, lda, plots
from src.utils.config import ROOT, load_cfg

CONFIG_PATH = "configs/templates/default.yaml"
DATASETS = ("nhts", "numosim", "geolife")


def batch_size_for(cfg: dict, dataset: str) -> int:
    """Per-dataset batch size, falling back to the shared LDA setting."""
    return cfg["datasets"][dataset].get("batch_size", cfg["lda"]["batch_size"])


def k_values(cfg: dict) -> list[int]:
    return list(range(cfg["k_sweep"]["k_min"], cfg["k_sweep"]["k_max"] + 1))


def fitting_matrix(documents: Documents, dataset_cfg: dict, seed: int):
    """The matrix to fit on — a subsample when the corpus is large enough."""
    subsample = dataset_cfg.get("subsample")
    if not subsample or documents.n_docs <= subsample:
        return documents.X

    rows = np.random.default_rng(seed).choice(
        documents.n_docs, size=subsample, replace=False
    )
    print(f"  fitting on a {subsample:,}-document subsample of {documents.n_docs:,}")
    return documents.X[rows]


def run_sweep(documents: Documents, cfg: dict, output_dir: Path):
    """Perplexity elbow over the configured K range."""
    dataset_cfg = cfg["datasets"][documents.name]
    sweep = lda.sweep_k(
        fitting_matrix(documents, dataset_cfg, cfg["seed"]),
        k_values(cfg),
        seed=cfg["seed"],
        max_iter=cfg["lda"]["max_iter"],
        batch_size=batch_size_for(cfg, documents.name),
    )
    sweep.to_csv(output_dir / f"{documents.name}_k_sweep.csv", index=False)
    plots.plot_perplexity_elbow(
        sweep,
        title=f"{documents.name.upper()} — LDA perplexity by K",
        output_path=output_dir / f"{documents.name}_k_sweep.png",
    )
    return sweep


def discover(
    dataset: str,
    cfg: dict,
    documents: Documents | None = None,
    k: int | None = None,
    make_plots: bool = True,
) -> dict:
    """Fit templates for one dataset, then save artifacts and figures.

    Returns the fitted model, the per-document mixtures, the topic-word matrix,
    and the membership diagnostics.
    """
    dataset_cfg = cfg["datasets"][dataset]
    k = k or dataset_cfg["k"]
    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if documents is None:
        documents = build_documents(dataset, cfg)
    print(
        f"{dataset}: {documents.n_docs:,} documents  "
        f"{documents.n_acts} activities  {documents.n_slots} slots"
    )

    model = lda.fit(
        fitting_matrix(documents, dataset_cfg, cfg["seed"]),
        k,
        seed=cfg["seed"],
        max_iter=cfg["lda"]["max_iter"],
        batch_size=batch_size_for(cfg, dataset),
    )
    theta = lda.transform(model, documents.X)
    phi = lda.topic_word_matrix(model)

    stats = diagnostics.membership_stats(theta)
    print("  " + "  ".join(f"{name}={value:.3f}" for name, value in stats.items()))

    path = artifacts.save(ROOT / cfg["artifacts_dir"], documents, model, theta, k)
    print(f"  saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")

    if make_plots:
        plots.plot_topic_heatmaps(
            phi,
            documents,
            title=f"{dataset.upper()} — LDA templates (K={k})",
            output_path=output_dir / f"{dataset}_templates.png",
            sort_activities=dataset == "numosim",
            gamma=0.7 if dataset in ("numosim", "geolife") else 0.55,
        )
        plots.plot_membership(
            theta,
            title=f"{dataset.upper()} — membership diagnostics",
            output_path=output_dir / f"{dataset}_membership.png",
        )

    return {
        "documents": documents,
        "model": model,
        "theta": theta,
        "phi": phi,
        "k": k,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--k", type=int, help="override the configured K")
    parser.add_argument(
        "--sweep", action="store_true", help="also run the perplexity K sweep"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    documents = build_documents(args.dataset, cfg)

    if args.sweep:
        run_sweep(documents, cfg, ROOT / cfg["output_dir"])

    discover(
        args.dataset,
        cfg,
        documents=documents,
        k=args.k,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
