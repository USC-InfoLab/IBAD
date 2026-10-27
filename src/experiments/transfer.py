"""Experiment — do behavioural templates transfer from one dataset to another?

Templates are fitted on a *source* dataset (NHTS, a US travel survey) and used,
without refitting, on a *target* dataset (Geolife, GPS traces from Beijing).
Two questions:

1. **Alignment.** Do the source's templates resemble the ones the target yields
   on its own? Each template is a weighting of (time slot, activity) cells.
   On the activities both datasets record, every source template is matched to
   one target template so the total cosine similarity is maximal (Hungarian
   matching), and the mean matched similarity is the statistic. Its null
   distribution comes from shuffling each target template's cells
   independently, which keeps each template's weights but breaks the time
   slot and activity they sit at, and with them any correspondence with the
   source (the paper's Appendix C.2).

2. **Utility.** Is IBAD as good an anomaly detector on the target with the
   source's templates as with the target's own? IBAD is trained on the target
   with each day's mixture over the source's templates and evaluated exactly as
   the main runs are, on a benchmark drawn with those same mixtures.

Run from the repo root:
    python -m src.experiments.transfer
    python -m src.experiments.transfer --part alignment
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.detection.data import load_cohort
from src.detection.run import CONFIG_PATH, evaluate, fitted_detector
from src.experiments import plots
from src.templates import artifacts
from src.utils.config import ROOT, load_cfg

EXPERIMENT_CONFIG = "configs/experiments/transfer.yaml"


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def shared_templates(saved_a: dict, saved_b: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Both artifacts' templates over the (slot, activity) cells they share.

    Returns the ``(K_a, n_slots * n_shared)`` and ``(K_b, ...)`` topic-word
    weights of the two LDA models, slot-major, and the shared activity names
    in sorted order. The weights are left unnormalised: cosine similarity
    ignores each template's scale, and shuffling a template's cells keeps it.
    """
    labels_a, labels_b = saved_a["vocab"]["labels"], saved_b["vocab"]["labels"]
    if saved_a["vocab"]["n_slots"] != saved_b["vocab"]["n_slots"]:
        raise ValueError("the two artifacts use different slot grids")
    shared = sorted(set(labels_a) & set(labels_b))

    def restrict(saved: dict, labels: list[str]) -> np.ndarray:
        phi = saved["lda"].components_
        cube = phi.reshape(len(phi), saved["vocab"]["n_slots"], len(labels))
        return cube[:, :, [labels.index(a) for a in shared]].reshape(len(phi), -1)

    return restrict(saved_a, labels_a), restrict(saved_b, labels_b), shared


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - cdist(a, b, metric="cosine")


def _matched(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian matching maximising total similarity: rows, columns, matched similarities."""
    rows, cols = linear_sum_assignment(-similarity)
    return rows, cols, similarity[rows, cols]


def _top_activities(template: np.ndarray, n_slots: int, shared: list[str], k: int = 3) -> str:
    """The template's ``k`` largest activities by share of its weight."""
    mass = template.reshape(n_slots, len(shared)).sum(axis=0)
    mass = mass / mass.sum()
    top = np.argsort(mass)[::-1][:k]
    return ", ".join(f"{shared[i]} {mass[i]:.0%}" for i in top)


def alignment(exp: dict, cfg: dict) -> dict:
    """Matched-similarity test between the source's and the target's templates."""
    params = exp["alignment"]
    source = artifacts.load(ROOT / cfg["templates_dir"], exp["source"])
    target = artifacts.load(ROOT / cfg["templates_dir"], exp["target"])
    a, b, shared = shared_templates(source, target)

    similarity = _cosine(a, b)
    rows, cols, matched = _matched(similarity)
    observed = float(matched.mean())

    rng = np.random.default_rng(params["seed"])
    null = np.array([
        _matched(_cosine(a, np.stack([rng.permutation(template) for template in b])))[2].mean()
        for _ in range(params["n_permutations"])
    ])
    # Add-one so a p-value is never reported as exactly zero.
    p_value = (1 + int((null >= observed).sum())) / (1 + len(null))
    n_slots = source["vocab"]["n_slots"]

    pairs = pd.DataFrame({
        "source_template": rows, "target_template": cols, "cosine": matched,
        "source_top_activities": [_top_activities(a[r], n_slots, shared) for r in rows],
        "target_top_activities": [_top_activities(b[c], n_slots, shared) for c in cols],
    })
    return {
        "source": exp["source"], "target": exp["target"],
        "shared_activities": shared,
        "mean_matched_cosine": observed,
        # The spread of the permutation distribution itself (population sd).
        "null_mean": float(null.mean()), "null_sd": float(null.std()),
        "z": float((observed - null.mean()) / null.std()),
        "p_value": p_value, "n_permutations": len(null),
        "pairs": pairs, "null": null,
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def checkpoint_path(exp: dict, seed: int) -> Path:
    """Where the method fitted on the transferred templates is saved."""
    return (ROOT / exp["artifacts_dir"]
            / f"{exp['target']}_{exp['method']}_{exp['source']}_seed{seed}.pt")


def utility(exp: dict, cfg: dict, seeds: list[int], device: torch.device,
            retrain: bool = False) -> pd.DataFrame:
    """The method on the target with its own templates and with the source's.

    Each arm is evaluated exactly as the main runs are: its benchmark is drawn
    with the template mixtures it forecasts. The native arm is the main runs'
    saved model; only the transferred arm is fitted (or loaded) here.
    """
    dataset, method = exp["target"], exp["method"]
    cohorts = {"native": load_cohort(dataset, cfg),
               "transferred": load_cohort(dataset, cfg, templates=exp["source"])}

    rows = []
    for seed in seeds:
        print(f"\n=== transfer · {exp['source']} -> {dataset} · seed {seed} ===")
        for arm, cohort in cohorts.items():
            transferred = arm == "transferred"
            detector, _ = fitted_detector(
                method, dataset, cohort, cfg, seed, device, retrain=retrain and transferred,
                checkpoint=checkpoint_path(exp, seed) if transferred else None,
            )
            for policy, metrics in evaluate(detector, cohort, cfg["datasets"][dataset],
                                            seed).items():
                rows.append({
                    "arm": arm, "templates": exp["source"] if transferred else dataset,
                    "dataset": dataset, "seed": seed, "policy": policy,
                    "auroc": metrics["auroc"], "ap": metrics["ap"], "f1": metrics["f1"],
                })
                print(f"  {policy:9s} {arm:12s} AUROC={metrics['auroc']:.4f} "
                      f"AP={metrics['ap']:.4f}")
    return pd.DataFrame(rows)


def utility_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per policy: each arm's mean ± sd, and the paired per-seed difference."""
    out = []
    for policy, group in frame.groupby("policy"):
        wide = group.pivot(index="seed", columns="arm", values=["auroc", "ap"])
        row = {"policy": policy}
        for metric in ("auroc", "ap"):
            for arm in ("native", "transferred"):
                row[f"{metric}_{arm}"] = plots.mean_sd(wide[(metric, arm)].to_numpy())
            diff = (wide[(metric, "transferred")] - wide[(metric, "native")]).to_numpy()
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=EXPERIMENT_CONFIG)
    parser.add_argument("--part", choices=("alignment", "utility", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--retrain", action="store_true",
                        help="refit IBAD on the transferred templates even when saved")
    args = parser.parse_args()

    exp = load_cfg(args.config)
    cfg = load_cfg(CONFIG_PATH)
    out_dir = ROOT / exp["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.part in ("alignment", "both"):
        result = alignment(exp, cfg)
        result["pairs"].to_csv(out_dir / "alignment_pairs.csv", index=False)
        summary = {k: v for k, v in result.items() if k not in ("pairs", "null")}
        (out_dir / "alignment.json").write_text(json.dumps(summary, indent=2))
        plots.alignment_pairs_figure(result, out_dir / "alignment_pairs")
        plots.alignment_null_figure(result, out_dir / "alignment_null")
        print(f"\nmean matched cosine {result['mean_matched_cosine']:.3f}  "
              f"(null {result['null_mean']:.3f} ± {result['null_sd']:.3f}, "
              f"z = {result['z']:.1f}, p = {result['p_value']:.4f})")
        print(result["pairs"].round(3).to_string(index=False))

    if args.part in ("utility", "both"):
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        frame = utility(exp, cfg, args.seeds or cfg["replicate_seeds"], device, args.retrain)
        frame.to_csv(out_dir / "utility.csv", index=False)
        table = utility_summary(frame)
        table.to_csv(out_dir / "utility_summary.csv", index=False)
        print("\n" + table.to_string(index=False))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
