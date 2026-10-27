"""Module 2 — fit one detector on one dataset and score its benchmark.

Run from the repo root:
    python -m src.detection.run --method ibad --dataset geolife
    python -m src.detection.run --method ibad --dataset geolife --seed 43
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.detection.data import Cohort, load_cohort
from src.detectors import Detector, detector_class
from src.detectors.base import for_dataset
from src.evaluation import benchmark, scoring
from src.utils.config import ROOT, load_cfg

CONFIG_PATH = "configs/detection/benchmark.yaml"
DATASETS = ("geolife", "numosim")


def checkpoint_path(cfg: dict, method: str, dataset: str, seed: int) -> Path:
    ext = detector_class(method).checkpoint_ext
    return ROOT / cfg["artifacts_dir"] / f"{dataset}_{method}_seed{seed}{ext}"


def build_detector(
    method: str, dataset: str, cohort: Cohort, seed: int, device: torch.device
) -> Detector:
    cls = detector_class(method)
    method_cfg = for_dataset(load_cfg(cls.config_path), dataset) if cls.config_path else {}
    return cls(method_cfg, cohort, seed, device)


def fitted_detector(
    method: str,
    dataset: str,
    cohort: Cohort,
    cfg: dict,
    seed: int,
    device: torch.device,
    retrain: bool = False,
    checkpoint: Path | None = None,
) -> tuple[Detector, dict]:
    """A detector for (method, dataset, seed): loaded if saved, else fitted and saved.

    ``checkpoint`` overrides the default location, for experiments that fit the
    same method on a variant of the dataset's cohort.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    detector = build_detector(method, dataset, cohort, seed, device)

    checkpoint = checkpoint or checkpoint_path(cfg, method, dataset, seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not retrain:
        print(f"  loading {checkpoint.relative_to(ROOT)}")
        detector.load(checkpoint)
        return detector, {"loaded": True}
    summary = detector.fit()
    detector.save(checkpoint)
    return detector, {"loaded": False, **summary}


def evaluate(detector: Detector, cohort: Cohort, dataset_cfg: dict, seed: int) -> dict:
    """Score the splice benchmark: metrics per policy."""
    window = detector.scored_window(cohort)
    sequences = window.sequences
    splice = dataset_cfg["splice"]

    def flat(policy: str, split: int):
        bench = benchmark.spliced(sequences, policy, split, seed, splice["min_shift"],
                                  splice["agent_frac"], splice["donors"])
        if split == 2:
            n_spliced = int((bench.plan[:, 0] >= 0).sum())
            print(f"  {policy}: {n_spliced}/{sequences.n_agents} agents spliced in test")
        return scoring.evaluate(detector, window, bench)

    # Splices in validation tune the F1 threshold; a dataset without
    # validation days reports AUROC and AP only.
    has_val = bool((sequences.split == 1).any())
    return {policy: scoring.metrics(flat(policy, 1) if has_val else None, flat(policy, 2))
            for policy in benchmark.POLICIES}


def run(
    method: str,
    dataset: str,
    cfg: dict,
    seed: int,
    device: torch.device,
    retrain: bool = False,
) -> dict:
    """Full pipeline for one (method, dataset, seed): cohort, fit, evaluation."""
    print(f"\n=== {method} · {dataset} · seed {seed} ===")
    cohort = load_cohort(dataset, cfg)
    sequences = cohort.sequences
    print(f"  {sequences.n_agents} agents x up to {sequences.n_days} days, "
          f"K={sequences.n_templates}")

    detector, fit = fitted_detector(method, dataset, cohort, cfg, seed, device, retrain)
    per_policy = evaluate(detector, cohort, cfg["datasets"][dataset], seed)

    result = {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "n_agents": int(sequences.n_agents),
        "n_templates": int(sequences.n_templates),
        "fit": fit,
        "per_policy": per_policy,
    }
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{dataset}_{method}_seed{seed}.json").write_text(json.dumps(result, indent=2))

    for policy, metrics in per_policy.items():
        print(f"  {policy:9s} AUROC={metrics['auroc']:.4f} "
              f"AP={metrics['ap']:.4f} F1={metrics['f1']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="ibad")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run(args.method, args.dataset, cfg, args.seed or cfg["seed"], device, retrain=args.retrain)


if __name__ == "__main__":
    main()
