"""Run detectors across several seeds and report mean +/- sd per method.

Each replicate fits its own detector and, on a spliced benchmark, draws its own
splices, so the spread covers both model variance and benchmark variance — the
honest error bar for "how well does this method do on this task". All methods
at a given seed score identical benchmarks, so the table is paired by seed.

Usage:
    python scripts/anomaly_detection/run_replicates.py --dataset geolife
    python scripts/anomaly_detection/run_replicates.py --dataset numosim --methods ibad --seeds 42 43
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.detection.run import CONFIG_PATH, DATASETS, run  # noqa: E402
from src.detectors import REGISTRY, detector_class  # noqa: E402
from src.utils.config import ROOT, load_cfg  # noqa: E402

KEY = ["method", "dataset", "seed", "policy"]


def rows_from(result: dict) -> list[dict]:
    """One row per policy."""
    return [
        {
            "method": result["method"],
            "dataset": result["dataset"],
            "seed": result["seed"],
            "policy": policy,
            "auroc": metrics["auroc"],
            "ap": metrics["ap"],
            "f1": metrics["f1"],
        }
        for policy, metrics in result["per_policy"].items()
    ]


def collect(output_dir: Path) -> pd.DataFrame:
    """Every saved run result, one row per (method, dataset, seed, policy)."""
    rows = [row for path in sorted(output_dir.glob("*_seed*.json"))
            for row in rows_from(json.loads(path.read_text()))]
    return pd.DataFrame(rows).sort_values(KEY).reset_index(drop=True)


def report(frame: pd.DataFrame) -> None:
    """Per dataset, one block per method."""
    for dataset in frame.dataset.unique():
        print(f"\n{'=' * 66}\n  {dataset.upper()}  (mean ± sd over seeds)\n{'=' * 66}")
        print(f"  {'method':12s}{'policy':10s}{'AUROC':>16s}{'AP':>16s}{'F1':>16s}")
        for method in frame[frame.dataset == dataset].method.unique():
            rows = frame[(frame.dataset == dataset) & (frame.method == method)]
            label = detector_class(method).display_name
            for policy in sorted(rows.policy.unique()):
                sel = rows[rows.policy == policy]
                print(f"  {label:12s}{policy:10s}"
                      + "".join(f"{sel[m].mean():>9.4f} ±{sel[m].std():<6.4f}"
                                for m in ("auroc", "ap", "f1")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "both"), default="geolife")
    parser.add_argument("--methods", nargs="+", default=["ibad"], choices=list(REGISTRY))
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="default: replicate_seeds from the benchmark config")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--retrain", action="store_true",
                        help="refit even when a saved detector exists")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    seeds = args.seeds or cfg["replicate_seeds"]
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    datasets = DATASETS if args.dataset == "both" else (args.dataset,)

    print(f"methods={args.methods}  seeds={seeds}  device={device}")
    for dataset in datasets:
        for method in args.methods:
            for seed in seeds:
                run(method, dataset, cfg, seed, device, args.retrain)

    # The table is rebuilt from every saved run, so runs launched separately
    # (or in parallel) all land in it.
    frame = collect(ROOT / cfg["output_dir"])
    path = ROOT / cfg["output_dir"] / "replicates.csv"
    frame.to_csv(path, index=False)

    report(frame[frame.dataset.isin(datasets) & frame.method.isin(args.methods)])
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
