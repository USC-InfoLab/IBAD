"""Experiment — how the dissimilarity threshold ε shapes the splice benchmark.

A splice overwrites a target day with a donor day, and the guard ``min_shift``
(ε here) admits a donor only when the two days' template mixtures differ by at
least ε nats of Jensen-Shannon divergence. At ε = 0 any donor qualifies, so some
"anomalies" are near-copies of the day they replace; raising ε keeps only
genuine behavioural shifts and the benchmark gets easier. How fast a method's
score climbs with ε shows how much of its difficulty came from those
indistinguishable splices.

ε is a property of the benchmark, never of a model, so every ε is scored with
the same fitted detectors, and at each (seed, ε, policy) every method scores the
identical splices. Each run records, alongside the metrics, how many agents the
benchmark managed to splice and how far the spliced days actually moved.

Run from the repo root:
    python -m src.experiments.threshold
    python -m src.experiments.threshold --datasets geolife --methods ibad histmean
    python -m src.experiments.threshold --plot-only
"""

import argparse
import fcntl

import numpy as np
import pandas as pd
import torch

from src.detection.data import load_cohort
from src.detection.run import CONFIG_PATH, fitted_detector
from src.detectors import REGISTRY, detector_class
from src.evaluation import benchmark, scoring
from src.experiments import plots
from src.utils.config import ROOT, load_cfg

EXPERIMENT_CONFIG = "configs/experiments/threshold.yaml"
KEY = ["method", "dataset", "eps", "seed", "policy"]


def _shift(sequences, plan: np.ndarray) -> float:
    """Mean JS divergence (nats) between each spliced day and the day it replaced."""
    spliced = np.flatnonzero(plan[:, 0] >= 0)
    if not len(spliced):
        return float("nan")
    target = sequences.theta[spliced, plan[spliced, 0]]
    donor = sequences.theta[plan[spliced, 1], plan[spliced, 2]]
    return float(benchmark.jensen_shannon(target, donor).mean())


def run_dataset(dataset: str, cfg: dict, exp: dict, methods: list[str],
                seeds: list[int], device: torch.device) -> pd.DataFrame:
    """Every (method, seed, ε, policy) cell for one dataset."""
    cohort = load_cohort(dataset, cfg)
    splice = cfg["datasets"][dataset]["splice"]

    rows = []
    for seed in seeds:
        print(f"\n=== threshold · {dataset} · seed {seed} ===")
        detectors, windows = {}, {}
        for method in methods:
            detectors[method], _ = fitted_detector(method, dataset, cohort, cfg, seed, device)
            windows[method] = detectors[method].scored_window(cohort)
        sequences = windows[methods[0]].sequences

        for eps in exp["epsilons"]:
            for policy in exp["policies"]:
                # One draw per cell, shared by every method.
                bench = benchmark.spliced(sequences, policy, 2, seed, eps,
                                          splice["agent_frac"], splice["donors"])
                n_spliced = int((bench.plan[:, 0] >= 0).sum())
                shift = _shift(sequences, bench.plan)
                for method, detector in detectors.items():
                    metrics = scoring.ranking_metrics(
                        scoring.evaluate(detector, windows[method], bench))
                    rows.append({
                        "method": method, "dataset": dataset, "eps": eps,
                        "seed": seed, "policy": policy,
                        "auroc": metrics["auroc"], "ap": metrics["ap"],
                        "n_spliced": n_spliced, "mean_shift": shift,
                    })
                print(f"  eps={eps:<5g} {policy:9s} spliced {n_spliced}/{sequences.n_agents}"
                      f"  mean shift {shift:.3f} nats")
    return pd.DataFrame(rows)


def results_path(exp: dict, dataset: str):
    return ROOT / exp["output_dir"] / f"threshold_{dataset}.csv"


def save(frame: pd.DataFrame, exp: dict, dataset: str) -> None:
    """Merge into the dataset's table, replacing re-run cells.

    The merge holds a lock, so sweeps of different methods run in parallel all
    land in the table.
    """
    path = results_path(exp, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            frame = (pd.concat([pd.read_csv(path), frame], ignore_index=True)
                     .drop_duplicates(subset=KEY, keep="last"))
        frame.sort_values(KEY).to_csv(path, index=False)
    print(f"Saved -> {path}")


def load_results(exp: dict) -> dict[str, pd.DataFrame]:
    """The saved sweep, one table per dataset that has been run."""
    tables = {d: pd.read_csv(results_path(exp, d)) for d in exp["datasets"]
              if results_path(exp, d).exists()}
    if not tables:
        raise FileNotFoundError(
            "no threshold results yet; run `python -m src.experiments.threshold` first")
    return tables


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def figure(tables: dict[str, pd.DataFrame], exp: dict, out_stem=None):
    """The paper figure, laid out as the experiment config's ``figure`` section says."""
    spec = exp["figure"]
    return plots.threshold_figure(tables, spec["methods"], spec["policy"], out_stem,
                                  row_height=spec["row_height"], share_y=spec["share_y"])


def table(frame: pd.DataFrame, metric: str = "auroc", policy: str = "combined") -> pd.DataFrame:
    """Methods × ε, each cell the mean ± sd over seeds."""
    frame = frame[frame.policy == policy]
    stats = frame.groupby(["method", "eps"])[metric].agg(["mean", "std"])
    cells = stats.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1).unstack("eps")
    methods = [m for m in REGISTRY if m in cells.index]
    cells = cells.loc[methods]
    cells.index = [detector_class(m).display_name for m in methods]
    cells.columns = [f"ε = {e:g}" for e in cells.columns]
    return cells


def splice_stats(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per dataset, policy and ε: share of agents spliced, and how far the splices moved.

    The shift is the mean Jensen-Shannon divergence (nats) between each spliced
    day's template mixture and that of the day it replaced, averaged over seeds.
    """
    rows = []
    for dataset, frame in tables.items():
        one = frame[frame.method == frame.method.iloc[0]].drop_duplicates(["eps", "seed", "policy"])
        stats = one.groupby(["policy", "eps"])[["n_spliced", "mean_shift"]].mean()
        for (policy, eps), row in stats.iterrows():
            rows.append({"dataset": dataset, "policy": policy, "eps": eps,
                         "agents_spliced": int(row.n_spliced),
                         "mean_shift_nats": round(row.mean_shift, 3)})
    return pd.DataFrame(rows).set_index(["dataset", "policy", "eps"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=EXPERIMENT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None, choices=list(REGISTRY))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figure from the saved tables")
    args = parser.parse_args()

    exp = load_cfg(args.config)
    cfg = load_cfg(CONFIG_PATH)
    datasets = args.datasets or exp["datasets"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not args.plot_only:
        for dataset in datasets:
            frame = run_dataset(dataset, cfg, exp, args.methods or exp["methods"],
                                args.seeds or cfg["replicate_seeds"], device)
            save(frame, exp, dataset)

    tables = load_results(exp)
    for dataset, frame in tables.items():
        print(f"\n{dataset} · {exp['figure']['policy']} · AUROC\n"
              + table(frame, "auroc", exp["figure"]["policy"]).to_string())
    out = ROOT / exp["output_dir"] / "threshold"
    figure(tables, exp, out)
    print(f"Figure -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
