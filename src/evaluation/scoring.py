"""From per-day anomaly scores to metrics — the same path for every detector.

A detector returns an ``(n_agents, n_days - 1)`` grid: column *j* scores day
*j + 1* (the first day has no history to forecast from). The cohort is scored
twice — clean and with the benchmark's anomalies in place — and the two grids
are compared position by position, after each agent's scores are standardised
against that agent's normal days:

* A cell is standardised only when its clean score is a real one — finite and,
  for every method but TrajGenAgent, non-zero, since detectors use 0 for a day
  they do not score (``zero_marks_unscored``). Otherwise both the clean cell
  and its perturbed counterpart are ranked as they are.
* When the test window is a sequence of its own (NUMOSIM), every day of it is
  evaluated; otherwise the days of the benchmark's split.
* Scores are ranked as float32.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from src.evaluation.benchmark import Benchmark
from src.features.visits import VisitSequences


@dataclass
class Window:
    """One fitted detector's scores on the days a benchmark is drawn on, reused
    across benchmarks (see :meth:`src.detectors.base.Detector.scored_window`)."""

    sequences: VisitSequences   # the days the benchmark is drawn on
    clean: np.ndarray           # (n_agents, T - 1) their scores
    calibration: np.ndarray     # each agent's scores on its normal days
    standalone: bool            # the test window is a sequence of its own
    labels_offset: int = 1      # score column j is judged against label column j + offset
    state: object = None        # whatever the detector's perturbed scores need


def agent_baseline(calibration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-agent mean and sd of a calibration grid, one row per agent.

    A pooled ranking over raw scores partly ranks *agents*: people differ in how
    many stays they log and how regular they are, so their score distributions
    sit at different offsets. Standardising each agent against its own normal
    days asks the question the benchmark poses — is this day unusual *for this
    person* — and an injected day never shifts its own baseline.

    Only finite, non-zero cells count. Agents with fewer than two calibration
    days get ``(0, 1)``: left unstandardised rather than scaled by a single day.
    """
    usable = np.isfinite(calibration) & (calibration != 0)
    n = usable.sum(axis=1, keepdims=True).astype(np.float64)
    values = np.where(usable, calibration, 0.0).astype(np.float64)
    mean = values.sum(axis=1, keepdims=True) / np.maximum(n, 1.0)
    var = (values ** 2).sum(axis=1, keepdims=True) / np.maximum(n, 1.0) - mean ** 2
    sd = np.sqrt(np.maximum(var, 0.0)) + 1e-6
    return np.where(n >= 2, mean, 0.0), np.where(n >= 2, sd, 1.0)


def flatten(
    clean_raw: np.ndarray,
    perturbed_raw: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    calibration: np.ndarray,
    zero_marks_unscored: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """The ``(y_true, y_score)`` arrays over ``positions``.

    Every position contributes a negative from the clean grid; the perturbed
    grid contributes a positive where the day was altered and another negative
    where it was not. ``labels`` is on the score grid's day axis. Missing scores
    drop out — TrajGenAgent, for one, scores only the days it generated chains for.
    """
    mean, sd = agent_baseline(calibration)
    clean = np.asarray(clean_raw, dtype=np.float64)
    perturbed = np.asarray(perturbed_raw, dtype=np.float64)
    standardised = np.isfinite(clean) & ((clean != 0) if zero_marks_unscored else True)
    clean = np.where(standardised, (clean - mean) / sd, clean)
    perturbed = np.where(standardised, (perturbed - mean) / sd, perturbed)
    y = np.concatenate([np.zeros(int(positions.sum()), dtype=np.int32),
                        labels[positions].astype(np.int32)])
    s = np.concatenate([clean[positions], perturbed[positions]]).astype(np.float32)
    scored = np.isfinite(s)
    return y[scored], s[scored]


def evaluate(detector, window: Window, bench: Benchmark) -> tuple[np.ndarray, np.ndarray]:
    """One benchmark's ``(y_true, y_score)``: its perturbed scores against the clean ones."""
    clean = window.clean
    positions = (np.ones(clean.shape, dtype=bool) if window.standalone
                 else window.sequences.split[:, 1:clean.shape[1] + 1] == bench.split)
    labels = bench.labels[:, window.labels_offset:window.labels_offset + clean.shape[1]]
    return flatten(clean, detector.perturbed_scores(window, bench), labels, positions,
                   window.calibration, detector.zero_marks_unscored)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1, and that F1. Tune on validation only."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    if thresholds.size == 0:
        return float("nan"), float("nan")
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def ranking_metrics(flat: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Threshold-free AUROC and AP."""
    y, s = flat
    defined = len(np.unique(y)) >= 2
    return {
        "auroc": float(roc_auc_score(y, s)) if defined else float("nan"),
        "ap": float(average_precision_score(y, s)) if defined else float("nan"),
    }


def metrics(
    val_flat: tuple[np.ndarray, np.ndarray] | None,
    test_flat: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    """AUROC and AP on test; F1 at the threshold chosen on validation (NaN without one)."""
    y_test, s_test = test_flat
    threshold, val_f1 = best_threshold(*val_flat) if val_flat is not None else (float("nan"),) * 2
    if np.isfinite(threshold) and len(np.unique(y_test)) >= 2:
        test_f1 = float(f1_score(y_test, (s_test >= threshold).astype(np.int32),
                                 zero_division=0))
    else:
        test_f1 = float("nan")
    return {**ranking_metrics(test_flat), "f1": test_f1,
            "threshold": threshold, "val_f1": val_f1}
