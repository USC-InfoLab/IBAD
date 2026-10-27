"""The interface every anomaly detector implements — IBAD and each baseline.

A detector is fitted once on a cohort, then turns any version of that cohort —
its real days, or the same days with anomalies in place — into one score per
agent-day. Everything downstream (per-agent standardisation, metrics, the
experiments) consumes only those scores, so it never needs to know which method
produced them, and every method is evaluated by the same code.

Fitting may read the train and validation days only. The test window is scored,
never fitted on.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from src.evaluation.benchmark import Benchmark
from src.evaluation.scoring import Window
from src.features.visits import VisitSequences


def for_dataset(cfg: dict, dataset: str) -> dict:
    """A method config with one dataset's overrides layered onto the defaults.

    Every top-level section is a dict of defaults; ``datasets.<name>.<section>``
    overrides keys within it.
    """
    overrides = cfg.get("datasets", {}).get(dataset, {})
    return {
        section: {**values, **overrides.get(section, {})}
        for section, values in cfg.items()
        if section != "datasets" and isinstance(values, dict)
    }


class Detector(ABC):
    """Scores agent-days; higher means more anomalous."""

    name: ClassVar[str]                       # id used in configs and file names
    display_name: ClassVar[str]               # how tables and figures name it
    config_path: ClassVar[str | None] = None  # hyper-parameters, relative to the repo root
    checkpoint_ext: ClassVar[str] = ".joblib"
    # Splits whose clean scores set each agent's standardisation baseline.
    calibration_splits: ClassVar[tuple[int, ...]] = (0,)
    # Whether a score of exactly 0 marks a day the method did not score (see
    # src/evaluation/scoring.py).
    zero_marks_unscored: ClassVar[bool] = True

    def __init__(self, cfg: dict, cohort, seed: int, device: torch.device):
        """``cfg`` is already resolved for the cohort's dataset (see :func:`for_dataset`)."""
        self.cfg = cfg
        self.cohort = cohort
        self.seed = seed
        self.device = device

    @abstractmethod
    def fit(self) -> dict:
        """Fit on the cohort's train (and val) days; returns a short summary."""

    @abstractmethod
    def score(self, sequences: VisitSequences) -> np.ndarray:
        """An ``(n_agents, n_days - 1)`` grid of scores for ``sequences``.

        ``sequences`` is the fitted cohort or a perturbed copy of it with the
        same agents and day layout. Column *j* scores day *j + 1*; NaN marks a
        day that was not scored (padding, or no history to judge it by).
        """

    def calibration_scores(self, sequences: VisitSequences, days: np.ndarray) -> np.ndarray:
        """Scores of the ``(n_agents, n_days)`` mask ``days``, NaN elsewhere.

        They set each agent's standardisation baseline. By default these are
        :meth:`score`'s, so an agent's first day, which it never scores, is
        left out.
        """
        return np.where(days[:, 1:], self.score(sequences), np.nan)

    def scored_window(self, cohort) -> Window:
        """The days the benchmark is drawn on, their clean scores, and each
        agent's scores on its normal days, which set its standardisation baseline.

        The normal days are the ``calibration_splits`` of the fitted cohort;
        when the test window stands alone (NUMOSIM) that is the whole fit
        window, scored as a sequence of its own.
        """
        fitted = cohort.sequences
        calibration = self.calibration_scores(
            fitted, np.isin(fitted.split, self.calibration_splits) & (fitted.template >= 0))
        sequences = fitted if cohort.test_window is None else cohort.test_window
        return Window(sequences, self.score(sequences), calibration,
                      standalone=cohort.test_window is not None)

    def perturbed_scores(self, window: Window, bench: Benchmark) -> np.ndarray:
        """Scores of ``window``'s days with ``bench``'s anomalies in place."""
        return self.score(bench.perturbed)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist whatever :meth:`fit` learned."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore a fitted detector saved by :meth:`save`."""
