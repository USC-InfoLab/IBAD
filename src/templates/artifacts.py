"""Persist and reload discovered templates."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from src.features.documents import Documents


def path_for(artifacts_dir: str | Path, dataset: str) -> Path:
    return Path(artifacts_dir) / f"{dataset}_lda.pkl"


def save(
    artifacts_dir: str | Path,
    documents: Documents,
    model: LatentDirichletAllocation,
    theta: np.ndarray,
    k: int,
) -> Path:
    """Write the fitted model, per-day mixtures, and the vocabulary they assume.

    The vocabulary is stored alongside the model because a topic matrix is only
    interpretable against the column order it was fitted on.
    """
    path = path_for(artifacts_dir, documents.name)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": documents.name,
        "k": int(k),
        "lda": model,
        "theta": theta,
        "index": documents.index,
        "vocab": {
            "labels": documents.labels,
            "n_slots": documents.n_slots,
            "n_acts": documents.n_acts,
        },
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


def load(artifacts_dir: str | Path, dataset: str) -> dict:
    """Reverse of :func:`save`."""
    with open(path_for(artifacts_dir, dataset), "rb") as fh:
        return pickle.load(fh)
