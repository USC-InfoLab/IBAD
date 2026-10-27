"""The document contract shared by every dataset adapter.

A document is one agent-day. Its columns are slot-major one-hot indicators over
(time slot, activity) pairs:

    column = slot * n_acts + activity_index

Every adapter emits this layout so the discovery core and the plots can reshape
any topic matrix to ``(K, n_slots, n_acts)`` without knowing the dataset.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Documents:
    name: str
    X: csr_matrix          # (n_docs, n_slots * n_acts) one-hot, slot-major
    labels: list[str]      # activity vocabulary; index == activity index
    n_slots: int
    index: pd.DataFrame    # one row per document, identifying the agent-day

    @property
    def n_acts(self) -> int:
        return len(self.labels)

    @property
    def n_docs(self) -> int:
        return self.X.shape[0]

    def __post_init__(self) -> None:
        expected = self.n_slots * self.n_acts
        if self.X.shape[1] != expected:
            raise ValueError(
                f"{self.name}: X has {self.X.shape[1]} columns, "
                f"expected n_slots*n_acts = {expected}"
            )
        if len(self.index) != self.n_docs:
            raise ValueError(
                f"{self.name}: index has {len(self.index)} rows for "
                f"{self.n_docs} documents"
            )


def timelines_to_matrix(timelines: np.ndarray, n_acts: int) -> csr_matrix:
    """Ordinal timelines ``(n_docs, n_slots)`` -> slot-major one-hot matrix.

    Each slot contributes exactly one non-zero, so the result is built straight
    as COO coordinates rather than densified first.
    """
    n_docs, n_slots = timelines.shape
    rows = np.repeat(np.arange(n_docs, dtype=np.int64), n_slots)
    cols = (
        np.arange(n_slots, dtype=np.int64) * n_acts + timelines.astype(np.int64)
    ).ravel()
    data = np.ones(n_docs * n_slots, dtype=np.float32)
    return csr_matrix(
        (data, (rows, cols)), shape=(n_docs, n_slots * n_acts), dtype=np.float32
    )
