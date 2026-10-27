"""LDA fitting and K selection.

Each LDA topic is a behavioural template: a distribution over (slot, activity)
pairs. Each document's topic mixture is its soft template membership.
"""

import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation


def fit(
    X: csr_matrix,
    k: int,
    seed: int,
    max_iter: int,
    batch_size: int,
) -> LatentDirichletAllocation:
    """Fit an LDA model with ``k`` topics on a document-term matrix."""
    model = LatentDirichletAllocation(
        n_components=k,
        learning_method="online",
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=seed,
        n_jobs=-1,
    )
    started = time.time()
    model.fit(X)
    print(
        f"  K={k:>2}  {time.time() - started:6.1f}s  "
        f"perplexity={model.perplexity(X):.1f}"
    )
    return model


def transform(model: LatentDirichletAllocation, X: csr_matrix) -> np.ndarray:
    """Per-document topic mixtures, shape ``(n_docs, K)``, rows on the simplex."""
    theta = model.transform(X).astype(np.float32)
    return theta / theta.sum(axis=1, keepdims=True).clip(min=1e-12)


def topic_word_matrix(model: LatentDirichletAllocation) -> np.ndarray:
    """Row-normalised topic-word distributions, shape ``(K, n_slots * n_acts)``."""
    return model.components_ / model.components_.sum(axis=1, keepdims=True)


def sweep_k(
    X: csr_matrix,
    k_values: list[int],
    seed: int,
    max_iter: int,
    batch_size: int,
) -> pd.DataFrame:
    """Fit one model per K and record perplexity, for the elbow plot."""
    print(f"Perplexity sweep over K={k_values[0]}..{k_values[-1]}")
    rows = [
        {
            "k": k,
            "perplexity": fit(X, k, seed, max_iter, batch_size).perplexity(X),
        }
        for k in k_values
    ]
    return pd.DataFrame(rows)
