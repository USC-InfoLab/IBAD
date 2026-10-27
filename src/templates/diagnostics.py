"""Membership diagnostics: does soft clustering buy anything over hard assignment?

If most days sit on a single topic, the mixture is degenerate and a hard
assignment would carry the same information.
"""

import numpy as np
import pandas as pd

PURE_THRESHOLD = 0.9


def membership_stats(theta: np.ndarray) -> dict[str, float]:
    """Summarise how mixed the per-document topic memberships are.

    Returns the fraction of documents dominated by one topic, the mean top
    weight, the mean membership entropy, and that entropy as a fraction of
    log(K) (1.0 = maximally mixed).
    """
    k = theta.shape[1]
    max_prob = theta.max(axis=1)
    entropy = -np.sum(theta * np.log(theta + 1e-12), axis=1)

    return {
        "frac_pure": float((max_prob > PURE_THRESHOLD).mean()),
        "mean_max_prob": float(max_prob.mean()),
        "mean_entropy": float(entropy.mean()),
        "frac_entropy_log_k": float(entropy.mean() / np.log(k)) if k > 1 else 0.0,
    }


def topic_usage(theta: np.ndarray) -> pd.DataFrame:
    """Per-topic corpus share, both soft (mean weight) and hard (argmax count)."""
    dominant = theta.argmax(axis=1)
    return pd.DataFrame({
        "topic": np.arange(theta.shape[1]),
        "soft_share": theta.mean(axis=0),
        "hard_share": np.bincount(dominant, minlength=theta.shape[1]) / len(theta),
    })
