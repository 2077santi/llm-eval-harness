"""Agreement metrics for ordinal ratings, implemented from the definitions."""

from collections.abc import Sequence

import numpy as np


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation (average ranks for ties)."""
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort()
        r = np.empty_like(v)
        r[order] = np.arange(len(v))
        for value in np.unique(v):  # average the ranks of tied values
            tied = v == value
            r[tied] = r[tied].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def weighted_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int] = (1, 2, 3, 4, 5)) -> float:
    """Quadratic-weighted Cohen's kappa: the standard agreement metric for 1-5 scales.

    Disagreeing by 2 points costs four times as much as disagreeing by 1.
    """
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[index[x], index[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    weights = (i - j) ** 2 / (k - 1) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return 1.0
    return float(1 - (weights * observed).sum() / denominator)


def within_one(a: Sequence[int], b: Sequence[int]) -> float:
    """Share of items where two scores differ by at most one point."""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)) <= 1))
