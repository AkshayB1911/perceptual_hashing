"""Evaluation metrics for perceptual hashes.

Terminology:
    genuine pair  - an image and a transformed copy of it. Should match (small distance).
    impostor pair - two different images. Should not match (large distance).

All distances here are *normalized* Hamming distances in [0, 1], so 64-bit pHash
and 256-bit PDQ can be compared on the same axis. A pair is declared a match when
distance <= threshold.
"""

from __future__ import annotations

import numpy as np


def match_rate(genuine: np.ndarray, threshold: float) -> float:
    """Fraction of genuine pairs still matched at ``threshold`` (true positive rate)."""
    genuine = np.asarray(genuine)
    return float(np.mean(genuine <= threshold)) if genuine.size else float("nan")


def false_match_rate(impostor: np.ndarray, threshold: float) -> float:
    """Fraction of impostor pairs wrongly matched at ``threshold`` (false positive rate)."""
    impostor = np.asarray(impostor)
    return float(np.mean(impostor <= threshold)) if impostor.size else float("nan")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a rate of ``successes`` out of ``n``.

    Used for match rates: each image contributes one independent match/no-match per
    (algorithm, attack, param), so the count is binomial. Unlike the normal
    approximation it stays inside [0, 1] and behaves well at rates of 0 or 1.
    The threshold is treated as fixed (its own uncertainty is not included).
    """
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return float(lo), float(hi)


def threshold_at_fpr(impostor: np.ndarray, target_fpr: float) -> float:
    """Largest threshold whose false match rate on ``impostor`` is <= ``target_fpr``.

    Choosing the threshold from the impostor distribution (rather than a fixed number
    from a paper) makes the comparison between algorithms fair: every algorithm is
    allowed the same false-positive budget.
    """
    impostor = np.sort(np.asarray(impostor, dtype=np.float64))
    if impostor.size == 0:
        raise ValueError("no impostor distances")
    values = np.unique(impostor)
    fmr = np.searchsorted(impostor, values, side="right") / impostor.size
    ok = values[fmr <= target_fpr]
    # If even the smallest impostor distance is too many, match nothing below it.
    return float(ok.max()) if ok.size else float(values[0] - 1e-9)


def _rate_le(sorted_values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_values, thresholds, side="right") / max(sorted_values.size, 1)


def roc_curve(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds) sweeping every distinct distance as the threshold."""
    genuine = np.sort(np.asarray(genuine, dtype=np.float64))
    impostor = np.sort(np.asarray(impostor, dtype=np.float64))
    thresholds = np.unique(np.concatenate([[-np.inf], genuine, impostor, [np.inf]]))
    return _rate_le(impostor, thresholds), _rate_le(genuine, thresholds), thresholds


def auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(genuine, impostor)
    return float(np.trapezoid(tpr, fpr))


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(EER, threshold) where false-reject rate == false-accept rate."""
    fpr, tpr, thresholds = roc_curve(genuine, impostor)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2), float(thresholds[i])
