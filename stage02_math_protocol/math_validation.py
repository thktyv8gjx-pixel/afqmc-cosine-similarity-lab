"""Small, data-independent helpers for Stage 2 mathematical validation.

This module is exercised only with hand-built vectors and tiny synthetic texts.
It is not an AFQMC experiment runner.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable

import numpy as np
from scipy import sparse


def normalize_text(text: str) -> str:
    """Stage 2 candidate normalizer: NFC, trim, collapse whitespace only."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip())


def safe_cosine(left: Iterable[float], right: Iterable[float]) -> tuple[float, bool]:
    """Return cosine and whether either vector was zero; zero-pair convention is 0."""
    x = np.asarray(tuple(left), dtype=np.float64)
    y = np.asarray(tuple(right), dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("vectors must be one-dimensional and have equal dimensions")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("vectors must contain only finite values")
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        return 0.0, True
    value = float(np.dot(x, y) / (nx * ny))
    return float(np.clip(value, -1.0, 1.0)), False


def paired_sparse_cosine(
    left: sparse.spmatrix, right: sparse.spmatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Compute row-aligned sparse cosine scores without an all-pairs matrix."""
    a = sparse.csr_matrix(left, dtype=np.float64)
    b = sparse.csr_matrix(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired sparse matrices must have identical shapes")
    if not np.all(np.isfinite(a.data)) or not np.all(np.isfinite(b.data)):
        raise ValueError("sparse matrices must contain only finite values")
    dot = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    norm_a = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
    norm_b = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
    zero_mask = (norm_a == 0.0) | (norm_b == 0.0)
    denom = norm_a * norm_b
    scores = np.zeros(a.shape[0], dtype=np.float64)
    valid = ~zero_mask
    scores[valid] = dot[valid] / denom[valid]
    scores = np.clip(scores, -1.0, 1.0)
    return scores, zero_mask


def threshold_candidates(scores: Iterable[float]) -> list[float]:
    """Distinct observed scores plus a finite next-float all-negative boundary."""
    values = np.asarray(tuple(scores), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite one-dimensional sequence")
    observed = sorted(set(float(v) for v in values))
    above_max = float(np.nextafter(max(observed), math.inf))
    if not math.isfinite(above_max):
        raise ValueError("cannot construct a finite all-negative threshold")
    return observed + [above_max]


def predict_at_threshold(scores: Iterable[float], threshold: float) -> np.ndarray:
    values = np.asarray(tuple(scores), dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or not math.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    return (values >= threshold).astype(np.int8)


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float | int]:
    """Metrics for positive class 1; all zero denominators are defined as 0."""
    truth_raw = np.asarray(tuple(y_true))
    pred_raw = np.asarray(tuple(y_pred))
    if not np.isin(truth_raw, (0, 1)).all() or not np.isin(pred_raw, (0, 1)).all():
        raise ValueError("labels must be binary 0/1")
    truth = truth_raw.astype(np.int8, copy=False)
    pred = pred_raw.astype(np.int8, copy=False)
    if truth.ndim != 1 or pred.ndim != 1 or truth.shape != pred.shape or truth.size == 0:
        raise ValueError("truth and prediction must be non-empty one-dimensional arrays of equal size")
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / truth.size
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def select_threshold_by_f1(y_true: Iterable[int], scores: Iterable[float]) -> tuple[float, dict[str, float | int]]:
    """Maximize F1, then recall, precision, and finally the higher threshold."""
    score_values = np.asarray(tuple(scores), dtype=np.float64)
    if score_values.ndim != 1 or not np.all(np.isfinite(score_values)):
        raise ValueError("scores must be a finite one-dimensional sequence")
    candidates = threshold_candidates(score_values)
    evaluated = []
    for threshold in candidates:
        metrics = binary_metrics(y_true, predict_at_threshold(score_values, threshold))
        evaluated.append((threshold, metrics))
    return max(
        evaluated,
        key=lambda item: (
            item[1]["f1"],
            item[1]["recall"],
            item[1]["precision"],
            item[0],
        ),
    )
