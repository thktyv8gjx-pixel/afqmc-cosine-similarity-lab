from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np


def _validated_arrays(y_true: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays of equal length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary 0/1")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return y.astype(np.int8, copy=False), s


def counts_to_metrics(tp: int, fp: int, fn: int, tn: int, n: int) -> dict[str, int | float]:
    if min(tp, fp, fn, tn) < 0 or tp + fp + fn + tn != n or n <= 0:
        raise ValueError("confusion counts must be non-negative and sum to a positive sample count")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / n
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def metrics_from_predictions(y_true: Any, y_pred: Any) -> dict[str, int | float]:
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("truth and predictions must be non-empty one-dimensional arrays of equal length")
    if not np.isin(y, (0, 1)).all() or not np.isin(p, (0, 1)).all():
        raise ValueError("truth and predictions must be binary 0/1")
    y = y.astype(np.int8, copy=False)
    p = p.astype(np.int8, copy=False)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    return counts_to_metrics(tp, fp, fn, tn, len(y))


def predict_at_threshold(scores: Any, threshold: float) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or not np.isfinite(s).all() or not math.isfinite(float(threshold)):
        raise ValueError("scores and threshold must be finite")
    return (s >= float(threshold)).astype(np.int8)


def efficient_threshold_scan(y_true: Any, scores: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate each unique-score cut and all-negative boundary in O(m log m)."""
    y, s = _validated_arrays(y_true, scores)
    n = len(y)
    positives = int(np.sum(y == 1))
    negatives = n - positives
    max_score = float(np.max(s))
    all_negative_threshold = float(np.nextafter(max_score, math.inf))
    if not math.isfinite(all_negative_threshold):
        raise ValueError("cannot represent a finite threshold above the maximum score")

    candidates: list[dict[str, Any]] = []
    # This extra candidate is equivalent to the Stage 2 all-negative boundary.
    candidates.append({
        "threshold": all_negative_threshold,
        **counts_to_metrics(0, 0, positives, negatives, n),
    })

    order = np.argsort(-s, kind="mergesort")
    tp = 0
    fp = 0
    index = 0
    while index < n:
        threshold = float(s[order[index]])
        end = index + 1
        while end < n and s[order[end]] == threshold:
            end += 1
        labels_at_score = y[order[index:end]]
        tp += int(np.sum(labels_at_score == 1))
        fp += int(np.sum(labels_at_score == 0))
        fn = positives - tp
        tn = negatives - fp
        candidates.append({
            "threshold": threshold,
            **counts_to_metrics(tp, fp, fn, tn, n),
        })
        index = end

    candidates.sort(key=lambda row: row["threshold"])
    best = max(
        candidates,
        key=lambda row: (row["f1"], row["recall"], row["precision"], row["threshold"]),
    )
    return candidates, best


def majority_label(y_fit: Any) -> tuple[int, dict[str, int]]:
    labels = np.asarray(y_fit)
    if labels.ndim != 1 or labels.size == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("fit labels must be a non-empty binary sequence")
    counts = Counter(int(v) for v in labels)
    label = 1 if counts[1] > counts[0] else 0
    return label, {"0": counts[0], "1": counts[1]}


def paired_correctness(y_true: Any, first_pred: Any, second_pred: Any) -> dict[str, int]:
    y = np.asarray(y_true, dtype=np.int8)
    a = np.asarray(first_pred, dtype=np.int8)
    b = np.asarray(second_pred, dtype=np.int8)
    if y.ndim != 1 or y.size == 0 or a.shape != y.shape or b.shape != y.shape:
        raise ValueError("paired correctness arrays must be aligned, one-dimensional and non-empty")
    a_ok = a == y
    b_ok = b == y
    return {
        "both_correct": int(np.sum(a_ok & b_ok)),
        "count_only_correct": int(np.sum(a_ok & ~b_ok)),
        "tfidf_only_correct": int(np.sum(~a_ok & b_ok)),
        "both_wrong": int(np.sum(~a_ok & ~b_ok)),
    }
