from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from data_pipeline import normalize_text


SHARED_VECTORIZER_PARAMS: dict[str, Any] = {
    "input": "content",
    "encoding": "utf-8",
    "decode_error": "strict",
    "strip_accents": None,
    "lowercase": False,
    "preprocessor": None,
    "tokenizer": None,
    "stop_words": None,
    "token_pattern": r"(?u)\b\w\w+\b",
    "ngram_range": (2, 2),
    "analyzer": "char",
    "max_df": 1.0,
    "min_df": 1,
    "max_features": None,
    "vocabulary": None,
    "binary": False,
    "dtype": np.float64,
}


def fit_vectorizers(fit_corpus: list[str]) -> tuple[CountVectorizer, TfidfVectorizer, list[str], str]:
    if not fit_corpus:
        raise ValueError("fit corpus must not be empty")
    count = CountVectorizer(**SHARED_VECTORIZER_PARAMS)
    tfidf = TfidfVectorizer(
        **SHARED_VECTORIZER_PARAMS,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    count.fit(fit_corpus)
    tfidf.fit(fit_corpus)
    if count.vocabulary_ != tfidf.vocabulary_:
        raise AssertionError("Count and TF-IDF vocabulary indices differ")
    count_features = count.get_feature_names_out().tolist()
    tfidf_features = tfidf.get_feature_names_out().tolist()
    if count_features != tfidf_features:
        raise AssertionError("Count and TF-IDF feature column order differs")
    if len(count_features) != len(count.vocabulary_):
        raise AssertionError("feature names and vocabulary size disagree")
    digest = hashlib.sha256(b"AFQMC-char2-feature-vocabulary-v1\0")
    for index, feature in enumerate(count_features):
        encoded = feature.encode("utf-8")
        digest.update(index.to_bytes(8, "big"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return count, tfidf, count_features, digest.hexdigest()


def write_feature_vocabulary(path: Path, features: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["feature_index", "char_2gram"])
        writer.writerows((index, feature) for index, feature in enumerate(features))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_pair_texts(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    row_ids = [row["source_row_id"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate row IDs in pair transform input")
    left = [normalize_text(row["sentence1"]) for row in rows]
    right = [normalize_text(row["sentence2"]) for row in rows]
    return row_ids, left, right


def paired_sparse_cosine(
    left: sparse.spmatrix, right: sparse.spmatrix, tolerance: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one cosine scores; no pairwise or all-question matrix is formed."""
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")
    a = sparse.csr_matrix(left, dtype=np.float64)
    b = sparse.csr_matrix(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired matrices must have the same shape, got {a.shape} and {b.shape}")
    if not np.isfinite(a.data).all() or not np.isfinite(b.data).all():
        raise ValueError("non-finite sparse feature value")
    if (a.data < 0).any() or (b.data < 0).any():
        raise ValueError("Count/TF-IDF features must be non-negative")

    dot = np.asarray(a.multiply(b).sum(axis=1), dtype=np.float64).ravel()
    left_sq = np.asarray(a.multiply(a).sum(axis=1), dtype=np.float64).ravel()
    right_sq = np.asarray(b.multiply(b).sum(axis=1), dtype=np.float64).ravel()
    if not np.isfinite(dot).all() or not np.isfinite(left_sq).all() or not np.isfinite(right_sq).all():
        raise ValueError("non-finite paired dot product or squared norm")
    left_norm = np.sqrt(left_sq)
    right_norm = np.sqrt(right_sq)
    zero_left = left_norm == 0.0
    zero_right = right_norm == 0.0
    zero_either = zero_left | zero_right
    denom = left_norm * right_norm
    scores = np.zeros(a.shape[0], dtype=np.float64)
    valid = ~zero_either
    scores[valid] = dot[valid] / denom[valid]
    if not np.isfinite(scores).all():
        raise ValueError("non-finite cosine score")
    if (scores < -tolerance).any() or (scores > 1.0 + tolerance).any():
        raise ValueError("cosine score outside the non-negative vector range [0,1]")
    # Values are clipped only after the explicit small-error range check.
    scores = np.clip(scores, 0.0, 1.0)
    return scores, zero_left, zero_right
