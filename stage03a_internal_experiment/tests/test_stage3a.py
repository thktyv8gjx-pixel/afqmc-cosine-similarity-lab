from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

TEST_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = TEST_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from data_pipeline import align_internal_train_rows, build_unique_fit_corpus, normalize_text, read_only_train_member  # noqa: E402
from features import fit_vectorizers, paired_sparse_cosine  # noqa: E402
from threshold_metrics import efficient_threshold_scan, metrics_from_predictions, predict_at_threshold  # noqa: E402


def load_stage2_reference():
    ref_path = PROJECT_ROOT / "stage02_math_protocol" / "math_validation.py"
    if not ref_path.is_file():
        raise FileNotFoundError(f"Stage 2 reference implementation missing: {ref_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage2_math_reference", ref_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FrozenDataAndFitTests(unittest.TestCase):
    def test_normalizer_preserves_negation_digits_letters_punctuation_and_space(self):
        self.assertEqual(normalize_text("  不要 A12!\t退款  "), "不要 A12! 退款")
        self.assertEqual(normalize_text("A\u030a"), "Å")

    def test_manifest_reordering_does_not_misalign_rows(self):
        raw = [
            {"source_row_id": "train:train.json:000001", "sentence1": "人工智能", "sentence2": "人工数学", "label": 0},
            {"source_row_id": "train:train.json:000002", "sentence1": "不要退款", "sentence2": "不想退款", "label": 1},
        ]
        manifest = [
            {"source_file": "train.json", "source_row_id": raw[1]["source_row_id"], "official_split": "train", "internal_split": "tune", "label": "1", "pair_group_id": "g2"},
            {"source_file": "train.json", "source_row_id": raw[0]["source_row_id"], "official_split": "train", "internal_split": "fit", "label": "0", "pair_group_id": "g1"},
        ]
        fit, tune = align_internal_train_rows(raw, manifest, expected_rows=2)
        self.assertEqual([row["source_row_id"] for row in fit], [raw[0]["source_row_id"]])
        self.assertEqual([row["source_row_id"] for row in tune], [raw[1]["source_row_id"]])
        self.assertEqual(tune[0]["sentence1"], "不要退款")

    def test_manifest_group_crossing_fails_closed(self):
        raw = [
            {"source_row_id": f"train:train.json:{i:06d}", "sentence1": "甲乙", "sentence2": "乙丙", "label": 0}
            for i in (1, 2)
        ]
        manifest = [
            {"source_file": "train.json", "source_row_id": raw[0]["source_row_id"], "official_split": "train", "internal_split": "fit", "label": "0", "pair_group_id": "same"},
            {"source_file": "train.json", "source_row_id": raw[1]["source_row_id"], "official_split": "train", "internal_split": "tune", "label": "0", "pair_group_id": "same"},
        ]
        with self.assertRaisesRegex(ValueError, "crosses fit/tune"):
            align_internal_train_rows(raw, manifest, expected_rows=2)

    def test_unique_fit_question_corpus_is_sorted_deduplicated_and_provenanced(self):
        rows = [
            {"source_row_id": "r2", "sentence1": "  人工  智能", "sentence2": "人工智能"},
            {"source_row_id": "r1", "sentence1": "退款查询", "sentence2": "人工智能"},
        ]
        docs, fingerprint = build_unique_fit_corpus(rows)
        texts = [doc["normalized_text"] for doc in docs]
        self.assertEqual(texts, sorted(set(texts)))
        self.assertEqual(len(texts), 3)
        self.assertEqual(docs[texts.index("人工 智能")]["source_row_ids"], ["r2"])
        self.assertEqual(docs[texts.index("人工智能")]["source_row_ids"], ["r1", "r2"])
        self.assertEqual(len(fingerprint), 64)

    def test_vectorizers_fit_same_corpus_and_have_identical_vocabulary_and_order(self):
        corpus = ["人工智能", "退款查询", "不要扣款"]
        count, tfidf, features, fingerprint = fit_vectorizers(corpus)
        self.assertEqual(count.vocabulary_, tfidf.vocabulary_)
        self.assertEqual(count.get_feature_names_out().tolist(), tfidf.get_feature_names_out().tolist())
        self.assertEqual(len(features), len(count.vocabulary_))
        self.assertEqual(len(fingerprint), 64)

    def test_tune_only_sentinel_is_not_added_to_fit_vocabulary(self):
        _, tfidf, _, _ = fit_vectorizers(["人工智能", "退款查询"])
        fit_vocab_before = dict(tfidf.vocabulary_)
        transformed = tfidf.transform(["火星客服"])
        self.assertEqual(tfidf.vocabulary_, fit_vocab_before)
        self.assertEqual(transformed.nnz, 0)

    def test_zip_loader_opens_only_train_member(self):
        config = json.loads((EXPERIMENT_DIR / "config" / "experiment.json").read_text(encoding="utf-8"))
        data = config["data_scope"]
        archive_path = PROJECT_ROOT / data["archive_path"]
        original_open = zipfile.ZipFile.open
        accessed: list[str] = []

        def tracked_open(zf, name, *args, **kwargs):
            accessed.append(name.filename if isinstance(name, zipfile.ZipInfo) else str(name))
            return original_open(zf, name, *args, **kwargs)

        with patch.object(zipfile.ZipFile, "open", tracked_open):
            rows, train_hash = read_only_train_member(
                archive_path,
                data["archive_sha256"],
                data["train_member_sha256"],
                expected_rows=data["expected_train_rows"],
            )
        self.assertEqual(accessed, ["train.json"])
        self.assertEqual(rows[0]["source_row_id"], "train:train.json:000001")
        self.assertEqual(train_hash, data["train_member_sha256"])


class SparseFeatureAndCosineTests(unittest.TestCase):
    def test_short_unseen_punctuation_and_repeated_punctuation_behavior(self):
        count, _, _, _ = fit_vectorizers(["不要 A12!", "***", "人工智能"])
        vocab = count.vocabulary_
        self.assertIn("**", vocab)
        self.assertEqual(int(count.transform(["***"])[0, vocab["**"]]), 2)
        self.assertIn("要 ", vocab)
        self.assertIn(" A", vocab)
        self.assertIn("A1", vocab)
        self.assertIn("12", vocab)
        self.assertIn("2!", vocab)
        transformed = count.transform(["甲", "火星客服"])
        self.assertEqual(transformed.nnz, 0)

    def test_paired_sparse_cosine_matches_standard_and_is_symmetric(self):
        a = sparse.csr_matrix([[1, 2, 0], [0, 1, 1], [0, 0, 0]], dtype=np.float64)
        b = sparse.csr_matrix([[2, 1, 0], [1, 0, 1], [1, 0, 0]], dtype=np.float64)
        scores, zl, zr = paired_sparse_cosine(a, b)
        swapped, _, _ = paired_sparse_cosine(b, a)
        standard = cosine_similarity(a, b).diagonal()
        np.testing.assert_allclose(scores, standard, atol=1e-12, rtol=0)
        np.testing.assert_allclose(scores, swapped, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(zl | zr, [False, False, True])
        self.assertEqual(scores[2], 0.0)

    def test_sparse_cosine_rejects_nonfinite_negative_and_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            paired_sparse_cosine(sparse.csr_matrix([[np.inf]]), sparse.csr_matrix([[1.0]]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            paired_sparse_cosine(sparse.csr_matrix([[-1.0]]), sparse.csr_matrix([[1.0]]))
        with self.assertRaisesRegex(ValueError, "same shape"):
            paired_sparse_cosine(sparse.csr_matrix([[1.0, 0.0]]), sparse.csr_matrix([[1.0]]))

    def test_large_high_dimension_sparse_pairing_never_densifies(self):
        n, d = 20000, 100000
        row_ids = np.repeat(np.arange(n, dtype=np.int32), 2)
        cols = np.column_stack((np.arange(n) % d, (np.arange(n) + n) % d)).ravel().astype(np.int32)
        data = np.ones(n * 2, dtype=np.float64)
        matrix = sparse.csr_matrix((data, (row_ids, cols)), shape=(n, d))
        with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("dense conversion forbidden")):
            scores, zl, zr = paired_sparse_cosine(matrix, matrix)
        self.assertEqual(len(scores), n)
        self.assertEqual(int(matrix.nnz), n * 2)
        self.assertEqual(int(np.sum(zl | zr)), 0)
        np.testing.assert_allclose(scores, np.ones(n), rtol=0, atol=1e-12)
        self.assertGreater(n * n * 8, 3_000_000_000)  # a dense all-pairs matrix would exceed 3 GB


class ThresholdOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_stage2_reference()

    def assert_scan_matches_reference(self, y: np.ndarray, scores: np.ndarray) -> None:
        optimized, optimized_best = efficient_threshold_scan(y, scores)
        ref_thresholds = self.reference.threshold_candidates(scores.tolist())
        ref_rows = []
        for threshold in ref_thresholds:
            pred = self.reference.predict_at_threshold(scores.tolist(), threshold)
            ref_rows.append({"threshold": threshold, **self.reference.binary_metrics(y.tolist(), pred.tolist())})
        self.assertEqual([row["threshold"] for row in optimized], [row["threshold"] for row in sorted(ref_rows, key=lambda row: row["threshold"])])
        ref_by_threshold = {row["threshold"]: row for row in ref_rows}
        for row in optimized:
            expected = ref_by_threshold[row["threshold"]]
            for key in ("tp", "fp", "fn", "tn"):
                self.assertEqual(row[key], expected[key], (row, expected))
            for key in ("precision", "recall", "f1", "accuracy"):
                self.assertAlmostEqual(row[key], expected[key], delta=1e-15)
        ref_best = max(ref_rows, key=lambda row: (row["f1"], row["recall"], row["precision"], row["threshold"]))
        self.assertEqual(optimized_best["threshold"], ref_best["threshold"])
        pred_fast = predict_at_threshold(scores, optimized_best["threshold"])
        pred_ref = self.reference.predict_at_threshold(scores.tolist(), ref_best["threshold"])
        np.testing.assert_array_equal(pred_fast, pred_ref)
        fast_metrics = metrics_from_predictions(y, pred_fast)
        for key in fast_metrics:
            self.assertAlmostEqual(fast_metrics[key], ref_best[key], delta=1e-15)

    def test_optimized_scan_matches_all_stage2_candidates_for_fixtures(self):
        fixtures = [
            (np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.8, 0.7])),
            (np.array([0, 0]), np.array([0.2, 0.8])),
            (np.array([1, 0, 1, 0]), np.array([0.0, 0.0, 1.0, 1.0])),
        ]
        for y, scores in fixtures:
            with self.subTest(y=y.tolist(), scores=scores.tolist()):
                self.assert_scan_matches_reference(y, scores)

    def test_optimized_scan_matches_reference_on_many_discrete_ties(self):
        rng = np.random.default_rng(20261012)
        for case in range(60):
            n = int(rng.integers(2, 90))
            y = rng.integers(0, 2, size=n, dtype=np.int8)
            if len(np.unique(y)) == 1:
                y[0] = 1 - y[0]
            scores = rng.choice(np.array([0.0, 0.05, 0.2, 0.5, 0.75, 0.9, 1.0]), size=n, replace=True)
            with self.subTest(case=case):
                self.assert_scan_matches_reference(y, scores)

    def test_threshold_boundary_uses_greater_than_or_equal(self):
        pred = predict_at_threshold(np.array([0.49, 0.5, 0.51]), 0.5)
        np.testing.assert_array_equal(pred, [0, 1, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
