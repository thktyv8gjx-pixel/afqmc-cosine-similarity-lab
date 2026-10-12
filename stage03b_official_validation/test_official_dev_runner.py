"""Synthetic-only tests for Stage 3B helpers; these tests never open AFQMC Dev."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_official_dev import classification_metrics, parse_jsonl_records  # noqa: E402


class FrozenEvaluatorHelperTests(unittest.TestCase):
    def test_parser_assigns_stable_synthetic_ids_and_binary_labels(self):
        raw = b'{"sentence1":"synthetic left","sentence2":"synthetic right","label":"1"}\n'
        rows = parse_jsonl_records(raw, "dev")
        self.assertEqual(rows[0]["source_row_id"], "dev:dev.json:000001")
        self.assertEqual((rows[0]["line_number"], rows[0]["label"]), (1, 1))

    def test_parser_rejects_invalid_rows_without_silently_dropping_them(self):
        with self.assertRaisesRegex(ValueError, "illegal label"):
            parse_jsonl_records(b'{"sentence1":"a","sentence2":"b","label":"2"}\n', "dev")
        with self.assertRaisesRegex(ValueError, "blank JSONL"):
            parse_jsonl_records(b"\n", "dev")

    def test_metric_record_matches_independent_confusion_and_sklearn_metrics(self):
        labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        predictions = np.asarray([1, 0, 1, 0], dtype=np.int8)
        scores = np.asarray([0.9, 0.4, 0.8, 0.1], dtype=np.float64)
        result = classification_metrics(labels, predictions, scores)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 1))
        self.assertEqual(result["false_positive_rate"], 0.5)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["roc_auc"], 0.75)

    def test_constant_baselines_have_no_ranking_metric_when_no_scores_exist(self):
        labels = np.asarray([1, 0, 0], dtype=np.int8)
        predictions = np.zeros(3, dtype=np.int8)
        result = classification_metrics(labels, predictions, None)
        self.assertEqual(result["roc_auc"], None)
        self.assertEqual(result["average_precision"], None)
        self.assertEqual(result["specificity"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
