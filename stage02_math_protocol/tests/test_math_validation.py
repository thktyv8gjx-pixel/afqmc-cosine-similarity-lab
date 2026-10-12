import math
import unittest

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from math_validation import (
    binary_metrics,
    normalize_text,
    paired_sparse_cosine,
    predict_at_threshold,
    safe_cosine,
    select_threshold_by_f1,
    threshold_candidates,
)


class VectorMathTests(unittest.TestCase):
    def test_identical_nonzero_vectors_have_cosine_one(self):
        score, zero = safe_cosine((1, 2, 3), (1, 2, 3))
        self.assertAlmostEqual(score, 1.0, places=15)
        self.assertFalse(zero)

    def test_orthogonal_nonzero_vectors_have_cosine_zero(self):
        score, zero = safe_cosine((1, 0), (0, 1))
        self.assertEqual(score, 0.0)
        self.assertFalse(zero)

    def test_same_direction_different_lengths_have_cosine_one(self):
        score, zero = safe_cosine((1, 2), (3, 6))
        self.assertAlmostEqual(score, 1.0, places=15)
        self.assertFalse(zero)

    def test_hand_example_partial_overlap_is_four_fifths(self):
        score, zero = safe_cosine((1, 2), (2, 1))
        self.assertAlmostEqual(score, 4 / 5, places=15)
        self.assertFalse(zero)

    def test_zero_vector_convention_is_zero_and_flagged(self):
        self.assertEqual(safe_cosine((0, 0), (1, 2)), (0.0, True))
        self.assertEqual(safe_cosine((0, 0), (0, 0)), (0.0, True))

    def test_cosine_is_symmetric_and_in_range(self):
        x, y = (1, 2, 4), (2, 0, 1)
        xy, _ = safe_cosine(x, y)
        yx, _ = safe_cosine(y, x)
        self.assertAlmostEqual(xy, yx, places=15)
        self.assertGreaterEqual(xy, 0.0)
        self.assertLessEqual(xy, 1.0)


class CharacterAndTfidfTests(unittest.TestCase):
    def test_hand_counted_chinese_character_bigrams(self):
        texts = ["人工智能", "人工数学"]
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2), lowercase=False)
        matrix = vectorizer.fit_transform(texts)
        # CountVectorizer assigns indices in sorted feature order.
        expected_vocabulary = {"人工": 0, "工数": 1, "工智": 2, "数学": 3, "智能": 4}
        self.assertEqual(vectorizer.vocabulary_, expected_vocabulary)
        np.testing.assert_array_equal(matrix.toarray()[0], [1, 0, 1, 0, 1])
        np.testing.assert_array_equal(matrix.toarray()[1], [1, 1, 0, 1, 0])
        score, _ = safe_cosine(matrix.toarray()[0], matrix.toarray()[1])
        self.assertAlmostEqual(score, 1 / 3, places=15)

    def test_tfidf_smoothed_idf_and_l2_normalization(self):
        vocab = {"甲乙": 0, "乙甲": 1, "乙丙": 2}
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 2),
            lowercase=False,
            vocabulary=vocab,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(["甲乙甲", "甲乙丙"])
        rare_idf = 1.0 + math.log(3 / 2)
        self.assertAlmostEqual(vectorizer.idf_[0], 1.0, places=14)
        self.assertAlmostEqual(vectorizer.idf_[1], rare_idf, places=14)
        self.assertAlmostEqual(vectorizer.idf_[2], rare_idf, places=14)
        expected = np.array([1.0, rare_idf, 0.0]) / math.sqrt(1.0 + rare_idf**2)
        np.testing.assert_allclose(matrix.toarray()[0], expected, rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(float(np.linalg.norm(matrix.toarray()[0])), 1.0, places=14)

    def test_count_and_tfidf_share_exact_feature_indices(self):
        corpus = ["人工智能", "人工客服", "退款查询"]
        common = dict(analyzer="char", ngram_range=(2, 2), lowercase=False, min_df=1, max_df=1.0)
        count = CountVectorizer(**common).fit(corpus)
        tfidf = TfidfVectorizer(**common).fit(corpus)
        self.assertEqual(count.vocabulary_, tfidf.vocabulary_)
        self.assertEqual(count.get_feature_names_out().tolist(), tfidf.get_feature_names_out().tolist())

    def test_unseen_bigrams_and_short_text_transform_to_zero(self):
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2), lowercase=False).fit(["人工智能"])
        transformed = vectorizer.transform(["火星客服", "甲", ""])
        self.assertEqual(transformed.nnz, 0)
        self.assertEqual(transformed.shape, (3, 3))

    def test_three_asterisks_produce_repeated_double_asterisk_feature(self):
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2), lowercase=False)
        matrix = vectorizer.fit_transform(["***"])
        self.assertEqual(vectorizer.vocabulary_, {"**": 0})
        self.assertEqual(int(matrix[0, 0]), 2)

    def test_negation_digits_and_punctuation_are_preserved_as_features(self):
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2), lowercase=False)
        vectorizer.fit(["不要12元，退款吗？"])
        self.assertIn("不要", vectorizer.vocabulary_)
        self.assertIn("12", vectorizer.vocabulary_)
        self.assertIn("，退", vectorizer.vocabulary_)
        normalized = normalize_text("  不要 12元，退款吗？  ")
        self.assertEqual(normalized, "不要 12元，退款吗？")

    def test_normalization_is_nfc_and_only_collapses_whitespace(self):
        self.assertEqual(normalize_text(" A\u030a\t不 退款! "), "Å 不 退款!")


class ThresholdMetricAndSparseTests(unittest.TestCase):
    def test_threshold_uses_greater_than_or_equal(self):
        np.testing.assert_array_equal(predict_at_threshold([0.499, 0.5, 0.501], 0.5), [0, 1, 1])

    def test_candidates_include_unique_scores_and_all_negative_boundary(self):
        scores = [0.2, 0.5, 0.5, 0.8]
        expected = [0.2, 0.5, 0.8, float(np.nextafter(0.8, math.inf))]
        self.assertEqual(threshold_candidates(scores), expected)
        np.testing.assert_array_equal(predict_at_threshold(scores, expected[-1]), [0, 0, 0, 0])

    def test_threshold_tie_prefers_recall_then_precision_then_higher_threshold(self):
        y = [1, 1, 0, 0]
        scores = [0.9, 0.4, 0.8, 0.7]
        threshold, metrics = select_threshold_by_f1(y, scores)
        self.assertEqual(threshold, 0.4)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)

    def test_final_threshold_tie_break_prefers_higher_all_negative_boundary(self):
        threshold, metrics = select_threshold_by_f1([0, 0], [0.2, 0.8])
        self.assertEqual(threshold, float(np.nextafter(0.8, math.inf)))
        self.assertEqual((metrics["f1"], metrics["recall"], metrics["precision"]), (0.0, 0.0, 0.0))

    def test_confusion_counts_recompute_binary_metrics(self):
        metrics = binary_metrics([1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0])
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)

    def test_zero_denominator_metric_policy(self):
        no_positive_prediction = binary_metrics([1, 0], [0, 0])
        self.assertEqual(no_positive_prediction["precision"], 0.0)
        self.assertEqual(no_positive_prediction["recall"], 0.0)
        self.assertEqual(no_positive_prediction["f1"], 0.0)
        self.assertEqual(no_positive_prediction["accuracy"], 0.5)

    def test_nonbinary_labels_are_rejected_without_coercion(self):
        with self.assertRaises(ValueError):
            binary_metrics([0.5, 1], [0, 1])

    def test_sparse_paired_scores_match_sklearn_diagonal_and_flag_zeros(self):
        left = sparse.csr_matrix([[1, 0], [1, 1], [0, 0]], dtype=float)
        right = sparse.csr_matrix([[0, 1], [2, 2], [1, 0]], dtype=float)
        scores, zero_mask = paired_sparse_cosine(left, right)
        expected = cosine_similarity(left, right).diagonal()
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(scores, [0.0, 1.0, 0.0], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(zero_mask, [False, False, True])


if __name__ == "__main__":
    unittest.main()
