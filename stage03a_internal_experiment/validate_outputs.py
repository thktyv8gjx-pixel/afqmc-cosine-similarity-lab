from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from data_pipeline import read_train_manifest, sha256_file  # noqa: E402
from pipeline import write_json, write_output_checksums  # noqa: E402
from threshold_metrics import metrics_from_predictions  # noqa: E402


def canonical_corpus_hash(texts: list[str]) -> str:
    digest = hashlib.sha256(b"AFQMC-IDF-unique-normalized-v1\0")
    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def canonical_vocab_hash(features: list[str]) -> str:
    digest = hashlib.sha256(b"AFQMC-char2-feature-vocabulary-v1\0")
    for index, feature in enumerate(features):
        raw = feature.encode("utf-8")
        digest.update(index.to_bytes(8, "big"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def run_validation() -> dict:
    output = EXPERIMENT_DIR / "outputs"
    config = json.loads((EXPERIMENT_DIR / "config" / "experiment.json").read_text(encoding="utf-8"))
    scope = config["data_scope"]
    manifest = read_train_manifest(PROJECT_ROOT / scope["manifest_path"])
    fit_rows = [row for row in manifest if row["internal_split"] == "fit"]
    tune_rows = [row for row in manifest if row["internal_split"] == "tune"]
    fit_ids = {row["source_row_id"] for row in fit_rows}
    tune_ids = [row["source_row_id"] for row in tune_rows]
    tune_labels = {row["source_row_id"]: int(row["label"]) for row in tune_rows}

    corpus_record = json.loads((output / "fit_corpus_record.json").read_text(encoding="utf-8"))
    corpus_path = output / "artifacts" / "fit_corpus.jsonl"
    docs = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    texts = [doc["normalized_text"] for doc in docs]
    checks = {
        "archive_hash_matches_frozen_config": sha256_file(PROJECT_ROOT / scope["archive_path"]) == scope["archive_sha256"],
        "manifest_hash_matches_frozen_config": sha256_file(PROJECT_ROOT / scope["manifest_path"]) == scope["manifest_sha256"],
        "fit_tune_counts_match_frozen_config": len(fit_rows) == scope["expected_fit_rows"] and len(tune_rows) == scope["expected_tune_rows"],
        "fit_corpus_is_sorted_unique": texts == sorted(set(texts)),
        "fit_corpus_count_matches_record": len(docs) == corpus_record["unique_normalized_documents"],
        "fit_corpus_fingerprint_matches_record": canonical_corpus_hash(texts) == corpus_record["canonical_content_sha256"],
        "fit_corpus_serialization_hash_matches_record": sha256_file(corpus_path) == corpus_record["serialized_jsonl_sha256"],
        "fit_corpus_provenance_fit_only": all(set(doc["source_row_ids"]).issubset(fit_ids) for doc in docs),
        "fit_corpus_uses_no_tune_or_dev_text": corpus_record["source_split"] == "official_train/internal_fit only" and not corpus_record["tune_used_for_fit"] and not corpus_record["official_dev_raw_text_read"],
    }

    vocab_path = output / "artifacts" / "feature_vocabulary.csv"
    with vocab_path.open("r", encoding="utf-8-sig", newline="") as handle:
        vocab_rows = list(csv.DictReader(handle))
    features = [row["char_2gram"] for row in vocab_rows]
    vocab_record = json.loads((output / "vocabulary_record.json").read_text(encoding="utf-8"))
    checks.update({
        "vocabulary_indices_contiguous": [int(row["feature_index"]) for row in vocab_rows] == list(range(len(vocab_rows))),
        "vocabulary_size_matches_record": len(features) == vocab_record["feature_count"],
        "count_tfidf_vocabulary_and_column_order_equal": vocab_record["count_tfidf_vocabulary_equal"] and vocab_record["count_tfidf_column_order_equal"],
        "vocabulary_fingerprint_matches_record": canonical_vocab_hash(features) == vocab_record["canonical_feature_index_sha256"],
        "vocabulary_file_hash_matches_record": sha256_file(vocab_path) == vocab_record["csv_sha256"],
    })

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (output / "internal_tune_predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["method"]].append(row)
    summary = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    expected_methods = {"count_char2", "tfidf_char2", "majority_fit_baseline"}
    checks["all_expected_prediction_methods_present"] = set(groups) == expected_methods
    recomputed: dict[str, dict] = {}
    for method in expected_methods:
        rows = groups[method]
        ids = [row["source_row_id"] for row in rows]
        y = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
        pred = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int8)
        metrics = metrics_from_predictions(y, pred)
        recomputed[method] = metrics
        checks[f"{method}_ids_complete_unique_ordered"] = ids == tune_ids and len(ids) == len(set(ids))
        checks[f"{method}_labels_match_manifest"] = all(int(row["label"]) == tune_labels[row["source_row_id"]] for row in rows)
        checks[f"{method}_confusion_sum_equals_tune_count"] = sum(metrics[key] for key in ("tp", "fp", "fn", "tn")) == len(tune_rows)
        target = summary["methods"][method]
        checks[f"{method}_csv_metrics_match_json"] = all(
            metrics[key] == target[key] if key in {"tp", "fp", "fn", "tn"}
            else abs(float(metrics[key]) - float(target[key])) <= 1e-12
            for key in metrics
        )
        checks[f"{method}_zero_rows_retained"] = len(rows) == len(tune_rows)

    fit_label_counts = Counter(row["label"] for row in fit_rows)
    majority = 1 if fit_label_counts["1"] > fit_label_counts["0"] else 0
    checks["majority_baseline_uses_fit_majority_only"] = all(int(row["prediction"]) == majority for row in groups["majority_fit_baseline"])

    scan_counts = {}
    for method in ("count_char2", "tfidf_char2"):
        prediction_rows = groups[method]
        scores = [float(row["score"]) for row in prediction_rows]
        expected_thresholds = sorted(set(scores)) + [float(np.nextafter(max(scores), math.inf))]
        scan_path = output / f"threshold_scan_{method}.csv"
        with scan_path.open("r", encoding="utf-8-sig", newline="") as handle:
            scan = list(csv.DictReader(handle))
        actual_thresholds = [float(row["threshold"]) for row in scan]
        selected = [row for row in scan if row["is_selected"] == "True"]
        chosen = float(summary["methods"][method]["threshold"])
        scan_counts[method] = len(scan)
        checks[f"{method}_candidate_set_exact"] = actual_thresholds == expected_thresholds
        checks[f"{method}_one_selected_threshold"] = len(selected) == 1 and float(selected[0]["threshold"]) == chosen
        checks[f"{method}_predictions_use_selected_threshold"] = all(
            int(row["prediction"]) == (1 if float(row["score"]) >= chosen else 0) for row in prediction_rows
        )

    error_path = output / "tfidf_error_cases.csv"
    with error_path.open("r", encoding="utf-8-sig", newline="") as handle:
        error_rows = list(csv.DictReader(handle))
    tfidf_rows = groups["tfidf_char2"]
    expected_error_ids = {row["source_row_id"] for row in tfidf_rows if int(row["prediction"]) != int(row["label"])}
    checks["tfidf_error_cases_cover_all_and_only_errors"] = {row["source_row_id"] for row in error_rows} == expected_error_ids and len(error_rows) == len(expected_error_ids)
    checks["tfidf_error_case_split_is_internal_tune"] = set(tune_ids).issuperset(row["source_row_id"] for row in error_rows)

    run_meta = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    checks["run_metadata_confirms_no_dev_raw_access"] = not run_meta["official_dev_raw_text_read"] and not run_meta["official_dev_model_comparison_threshold_selection_or_debugging"] and run_meta["zip_members_opened_for_text"] == ["train.json"]
    stage2_tests = json.loads((PROJECT_ROOT / "stage02_math_protocol" / "math_test_results.json").read_text(encoding="utf-8"))
    stage3_tests = json.loads((output / "stage3a_test_results.json").read_text(encoding="utf-8"))
    checks["stage2_21_math_tests_pass"] = stage2_tests["tests_run"] == 21 and stage2_tests["was_successful"] and stage2_tests["failures"] == stage2_tests["errors"] == 0
    checks["stage3a_implementation_tests_pass"] = stage3_tests["was_successful"] and stage3_tests["failures"] == stage3_tests["errors"] == 0

    plot_meta = json.loads((output / "plot_metadata.json").read_text(encoding="utf-8"))
    figure_results = {}
    for name, item in plot_meta["figures"].items():
        path = EXPERIMENT_DIR / item["path"]
        valid_png = False
        if path.is_file():
            with Image.open(path) as image:
                image.verify()
                valid_png = image.format == "PNG" and image.size == (1600, 1000)
        figure_results[name] = valid_png and sha256_file(path) == item["sha256"]
    checks["all_four_png_figures_valid_and_hashed"] = len(figure_results) == 4 and all(figure_results.values())
    checks["plot_sources_are_saved_tune_csvs"] = plot_meta["source_prediction_csv_sha256"] == sha256_file(output / "internal_tune_predictions.csv")

    # Verify the prior artifact checksum manifest before adding this report.
    checksum_file = output / "SHA256SUMS"
    checksum_results = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = EXPERIMENT_DIR / relative
        checksum_results.append(path.is_file() and sha256_file(path) == digest)
    checks["all_prevalidation_output_checksums_match"] = bool(checksum_results) and all(checksum_results)

    payload = {
        "scope": "Stage 3A internal Train fit/tune artifact validation only; no official Dev raw text read",
        "checks": checks,
        "passed": sum(checks.values()),
        "failed": len(checks) - sum(checks.values()),
        "threshold_scan_candidate_counts": scan_counts,
        "recomputed_metrics_from_prediction_csv": recomputed,
        "figure_checks": figure_results,
        "prevalidation_checksum_file_count": len(checksum_results),
    }
    write_json(output / "post_run_validation.json", payload)
    write_output_checksums(EXPERIMENT_DIR)
    return payload


if __name__ == "__main__":
    result = run_validation()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["failed"] == 0 else 1)
