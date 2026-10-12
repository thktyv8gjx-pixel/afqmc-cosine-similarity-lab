from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import scipy
import sklearn

from data_pipeline import (
    align_internal_train_rows,
    build_unique_fit_corpus,
    label_counts,
    normalize_text,
    read_only_train_member,
    read_train_manifest,
    sha256_file,
    write_fit_corpus,
)
from features import (
    SHARED_VECTORIZER_PARAMS,
    fit_vectorizers,
    normalized_pair_texts,
    paired_sparse_cosine,
    write_feature_vocabulary,
)
from threshold_metrics import (
    efficient_threshold_scan,
    majority_label,
    metrics_from_predictions,
    paired_correctness,
    predict_at_threshold,
)


PREDICTION_FIELDS = [
    "source_row_id", "official_split", "internal_split", "method", "label", "score", "threshold",
    "prediction", "zero_vector_left", "zero_vector_right",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_output_checksums(experiment_dir: Path) -> Path:
    output_dir = experiment_dir / "outputs"
    checksum_lines = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        relative = path.relative_to(experiment_dir).as_posix()
        checksum_lines.append(f"{sha256_file(path)}  {relative}")
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return checksum_path


def canonical_source_id_fingerprint(row_ids: list[str]) -> str:
    digest = hashlib.sha256(b"AFQMC-fit-source-row-ids-v1\0")
    for row_id in sorted(row_ids):
        encoded = row_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def bigram_overlap(left: str, right: str) -> str:
    left_counts = Counter(left[i:i + 2] for i in range(max(0, len(left) - 1)))
    right_counts = Counter(right[i:i + 2] for i in range(max(0, len(right) - 1)))
    shared = [(gram, min(left_counts[gram], right_counts[gram])) for gram in left_counts.keys() & right_counts.keys()]
    shared.sort(key=lambda item: (-item[1], item[0]))
    return ";".join(f"{gram}×{count}" for gram, count in shared)


def error_hypothesis(row: dict[str, Any]) -> str:
    if row["zero_vector_left"] or row["zero_vector_right"]:
        return "至少一侧是零向量，按协议记 0；需核查词表外片段或过短文本。"
    negation_cues = ("不", "没", "未", "别", "无", "不能", "无法")
    left_has_negation = any(token in row["sentence1_normalized"] for token in negation_cues)
    right_has_negation = any(token in row["sentence2_normalized"] for token in negation_cues)
    if left_has_negation != right_has_negation:
        return "两侧否定线索不对称；字符重合可能未正确反映极性差异，需人工核对。"
    if row["error_type"] == "FP":
        return "有字符片段重合但标签为不相似；可能是共享常见片段而意图/实体不同，需人工核对。"
    return "标签为相似但匹配分数未过阈值；可能涉及同义改写、词序或字符重合稀疏，需人工核对。"


def _assert_expected_counts(rows: list[dict[str, Any]], expected: dict[str, int], name: str) -> None:
    actual = label_counts(rows)
    if actual != expected:
        raise ValueError(f"{name} label counts mismatch: {actual} != {expected}")


def _verify_and_recompute_predictions(
    path: Path,
    expected_summaries: dict[str, dict[str, Any]],
    expected_ids: list[str],
) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_FIELDS:
            raise AssertionError("prediction CSV columns are not as frozen")
        for row in reader:
            grouped.setdefault(row["method"], []).append(row)
    expected_methods = set(expected_summaries)
    if set(grouped) != expected_methods:
        raise AssertionError("prediction CSV method set differs from summary")
    for method, rows in grouped.items():
        ids = [row["source_row_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"duplicate prediction IDs for {method}")
        if ids != expected_ids:
            raise AssertionError(f"{method} prediction IDs or order differ from internal tune source IDs")
        y = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
        p = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int8)
        computed = metrics_from_predictions(y, p)
        target = expected_summaries[method]
        for key in ("tp", "fp", "fn", "tn"):
            if computed[key] != target[key]:
                raise AssertionError(f"{method} {key} differs between predictions and summary")
        for key in ("precision", "recall", "f1", "accuracy"):
            if abs(float(computed[key]) - float(target[key])) > 1e-12:
                raise AssertionError(f"{method} {key} differs between predictions and summary")


def run_experiment(project_root: Path, experiment_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    start_clock = __import__("time").perf_counter()
    config_path = experiment_dir / "config" / "experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = config["data_scope"]
    archive_path = project_root / paths["archive_path"]
    manifest_path = project_root / paths["manifest_path"]
    if not archive_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("frozen Stage 1 archive or manifest is missing")

    config_hash = sha256_file(config_path)
    if sha256_file(manifest_path) != paths["manifest_sha256"]:
        raise ValueError("Stage 1 split_manifest.csv SHA-256 mismatch")

    raw_train, train_hash = read_only_train_member(
        archive_path,
        paths["archive_sha256"],
        paths["train_member_sha256"],
        paths["expected_train_rows"],
    )
    train_manifest = read_train_manifest(manifest_path)
    fit_rows, tune_rows = align_internal_train_rows(raw_train, train_manifest, paths["expected_train_rows"])
    if len(fit_rows) != paths["expected_fit_rows"] or len(tune_rows) != paths["expected_tune_rows"]:
        raise ValueError("internal fit/tune row counts differ from the approved Stage 1 split")
    _assert_expected_counts(fit_rows, paths["expected_fit_labels"], "fit")
    _assert_expected_counts(tune_rows, paths["expected_tune_labels"], "tune")
    if {row["internal_split"] for row in fit_rows} != {"fit"} or {row["internal_split"] for row in tune_rows} != {"tune"}:
        raise ValueError("official Dev/Test or mixed internal split leaked into experiment rows")

    output_dir = experiment_dir / "outputs"
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    corpus_docs, corpus_fingerprint = build_unique_fit_corpus(fit_rows)
    corpus_file = artifacts_dir / "fit_corpus.jsonl"
    corpus_file_hash = write_fit_corpus(corpus_file, corpus_docs)
    fit_ids = [row["source_row_id"] for row in fit_rows]
    corpus_record = {
        "rule": config["idf_fit_corpus"],
        "authorization": "user approved this unique-normalized-question rule in the Stage 3A request on 2026-10-12",
        "source_archive_sha256": paths["archive_sha256"],
        "source_train_member_sha256": train_hash,
        "source_manifest_sha256": paths["manifest_sha256"],
        "source_split": "official_train/internal_fit only",
        "fit_pair_rows": len(fit_rows),
        "fit_pair_source_row_ids_sha256": canonical_source_id_fingerprint(fit_ids),
        "input_question_occurrences": 2 * len(fit_rows),
        "unique_normalized_documents": len(corpus_docs),
        "canonical_content_sha256": corpus_fingerprint,
        "serialized_jsonl_path": "outputs/artifacts/fit_corpus.jsonl",
        "serialized_jsonl_bytes": corpus_file.stat().st_size,
        "serialized_jsonl_sha256": corpus_file_hash,
        "all_source_rows_are_fit_only": True,
        "tune_used_for_fit": False,
        "official_dev_raw_text_read": False,
    }
    write_json(output_dir / "fit_corpus_record.json", corpus_record)

    corpus_texts = [doc["normalized_text"] for doc in corpus_docs]
    count_vectorizer, tfidf_vectorizer, features, vocabulary_hash = fit_vectorizers(corpus_texts)
    vocabulary_file_hash = write_feature_vocabulary(artifacts_dir / "feature_vocabulary.csv", features)
    write_json(output_dir / "vocabulary_record.json", {
        "feature_count": len(features),
        "count_tfidf_vocabulary_equal": count_vectorizer.vocabulary_ == tfidf_vectorizer.vocabulary_,
        "count_tfidf_column_order_equal": count_vectorizer.get_feature_names_out().tolist() == tfidf_vectorizer.get_feature_names_out().tolist(),
        "canonical_feature_index_sha256": vocabulary_hash,
        "csv_sha256": vocabulary_file_hash,
        "fit_documents": len(corpus_docs),
        "fit_source_split": "internal_fit_only",
        "tune_used_for_vectorizer_fit": False,
        "official_dev_raw_text_read": False,
    })

    tune_ids, tune_left, tune_right = normalized_pair_texts(tune_rows)
    if tune_ids != [row["source_row_id"] for row in tune_rows]:
        raise AssertionError("tune transform row ID order changed")
    y_tune = np.asarray([row["label"] for row in tune_rows], dtype=np.int8)
    prediction_records: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    all_scans: dict[str, list[dict[str, Any]]] = {}
    predictions_by_method: dict[str, np.ndarray] = {}
    score_by_method: dict[str, np.ndarray] = {}

    for method, vectorizer in (("count_char2", count_vectorizer), ("tfidf_char2", tfidf_vectorizer)):
        left_matrix = vectorizer.transform(tune_left).tocsr()
        right_matrix = vectorizer.transform(tune_right).tocsr()
        if left_matrix.shape[0] != len(tune_ids) or right_matrix.shape[0] != len(tune_ids):
            raise AssertionError("transform output row count differs from tune IDs")
        scores, zero_left, zero_right = paired_sparse_cosine(
            left_matrix, right_matrix, tolerance=config["cosine"]["floating_tolerance"]
        )
        if len(scores) != len(tune_rows):
            raise AssertionError("cosine scores lost tune rows")
        scan, best = efficient_threshold_scan(y_tune, scores)
        pred = predict_at_threshold(scores, best["threshold"])
        actual = metrics_from_predictions(y_tune, pred)
        for key in ("tp", "fp", "fn", "tn"):
            if actual[key] != best[key]:
                raise AssertionError(f"threshold scan and predictions disagree for {method} {key}")
        for key in ("precision", "recall", "f1", "accuracy"):
            if abs(float(actual[key]) - float(best[key])) > 1e-12:
                raise AssertionError(f"threshold scan and predictions disagree for {method} {key}")

        summaries[method] = {
            "method": method,
            "dataset_split": "internal_tune",
            "threshold": float(best["threshold"]),
            "threshold_selection_source": "internal_tune_only",
            "threshold_candidate_count": len(scan),
            "sample_count": len(tune_rows),
            **actual,
            "zero_vector_left_count": int(np.sum(zero_left)),
            "zero_vector_right_count": int(np.sum(zero_right)),
            "zero_vector_pair_count": int(np.sum(zero_left | zero_right)),
        }
        all_scans[method] = scan
        predictions_by_method[method] = pred
        score_by_method[method] = scores
        for row, score, prediction, zl, zr in zip(tune_rows, scores, pred, zero_left, zero_right):
            prediction_records.append({
                "source_row_id": row["source_row_id"],
                "official_split": "train",
                "internal_split": "tune",
                "method": method,
                "label": int(row["label"]),
                "score": float(score),
                "threshold": float(best["threshold"]),
                "prediction": int(prediction),
                "zero_vector_left": bool(zl),
                "zero_vector_right": bool(zr),
            })
        scan_rows = [{"candidate_index": i, **item, "is_selected": item["threshold"] == best["threshold"]}
                     for i, item in enumerate(scan)]
        scan_fields = ["candidate_index", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy", "is_selected"]
        write_csv(output_dir / f"threshold_scan_{method}.csv", scan_fields, scan_rows)

    majority, fit_label_counts = majority_label([row["label"] for row in fit_rows])
    baseline_pred = np.full(len(tune_rows), majority, dtype=np.int8)
    baseline_metrics = metrics_from_predictions(y_tune, baseline_pred)
    summaries["majority_fit_baseline"] = {
        "method": "majority_fit_baseline",
        "dataset_split": "internal_tune",
        "threshold": None,
        "threshold_selection_source": None,
        "threshold_candidate_count": 0,
        "sample_count": len(tune_rows),
        **baseline_metrics,
        "majority_label": majority,
        "fit_label_counts": fit_label_counts,
        "zero_vector_left_count": 0,
        "zero_vector_right_count": 0,
        "zero_vector_pair_count": 0,
    }
    for row, prediction in zip(tune_rows, baseline_pred):
        prediction_records.append({
            "source_row_id": row["source_row_id"],
            "official_split": "train",
            "internal_split": "tune",
            "method": "majority_fit_baseline",
            "label": int(row["label"]),
            "score": "",
            "threshold": "",
            "prediction": int(prediction),
            "zero_vector_left": False,
            "zero_vector_right": False,
        })

    prediction_path = output_dir / "internal_tune_predictions.csv"
    write_csv(prediction_path, PREDICTION_FIELDS, prediction_records)
    _verify_and_recompute_predictions(prediction_path, summaries, [row["source_row_id"] for row in tune_rows])

    summary_rows = [{key: summary.get(key, "") for key in (
        "method", "dataset_split", "threshold", "threshold_selection_source", "threshold_candidate_count", "sample_count",
        "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy",
        "zero_vector_left_count", "zero_vector_right_count", "zero_vector_pair_count", "majority_label",
    )} for summary in summaries.values()]
    write_csv(output_dir / "metrics_summary.csv", list(summary_rows[0]), summary_rows)

    count_pred = predictions_by_method["count_char2"]
    tfidf_pred = predictions_by_method["tfidf_char2"]
    paired = paired_correctness(y_tune, count_pred, tfidf_pred)
    if sum(paired.values()) != len(tune_rows):
        raise AssertionError("paired correctness cells do not sum to tune sample count")
    paired_payload = {
        "scope": "descriptive paired correctness comparison on internal tune; no independent-sample significance claim",
        "sample_count": len(tune_rows),
        "count_char2_vs_tfidf_char2": paired,
    }
    write_json(output_dir / "paired_correctness.json", paired_payload)

    tune_row_map = {row["source_row_id"]: row for row in tune_rows}
    tune_index = {row_id: index for index, row_id in enumerate(tune_ids)}
    fp_fn: list[dict[str, Any]] = []
    main_summary = summaries["tfidf_char2"]
    for source_id, row in tune_row_map.items():
        index = tune_index[source_id]
        prediction = int(tfidf_pred[index])
        label = int(row["label"])
        if prediction == label:
            continue
        left = normalize_text(row["sentence1"])
        right = normalize_text(row["sentence2"])
        item = {
            "source_row_id": source_id,
            "error_type": "FP" if label == 0 else "FN",
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "sentence1_normalized": left,
            "sentence2_normalized": right,
            "label": label,
            "score": float(score_by_method["tfidf_char2"][index]),
            "threshold": main_summary["threshold"],
            "prediction": prediction,
            "zero_vector_left": False,
            "zero_vector_right": False,
            "shared_char_2grams": bigram_overlap(left, right),
        }
        # Replace placeholders with the flags recorded in the row-level prediction table.
        item["possible_error_cause"] = error_hypothesis(item)
        fp_fn.append(item)
    fp_fn.sort(key=lambda r: (0 if r["error_type"] == "FP" else 1,
                              -r["score"] if r["error_type"] == "FP" else r["score"],
                              r["source_row_id"]))
    error_fields = ["source_row_id", "error_type", "sentence1", "sentence2", "sentence1_normalized", "sentence2_normalized",
                    "label", "score", "threshold", "prediction", "zero_vector_left", "zero_vector_right",
                    "shared_char_2grams", "possible_error_cause"]
    # Use the method's actual zero flags, preserving row alignment by ID.
    pred_by_id = {row["source_row_id"]: row for row in prediction_records if row["method"] == "tfidf_char2"}
    for item in fp_fn:
        item["zero_vector_left"] = pred_by_id[item["source_row_id"]]["zero_vector_left"]
        item["zero_vector_right"] = pred_by_id[item["source_row_id"]]["zero_vector_right"]
        item["possible_error_cause"] = error_hypothesis(item)
    write_csv(output_dir / "tfidf_error_cases.csv", error_fields, fp_fn)

    summary_json = {
        "scope": "All model performance values below are internal tune results, with thresholds selected on the same tune set; not final independent validation.",
        "positive_label": 1,
        "methods": summaries,
        "paired_correctness": paired_payload,
        "error_case_counts": {
            "tfidf_false_positive": sum(row["error_type"] == "FP" for row in fp_fn),
            "tfidf_false_negative": sum(row["error_type"] == "FN" for row in fp_fn),
            "tfidf_total_errors": len(fp_fn),
        },
        "actual_experiment_metrics": True,
        "official_dev_raw_text_read": False,
        "official_dev_prediction_or_selection": False,
    }
    write_json(output_dir / "metrics_summary.json", summary_json)

    write_json(output_dir / "config_fingerprint.json", {
        "config_path": "config/experiment.json",
        "config_sha256": config_hash,
        "config_bytes": config_path.stat().st_size,
    })
    finished = datetime.now(timezone.utc)
    run_metadata = {
        "run_started_utc": started.isoformat(),
        "run_finished_utc": finished.isoformat(),
        "elapsed_seconds": __import__("time").perf_counter() - start_clock,
        "project_root": str(project_root),
        "config_sha256": config_hash,
        "archive_sha256": sha256_file(archive_path),
        "train_member_sha256": train_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows_loaded_for_experiment": len(train_manifest),
        "official_train_rows_parsed": len(raw_train),
        "internal_fit_rows": len(fit_rows),
        "internal_tune_rows": len(tune_rows),
        "fit_label_counts": label_counts(fit_rows),
        "tune_label_counts": label_counts(tune_rows),
        "unique_idf_questions": len(corpus_docs),
        "fit_corpus_content_sha256": corpus_fingerprint,
        "fit_corpus_file_sha256": corpus_file_hash,
        "vocabulary_size": len(features),
        "vocabulary_sha256": vocabulary_hash,
        "count_tfidf_vocabulary_equal": count_vectorizer.vocabulary_ == tfidf_vectorizer.vocabulary_,
        "count_tfidf_column_order_equal": count_vectorizer.get_feature_names_out().tolist() == tfidf_vectorizer.get_feature_names_out().tolist(),
        "cosine_float_dtype": "float64",
        "official_dev_raw_text_read": False,
        "official_dev_model_comparison_threshold_selection_or_debugging": False,
        "zip_members_opened_for_text": ["train.json"],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "fit_rows_retained_without_filtering": len(fit_rows),
        "tune_rows_retained_without_filtering": len(tune_rows),
        "results_are_internal_tune_only": True,
    }
    write_json(output_dir / "run_metadata.json", run_metadata)

    # Build every plot exclusively from the just-written internal tune files.
    from visualize import generate_all_figures
    plot_meta = generate_all_figures(experiment_dir)
    write_json(output_dir / "plot_metadata.json", plot_meta)

    # Hash every artifact except this manifest itself, so outputs are independently checkable.
    write_output_checksums(experiment_dir)
    return {"run_metadata": run_metadata, "summary": summary_json, "plots": plot_meta}
