"""Frozen Stage 3B evaluator. Default mode fits the frozen Train fit space only.

Official Dev is opened only when --execute-official-dev is explicitly supplied
and the local publication gate confirms the exact public pre-Dev commit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import scipy
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "stage03a_internal_experiment"
STAGE3B_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from data_pipeline import (  # noqa: E402
    align_internal_train_rows,
    build_unique_fit_corpus,
    read_only_train_member,
    read_train_manifest,
    sha256_file,
)
from features import fit_vectorizers, normalized_pair_texts, paired_sparse_cosine  # noqa: E402
from pipeline import canonical_source_id_fingerprint  # noqa: E402
from threshold_metrics import metrics_from_predictions, predict_at_threshold  # noqa: E402


FREEZE_PATH = STAGE3B_DIR / "pre_dev_freeze_manifest.json"
GATE_PATH = STAGE3B_DIR / "local_only" / "part_a_publication_record.json"
OUTPUT_DIR = STAGE3B_DIR / "outputs"
PREDICTION_FIELDS = [
    "source_row_id", "line_number", "method", "label", "score", "threshold",
    "prediction", "zero_vector_left", "zero_vector_right",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def expected_environment() -> dict[str, str]:
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
    }


def parse_jsonl_records(raw: bytes, split: str) -> list[dict[str, Any]]:
    """Parse an already selected member; the caller controls which member is opened."""
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row in {split} member at line {line_no}")
        item = json.loads(line.decode("utf-8"))
        if not isinstance(item, dict):
            raise ValueError(f"non-object JSON value in {split} member at line {line_no}")
        if not isinstance(item.get("sentence1"), str) or not isinstance(item.get("sentence2"), str):
            raise ValueError(f"invalid question fields in {split} member at line {line_no}")
        label = str(item.get("label"))
        if label not in {"0", "1"}:
            raise ValueError(f"illegal label in {split} member at line {line_no}")
        records.append({
            "source_row_id": f"{split}:{split}.json:{line_no:06d}",
            "line_number": line_no,
            "sentence1": item["sentence1"],
            "sentence2": item["sentence2"],
            "label": int(label),
        })
    return records


def parse_manifest_split(manifest_path: Path, split_name: str) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_file", "source_row_id", "official_split", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("split manifest is missing required identity or label fields")
        for row in reader:
            if row["official_split"] != split_name:
                continue
            expected_file = f"{split_name}.json"
            if row["source_file"] != expected_file or row["label"] not in {"0", "1"}:
                raise ValueError(f"invalid {split_name} manifest row")
            row_id = row["source_row_id"]
            if row_id in seen:
                raise ValueError(f"duplicate {split_name} manifest ID")
            seen.add(row_id)
            selected.append({"source_row_id": row_id, "label": row["label"]})
    return selected


def verify_frozen_code(freeze: dict[str, Any]) -> None:
    payload = {key: value for key, value in freeze.items() if key != "freeze_manifest_sha256"}
    if sha256_bytes(stable_json_bytes(payload)) != freeze["freeze_manifest_sha256"]:
        raise AssertionError("canonical freeze manifest fingerprint mismatch")
    for relative_path, expected in freeze["frozen_source_files_sha256"].items():
        actual = sha256_file(ROOT / relative_path)
        if actual != expected:
            raise AssertionError(f"frozen source fingerprint mismatch: {relative_path}")
    env = expected_environment()
    if env != freeze["environment"]:
        raise AssertionError(f"runtime versions differ from the frozen environment: {env}")


def build_frozen_fit_space(freeze: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = EXPERIMENT_DIR / "config" / "experiment.json"
    config = read_json(config_path)
    if sha256_file(config_path) != freeze["stage3a_config_sha256"]:
        raise AssertionError("Stage 3A configuration hash mismatch")
    data = config["data_scope"]
    archive_path = ROOT / data["archive_path"]
    manifest_path = ROOT / data["manifest_path"]
    if sha256_file(archive_path) != freeze["inputs"]["archive_sha256"]:
        raise AssertionError("AFQMC archive SHA-256 mismatch")
    if sha256_file(manifest_path) != freeze["inputs"]["manifest_sha256"]:
        raise AssertionError("split manifest SHA-256 mismatch")

    manifest_rows = read_train_manifest(manifest_path)
    raw_train, train_member_hash = read_only_train_member(
        archive_path,
        freeze["inputs"]["archive_sha256"],
        freeze["inputs"]["train_member_sha256"],
        freeze["inputs"]["train_rows"],
    )
    if train_member_hash != freeze["inputs"]["train_member_sha256"]:
        raise AssertionError("Train member SHA-256 mismatch")
    fit_rows, tune_rows = align_internal_train_rows(raw_train, manifest_rows, freeze["inputs"]["train_rows"])
    if len(fit_rows) != freeze["fit_space"]["fit_pair_rows"] or len(tune_rows) != freeze["fit_space"]["tune_pair_rows"]:
        raise AssertionError("internal fit/tune row count differs from the freeze")
    if canonical_source_id_fingerprint([row["source_row_id"] for row in fit_rows]) != freeze["fit_space"]["fit_pair_source_row_ids_sha256"]:
        raise AssertionError("fit row identity fingerprint mismatch")

    fit_docs, corpus_hash = build_unique_fit_corpus(fit_rows)
    fit_texts = [doc["normalized_text"] for doc in fit_docs]
    serialized = "".join(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n" for doc in fit_docs).encode("utf-8")
    if corpus_hash != freeze["fit_space"]["idf_corpus_content_sha256"]:
        raise AssertionError("unique normalized fit corpus hash mismatch")
    if sha256_bytes(serialized) != freeze["fit_space"]["idf_corpus_serialized_sha256"]:
        raise AssertionError("serialized fit corpus hash mismatch")
    if len(fit_texts) != freeze["fit_space"]["unique_normalized_fit_question_count"]:
        raise AssertionError("unique fit question count mismatch")

    count_vectorizer, tfidf_vectorizer, features, vocab_hash = fit_vectorizers(fit_texts)
    if len(features) != freeze["fit_space"]["vocabulary_dimension"] or vocab_hash != freeze["fit_space"]["vocabulary_index_sha256"]:
        raise AssertionError("re-fitted character 2-gram vocabulary mismatch")
    if count_vectorizer.vocabulary_ != tfidf_vectorizer.vocabulary_:
        raise AssertionError("Count and TF-IDF feature indices differ")
    fit_matrix = tfidf_vectorizer.transform(fit_texts).tocsr()
    df = np.asarray(fit_matrix.getnnz(axis=0), dtype=np.int64).ravel()
    expected_idf = np.log((1.0 + len(fit_texts)) / (1.0 + df)) + 1.0
    if not np.allclose(tfidf_vectorizer.idf_, expected_idf, rtol=0.0, atol=1e-12):
        raise AssertionError("IDF values do not match the frozen fit-only formula")
    idf_hash = sha256_bytes(np.asarray(tfidf_vectorizer.idf_, dtype="<f8").tobytes(order="C"))
    if idf_hash != freeze["fit_space"]["idf_vector_sha256_little_endian_float64"]:
        raise AssertionError("IDF vector fingerprint mismatch")

    return {"count_char2": count_vectorizer, "tfidf_char2": tfidf_vectorizer}, fit_rows


def require_publication_gate(freeze: dict[str, Any]) -> str:
    gate = read_json(GATE_PATH)
    if gate.get("repository_full_name") != freeze["publication_target"]["repository_full_name"]:
        raise AssertionError("publication gate names a different repository")
    if gate.get("visibility") != "public" or gate.get("verified") is not True:
        raise AssertionError("public repository state is not verified in the local gate record")
    if gate.get("freeze_manifest_sha256") != freeze["freeze_manifest_sha256"]:
        raise AssertionError("publication gate refers to a different freeze manifest")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if branch != "main" or head != gate.get("freeze_commit_sha") or head != gate.get("remote_main_sha"):
        raise AssertionError("local main, frozen commit, and verified remote main do not match")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    if dirty:
        raise AssertionError("tracked files changed after the pre-Dev freeze commit")
    return head


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray | None) -> dict[str, Any]:
    base = metrics_from_predictions(labels, predictions)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    if (tp, fp, fn, tn) != (base["tp"], base["fp"], base["fn"], base["tn"]):
        raise AssertionError("independent confusion matrix differs from project metric implementation")
    precision = float(precision_score(labels, predictions, zero_division=0))
    recall = float(recall_score(labels, predictions, zero_division=0))
    f1 = float(f1_score(labels, predictions, zero_division=0))
    accuracy = float(accuracy_score(labels, predictions))
    for key, independent in (("precision", precision), ("recall", recall), ("f1", f1), ("accuracy", accuracy)):
        if not math.isclose(float(base[key]), independent, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"independent sklearn {key} differs from project metric implementation")
    specificity = tn / (tn + fp) if tn + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    result: dict[str, Any] = {
        **base,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "predicted_positive_proportion": float(np.mean(predictions == 1)),
        "roc_auc": float(roc_auc_score(labels, scores)) if scores is not None else None,
        "average_precision": float(average_precision_score(labels, scores)) if scores is not None else None,
    }
    return result


def read_dev_records(archive_path: Path, freeze: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if sha256_file(archive_path) != freeze["inputs"]["archive_sha256"]:
        raise AssertionError("archive changed immediately before the Dev read")
    with zipfile.ZipFile(archive_path, "r") as archive:
        with archive.open("dev.json", "r") as member:
            raw = member.read()
    actual_hash = sha256_bytes(raw)
    if actual_hash != freeze["inputs"]["dev_member_sha256"]:
        raise AssertionError("official Dev member SHA-256 mismatch")
    records = parse_jsonl_records(raw, "dev")
    if len(records) != freeze["inputs"]["dev_rows"]:
        raise AssertionError("official Dev row count mismatch")
    return records, actual_hash


def align_dev_manifest(records: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_rows = parse_manifest_split(manifest_path, "dev")
    row_ids = [row["source_row_id"] for row in records]
    manifest_ids = [row["source_row_id"] for row in manifest_rows]
    if row_ids != manifest_ids or len(set(row_ids)) != len(row_ids):
        raise AssertionError("Dev row IDs are not a complete ordered one-to-one match with the manifest")
    for raw, meta in zip(records, manifest_rows, strict=True):
        if str(raw["label"]) != meta["label"]:
            raise AssertionError("Dev raw label does not match the frozen manifest")


def compute_run(freeze: dict[str, Any], vectorizers: dict[str, Any], archive_path: Path, manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records, dev_hash = read_dev_records(archive_path, freeze)
    align_dev_manifest(records, manifest_path)
    labels = np.asarray([row["label"] for row in records], dtype=np.int8)
    ids = [row["source_row_id"] for row in records]
    left_text = [row["sentence1"] for row in records]
    right_text = [row["sentence2"] for row in records]
    methods: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    thresholds = freeze["thresholds"]
    for method in ("count_char2", "tfidf_char2"):
        vectorizer = vectorizers[method]
        left = vectorizer.transform(left_text).tocsr()
        right = vectorizer.transform(right_text).tocsr()
        scores, zero_left, zero_right = paired_sparse_cosine(left, right)
        threshold = float(thresholds[method])
        predictions = predict_at_threshold(scores, threshold)
        result = classification_metrics(labels, predictions, scores)
        result.update({
            "method": method,
            "dataset_split": "official_dev",
            "sample_count": len(records),
            "positive_label": 1,
            "threshold": threshold,
            "threshold_source": "frozen Stage 3A internal tune",
            "decision_rule": "score >= threshold",
            "zero_vector_left_count": int(zero_left.sum()),
            "zero_vector_right_count": int(zero_right.sum()),
            "zero_vector_pair_count": int(np.sum(zero_left | zero_right)),
        })
        methods[method] = result
        for index, row in enumerate(records):
            prediction_rows.append({
                "source_row_id": ids[index],
                "line_number": row["line_number"],
                "method": method,
                "label": int(labels[index]),
                "score": float(scores[index]),
                "threshold": threshold,
                "prediction": int(predictions[index]),
                "zero_vector_left": bool(zero_left[index]),
                "zero_vector_right": bool(zero_right[index]),
            })

    baselines: dict[str, Any] = {}
    for method, value in (("always_predict_0", 0), ("always_predict_1", 1)):
        predictions = np.full(len(labels), value, dtype=np.int8)
        result = classification_metrics(labels, predictions, None)
        result.update({
            "method": method,
            "dataset_split": "official_dev",
            "sample_count": len(records),
            "positive_label": 1,
            "threshold": None,
            "threshold_source": None,
            "zero_vector_pair_count": None,
        })
        baselines[method] = result

    summary = {
        "scope": "Official Dev evaluation with the pre-registered feature space and exact frozen thresholds",
        "positive_label": 1,
        "metrics": methods,
        "constant_baselines": baselines,
        "input_fingerprints": {
            "archive_sha256": freeze["inputs"]["archive_sha256"],
            "dev_member_sha256": dev_hash,
            "manifest_sha256": freeze["inputs"]["manifest_sha256"],
            "fit_corpus_sha256": freeze["fit_space"]["idf_corpus_content_sha256"],
            "vocabulary_index_sha256": freeze["fit_space"]["vocabulary_index_sha256"],
            "idf_vector_sha256": freeze["fit_space"]["idf_vector_sha256_little_endian_float64"],
            "freeze_manifest_sha256": freeze["freeze_manifest_sha256"],
        },
        "alignment": {
            "row_count": len(records),
            "unique_ids": len(set(ids)),
            "raw_order_matches_manifest": True,
            "all_labels_match_manifest": True,
            "all_rows_scored_by_both_methods": True,
        },
    }
    return summary, prediction_rows


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> bytes:
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def metric_csv_bytes(summary: dict[str, Any]) -> bytes:
    from io import StringIO
    fields = ["method", "sample_count", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy",
              "false_positive_rate", "specificity", "balanced_accuracy", "roc_auc", "average_precision",
              "predicted_positive_proportion", "threshold", "threshold_source", "zero_vector_pair_count"]
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in (*summary["constant_baselines"].values(), *summary["metrics"].values()):
        writer.writerow({key: row.get(key) for key in fields})
    return stream.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight the frozen fit space; official Dev requires an explicit one-time flag.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-official-dev", action="store_true", help="open and evaluate Dev once, after the public-commit gate passes")
    mode.add_argument("--replay-for-reproducibility", action="store_true", help="repeat the identical frozen evaluation and compare with saved outputs")
    args = parser.parse_args()
    freeze = read_json(FREEZE_PATH)
    verify_frozen_code(freeze)
    vectorizers, _fit_rows = build_frozen_fit_space(freeze)
    print("PASS: frozen Train fit corpus, vocabulary, IDF, configuration, and runtime versions match.")

    if not args.execute_official_dev and not args.replay_for_reproducibility:
        print("PRE-FLIGHT ONLY: official Dev was not opened.")
        return 0

    commit_sha = require_publication_gate(freeze)
    config = read_json(EXPERIMENT_DIR / "config" / "experiment.json")
    archive_path = ROOT / config["data_scope"]["archive_path"]
    manifest_path = ROOT / config["data_scope"]["manifest_path"]
    if args.execute_official_dev:
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError("official Dev output directory is non-empty; refusing a second evaluation")
        summary, rows = compute_run(freeze, vectorizers, archive_path, manifest_path)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        predictions_bytes = write_predictions_csv(OUTPUT_DIR / "dev_predictions_local_only.csv", rows)
        metrics_bytes = stable_json_bytes(summary)
        (OUTPUT_DIR / "dev_metrics.json").write_bytes(metrics_bytes)
        (OUTPUT_DIR / "dev_metrics.csv").write_bytes(metric_csv_bytes(summary))
        write_json(OUTPUT_DIR / "execution_record.json", {
            "executed_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_commit_sha": commit_sha,
            "official_dev_member_opened": True,
            "feature_vectorizer_fit_calls_after_dev_open": 0,
            "dev_transform_only": True,
            "thresholds_changed": False,
            "prediction_rows_sha256": sha256_bytes(predictions_bytes),
            "metrics_json_sha256": sha256_bytes(metrics_bytes),
        })
        print(f"Wrote local-only frozen Dev predictions and aggregate metrics for {summary['alignment']['row_count']} rows.")
        return 0

    expected_predictions = (OUTPUT_DIR / "dev_predictions_local_only.csv").read_bytes()
    expected_metrics = (OUTPUT_DIR / "dev_metrics.json").read_bytes()
    summary, rows = compute_run(freeze, vectorizers, archive_path, manifest_path)
    from io import StringIO
    replay_stream = StringIO(newline="")
    writer = csv.DictWriter(replay_stream, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    replay_predictions = replay_stream.getvalue().encode("utf-8")
    replay_metrics = stable_json_bytes(summary)
    checks = {
        "same_prediction_csv_sha256": sha256_bytes(replay_predictions) == sha256_bytes(expected_predictions),
        "same_metrics_json_sha256": sha256_bytes(replay_metrics) == sha256_bytes(expected_metrics),
        "same_row_count": len(rows) == summary["alignment"]["row_count"] * 2 == freeze["inputs"]["dev_rows"] * 2,
    }
    write_json(OUTPUT_DIR / "reproducibility_check.json", {
        "replayed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Identical frozen rerun for reproducibility only; no parameter selection or edits",
        "checks": checks,
        "passed": all(checks.values()),
    })
    if not all(checks.values()):
        raise AssertionError(f"frozen reproducibility replay mismatch: {checks}")
    print("PASS: identical frozen Dev replay matches the saved predictions and metrics exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
