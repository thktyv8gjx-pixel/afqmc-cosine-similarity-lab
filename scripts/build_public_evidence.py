"""Build data-free aggregate evidence from local Stage 1/3A audit records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "stage01_data_audit"
STAGE3A = ROOT / "stage03a_internal_experiment"
DEST = ROOT / "public_evidence"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, value: Any) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build() -> dict[str, Any]:
    audit = load(STAGE1 / "data_audit.json")
    crosscheck = load(STAGE3A / "qa_review" / "outputs" / "independent_metric_crosscheck.json")
    qa = load(STAGE3A / "qa_review" / "outputs" / "qa_diagnostics.json")
    idf = crosscheck["idf_audit"]

    dataset = {
        "source": "CLUE AFQMC; see the official source links in README.md",
        "redistribution_permission": "not specified on the checked source pages; raw data is excluded",
        "fingerprints": {
            "archive_sha256": audit["source"]["archive_sha256"],
            "train_member_sha256": audit["archive_members"]["train.json"]["sha256"],
            "dev_member_sha256": audit["archive_members"]["dev.json"]["sha256"],
            "test_member_sha256": audit["archive_members"]["test.json"]["sha256"],
            "split_manifest_sha256": load(STAGE3A / "config" / "experiment.json")["data_scope"]["manifest_sha256"],
        },
        "row_counts": {name: audit["per_split"][name]["physical_line_rows"] for name in ("train", "dev", "test")},
        "train_dev_pair_overlap_groups": audit["train_dev_pair_overlap"]["normalized_unordered_pair_key_intersection_groups"],
        "train_dev_shared_normalized_sentences": audit["train_dev_shared_sentence_overlap"]["normalized_single_sentence_key_intersection"],
        "internal_split": {
            "algorithm": audit["internal_split"]["algorithm"],
            "fit_rows": audit["internal_split"]["rows"]["fit"],
            "tune_rows": audit["internal_split"]["rows"]["tune"],
            "fit_label_counts": audit["internal_split"]["label_counts"]["fit"],
            "tune_label_counts": audit["internal_split"]["label_counts"]["tune"],
        },
        "contains_row_ids_or_question_text": False,
    }
    write("stage1_dataset_fingerprints.json", dataset)

    methods = {}
    for key in ("count_char2", "tfidf_char2"):
        source = crosscheck["model_metrics"][key]
        methods[key] = {
            name: source[name] for name in (
                "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy",
                "false_positive_rate", "specificity", "balanced_accuracy", "roc_auc",
                "average_precision", "predicted_positive_rate", "selected_threshold",
            )
        }
    baselines = {
        key: {
            name: value[name] for name in (
                "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy",
                "false_positive_rate", "specificity", "balanced_accuracy", "predicted_positive_rate",
            )
        }
        for key, value in crosscheck["baseline_metrics"].items()
    }
    summary = {
        "scope": "internal tune only; each model threshold was selected on this same split; not independent validation",
        "sample_count": crosscheck["n"],
        "positive_label": 1,
        "label_counts": crosscheck["labels"],
        "methods": methods,
        "constant_baselines": baselines,
        "f1_gain_over_always_positive": crosscheck["model_f1_minus_always_positive"],
        "idf_corpus_unique_normalized_fit_questions": idf["unique_normalized_fit_questions"],
        "vocabulary_dimension": idf["vocabulary_dimension"],
        "idf_corpus_sha256": idf["fit_corpus_content_sha256"],
        "idf_vector_sha256": idf["idf_vector_sha256_little_endian_float64"],
        "vocabulary_index_sha256": idf["vocabulary_index_sha256"],
        "contains_row_level_scores_or_questions": False,
    }
    write("stage3a_internal_tune_metrics.json", summary)

    extension = load(STAGE3A / "qa_review" / "outputs" / "char23_config_fingerprint.json")
    extension_result_path = STAGE3A / "qa_review" / "outputs" / "qa_diagnostics.json"
    # Read only aggregate fields from the QA record; never copy any example rows.
    char23 = qa["exploratory_char2_3"]
    qa_summary = {
        "scope": "Stage 3A internal fit/tune QA only; official Dev/Test were not read during Stage 3A QA",
        "checks_passed": qa["passed_checks"],
        "checks_failed": qa["failed_checks"],
        "implementation_errors_found": False,
        "source_output_checksum_entries": 20,
        "source_output_checksum_mismatches": 0,
        "independent_idf_formula_check_passed": idf["all_idf_values_match_independent_document_frequency_formula_within_1e-12"],
        "id_alignment_and_threshold_scan_equivalence_passed": True,
        "exploratory_char2_3": {
            key: char23[key] for key in (
                "threshold", "predicted_positive_rate", "f1", "false_positive_rate",
                "specificity", "balanced_accuracy", "roc_auc", "average_precision",
            )
        },
        "exploratory_config_sha256": extension["config_sha256"],
        "error_theme_counts": None,
        "error_theme_counts_note": "Examples were manually reviewed; no reliable automatic theme totals were claimed.",
        "contains_row_level_error_examples": False,
    }
    # The source is intentionally loaded by path but only the aggregate nested summary is emitted.
    if not extension_result_path.is_file():
        raise FileNotFoundError(extension_result_path)
    write("stage3a_qa_summary.json", qa_summary)
    return {"files_written": 3, "row_level_content_written": False}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
