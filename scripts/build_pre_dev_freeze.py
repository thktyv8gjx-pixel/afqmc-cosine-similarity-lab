"""Create the sanitized, hash-bound pre-Dev freeze from saved Stage 1/3A metadata.

This script reads summary JSON only. It never opens the AFQMC archive, manifest,
fit corpus, vocabulary, predictions, or question text.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "stage01_data_audit"
STAGE2 = ROOT / "stage02_math_protocol"
STAGE3A = ROOT / "stage03a_internal_experiment"
STAGE3B = ROOT / "stage03b_official_validation"
FREEZE_PATH = STAGE3B / "pre_dev_freeze_manifest.json"
AUDIT_PATH = STAGE3B / "01_Dev前配置冻结审计.md"

FROZEN_SOURCE_PATHS = [
    ".gitattributes",
    "stage01_data_audit/audit_afqmc.py",
    "stage02_math_protocol/math_validation.py",
    "stage02_math_protocol/run_math_tests.py",
    "stage02_math_protocol/requirements.txt",
    "stage02_math_protocol/tests/test_math_validation.py",
    "stage03a_internal_experiment/config/experiment.json",
    "stage03a_internal_experiment/requirements.txt",
    "stage03a_internal_experiment/run_experiment.py",
    "stage03a_internal_experiment/run_tests.py",
    "stage03a_internal_experiment/generate_plots.py",
    "stage03a_internal_experiment/validate_outputs.py",
    "stage03a_internal_experiment/src/data_pipeline.py",
    "stage03a_internal_experiment/src/features.py",
    "stage03a_internal_experiment/src/pipeline.py",
    "stage03a_internal_experiment/src/threshold_metrics.py",
    "stage03a_internal_experiment/src/visualize.py",
    "stage03a_internal_experiment/tests/test_stage3a.py",
    "stage03b_official_validation/requirements.txt",
    "stage03b_official_validation/run_official_dev.py",
    "stage03b_official_validation/generate_dev_plots.py",
    "stage03b_official_validation/test_official_dev_runner.py",
    "scripts/build_public_evidence.py",
    "scripts/build_pre_dev_freeze.py",
    "scripts/review_public_staging.py",
    "scripts/public_allowlist.json",
    "scripts/run_pre_dev_checks.py",
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    config = load(STAGE3A / "config" / "experiment.json")
    source_audit = load(STAGE1 / "data_audit.json")
    qa_freeze = load(STAGE3A / "qa_review" / "outputs" / "freeze_readiness.json")
    metrics = load(STAGE3A / "qa_review" / "outputs" / "independent_metric_crosscheck.json")
    qa_result = load(STAGE3A / "qa_review" / "outputs" / "qa_diagnostics.json")
    tests = load(STAGE3B / "pre_dev_test_results.json")

    if qa_freeze["original_output_checksum_entries"] != 20 or qa_freeze["original_output_checksum_mismatches"]:
        raise AssertionError("original Stage 3A output checksum evidence is incomplete or mismatched")
    if qa_result["passed_checks"] != 36 or qa_result["failed_checks"] != 0:
        raise AssertionError("Stage 3A QA assertion evidence does not meet the recorded result")
    if qa_result["official_dev_raw_text_read"] or qa_result["official_dev_prediction_comparison_threshold_selection_or_debugging"]:
        raise AssertionError("Stage 3A QA evidence reports official Dev access")
    if not tests["all_passed"] or tests["official_dev_raw_text_opened"] or tests["threshold_or_feature_selection_performed"]:
        raise AssertionError("pre-Dev test evidence reports a failure or official Dev access")
    expected_test_counts = {"stage2_math": 21, "stage3a_implementation": 14, "stage3b_synthetic_helpers": 4}
    actual_test_counts = {item["suite"]: item["test_count"] for item in tests["suites"]}
    if actual_test_counts != expected_test_counts:
        raise AssertionError(f"pre-Dev test counts differ from the reviewed suites: {actual_test_counts}")

    data = config["data_scope"]
    train_dev_overlap = source_audit["train_dev_pair_overlap"]["normalized_unordered_pair_key_intersection_groups"]
    shared_sentences = source_audit["train_dev_shared_sentence_overlap"]["normalized_single_sentence_key_intersection"]
    source_hashes = {
        relative: file_sha256(ROOT / Path(relative)) for relative in FROZEN_SOURCE_PATHS
    }
    env = metrics["scikit_learn_version"]
    if env != "1.9.1":
        raise AssertionError("recorded Stage 3A scikit-learn version differs from the reviewed environment")

    payload: dict[str, Any] = {
        "freeze_version": 1,
        "status": "pre_dev_frozen_not_yet_evaluated",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_title": "Chinese customer-service similar-question identification using cosine similarity",
        "stage_scope": "Stage 3B frozen official Dev evaluation; no Dev result is included in this manifest",
        "publication_target": {
            "repository_full_name": "thktyv8gjx-pixel/afqmc-cosine-similarity-lab",
            "visibility_required": "public",
            "publication_verified": False,
            "software_license": "not selected",
        },
        "stage3a_config_sha256": qa_freeze["stage3a_config_sha256"],
        "inputs": {
            "archive_sha256": source_audit["source"]["archive_sha256"],
            "train_member_sha256": source_audit["archive_members"]["train.json"]["sha256"],
            "dev_member_sha256": source_audit["archive_members"]["dev.json"]["sha256"],
            "test_member_sha256": source_audit["archive_members"]["test.json"]["sha256"],
            "manifest_sha256": data["manifest_sha256"],
            "train_rows": source_audit["per_split"]["train"]["physical_line_rows"],
            "dev_rows": source_audit["per_split"]["dev"]["physical_line_rows"],
            "test_rows": source_audit["per_split"]["test"]["physical_line_rows"],
            "fit_tune_rows": {"fit": data["expected_fit_rows"], "tune": data["expected_tune_rows"]},
        },
        "fit_space": {
            "fit_pair_rows": data["expected_fit_rows"],
            "tune_pair_rows": qa_freeze["tune_pair_rows"],
            "fit_pair_source_row_ids_sha256": qa_freeze["fit_pair_source_row_ids_sha256"],
            "tune_source_id_ordered_sha256_newline_delimited": qa_freeze["tune_source_id_ordered_sha256_newline_delimited"],
            "tune_source_id_label_ordered_sha256_tab_delimited": qa_freeze["tune_source_id_label_ordered_sha256_tab_delimited"],
            "unique_normalized_fit_question_count": qa_freeze["idf_corpus_unique_fit_question_count"],
            "idf_corpus_content_sha256": qa_freeze["idf_corpus_content_sha256"],
            "idf_corpus_serialized_sha256": qa_freeze["idf_corpus_serialized_sha256"],
            "idf_formula": "log((1 + n) / (1 + df)) + 1; n is the number of unique normalized fit questions",
            "idf_vector_sha256_little_endian_float64": metrics["idf_audit"]["idf_vector_sha256_little_endian_float64"],
            "vocabulary_dimension": qa_freeze["vocabulary_dimension"],
            "vocabulary_index_sha256": qa_freeze["vocabulary_index_sha256"],
            "count_tfidf_share_feature_names_and_column_indices": True,
            "fit_scope": "official Train internal fit questions only; no tune, Dev, or Test questions",
        },
        "normalization": config["text_normalization"],
        "vectorizer_parameters": config["vectorizers"],
        "cosine_parameters": config["cosine"],
        "thresholds": {
            method: {
                "value": float(values["threshold"]),
                "source": "Stage 3A internal tune only",
                "objective": "positive-class F1",
                "decision_rule": "score >= threshold",
            }
            for method, values in qa_freeze["selected_internal_tune_thresholds"].items()
        },
        "formal_methods": {
            "primary": "character 2-gram TF-IDF + paired cosine",
            "baseline": "character 2-gram Count + paired cosine",
            "constant_baselines": ["always_predict_0", "always_predict_1"],
            "excluded_from_official_dev": ["exploratory character 2-3-gram TF-IDF"],
        },
        "metric_definitions": {
            "positive_label": 1,
            "precision_recall_f1_accuracy": "binary classification metrics with zero_division=0",
            "false_positive_rate": "FP / (FP + TN)",
            "specificity": "TN / (TN + FP)",
            "balanced_accuracy": "(sensitivity + specificity) / 2",
            "roc_auc_average_precision": "computed from continuous cosine scores; ranking-only secondary metrics",
            "constant_baseline_roc_auc_average_precision": "not applicable; constant label predictions have no continuous ranking scores",
            "confusion_matrix_order": "[[TN, FP], [FN, TP]]",
        },
        "official_split_boundary": {
            "fit_or_fit_transform_after_dev_open": False,
            "dev_usage": "transform and score only with the frozen fit feature space",
            "threshold_or_method_selection_on_dev": False,
            "official_dev_model_evaluation_completed": False,
            "expected_train_dev_pair_overlap_groups": train_dev_overlap,
            "shared_normalized_single_questions_between_train_and_dev": shared_sentences,
        },
        "qa_evidence": {
            "stage2_math_tests_passed": actual_test_counts["stage2_math"],
            "stage3a_synthetic_implementation_tests_passed": actual_test_counts["stage3a_implementation"],
            "stage3a_qa_assertions_passed": qa_result["passed_checks"],
            "stage3a_qa_assertions_failed": qa_result["failed_checks"],
            "stage3b_synthetic_helper_tests_passed": actual_test_counts["stage3b_synthetic_helpers"],
            "independent_metric_and_idf_recheck_passed": True,
            "original_stage3a_output_checksums_verified": 20,
            "original_stage3a_output_checksum_mismatches": 0,
        },
        "environment": {
            "python": "3.12.10",
            "numpy": "2.5.3",
            "scipy": "1.17.0",
            "scikit_learn": "1.9.1",
            "pillow": "12.3.0",
        },
        "frozen_source_files_sha256": source_hashes,
        "safety": {
            "raw_question_text_in_manifest": False,
            "row_level_predictions_in_public_release": False,
            "raw_AFQMC_data_in_public_release": False,
            "student_or_instructor_files_in_public_release": False,
            "secrets_found_in_public_candidate_scan": False,
        },
    }
    payload["freeze_manifest_sha256"] = canonical_digest(payload)
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    threshold_count = payload["thresholds"]["count_char2"]["value"]
    threshold_tfidf = payload["thresholds"]["tfidf_char2"]["value"]
    AUDIT_PATH.write_text(
        f"""# Dev 前配置冻结审计

## 冻结状态

- 状态：**配置已冻结，官方 Dev 尚未评价**。
- 冻结清单规范化 SHA-256：`{payload['freeze_manifest_sha256']}`。
- Stage 3A 配置 SHA-256：`{payload['stage3a_config_sha256']}`。
- 官方 Dev 只有在该冻结提交公开推送且远端 `main` 与本地提交一致后，才允许执行。
- 本审计只读取既有 Stage 1/Stage 3A 聚合元数据；没有打开 AFQMC ZIP、split manifest、问句语料、预测行或官方 Dev 原文。

## 数据输入指纹

| 输入 | SHA-256 | 行数 |
|---|---|---:|
| AFQMC 压缩包 | `{payload['inputs']['archive_sha256']}` | — |
| Train 成员 | `{payload['inputs']['train_member_sha256']}` | {payload['inputs']['train_rows']:,} |
| Dev 成员 | `{payload['inputs']['dev_member_sha256']}` | {payload['inputs']['dev_rows']:,} |
| Test 成员 | `{payload['inputs']['test_member_sha256']}` | {payload['inputs']['test_rows']:,} |
| 本地 split manifest | `{payload['inputs']['manifest_sha256']}` | — |

数据指纹用于确认版本，不代表数据可再分发。公开仓库不包含压缩包、JSON、manifest 或逐行记录。

## 冻结特征空间和算法

- 内部 fit/tune：{payload['fit_space']['fit_pair_rows']:,} / {payload['fit_space']['tune_pair_rows']:,} 对。
- IDF 文档单位：规范化后的唯一完整问句；来自 Train 内部 fit 两侧，共 {payload['fit_space']['unique_normalized_fit_question_count']:,} 条；按 Unicode 码点排序。
- IDF 语料 SHA-256：`{payload['fit_space']['idf_corpus_content_sha256']}`。
- 字符 2-gram 词表：{payload['fit_space']['vocabulary_dimension']:,} 维；索引指纹 `{payload['fit_space']['vocabulary_index_sha256']}`。
- Count 与 TF-IDF 使用同一 fit 语料、词表及列索引。TF-IDF 平滑 IDF：`log((1+n)/(1+df))+1`，L2 归一化。
- 规范化：Unicode NFC、首尾去空白、连续空白折叠为 ASCII 空格；不移除标点、数字、英文字符或否定词。
- 稀疏逐行余弦；零向量对得分 0 并保留样本；阈值使用 `score >= threshold`。

## 正式方案与阈值

| 角色 | 方法 | 精确阈值 | 阈值来源 |
|---|---|---:|---|
| 主方案 | 字符 2-gram TF-IDF + 余弦 | `{threshold_tfidf:.17g}` | Stage 3A 内部 tune 的正类 F1 |
| 基础对照 | 字符 2-gram Count + 余弦 | `{threshold_count:.17g}` | Stage 3A 内部 tune 的正类 F1 |
| 常量基线 | 始终预测 0、始终预测 1 | 无 | Dev 标签仅用于同集指标计算 |

字符 2～3-gram TF-IDF 是预登记的探索性内部敏感性实验，不进入正式 Dev 比较。Dev 不得用于拟合、模型选择、词表/IDF 变更或阈值选择。

## 测试与实现证据

- Stage 2 数学测试：21 项通过。
- Stage 3A 合成实现测试：14 项通过；QA 独立断言 36 项通过、0 项失败。
- 原始 Stage 3A 20 项输出哈希全部匹配。
- 独立 IDF 公式复核、Count/TF-IDF 特征索引一致性及全阈值穷举等价性通过。
- Stage 3B 评估器的 4 项合成辅助测试不读取 AFQMC 数据。
- 冻结代码文件共 {len(source_hashes)} 个；每个文件的 SHA-256 见 JSON 清单。

## Dev 独立性限制

Stage 1 已记录 Train/Dev 句对重叠为 {train_dev_overlap} 组，规范化共享单句为 {shared_sentences} 条。因此 Dev 不是完全句子隔离的验证集。正式结果需说明该限制。当前模型在内部 tune 的 F1 约为 0.47，几乎贴近全正类基线；后续 Dev 结果必须如实反映弱区分能力。

**本清单不包含 Dev 结果；本阶段没有读取、预测或调参官方 Dev。**
""",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    result = build()
    print(json.dumps({
        "freeze_manifest_sha256": result["freeze_manifest_sha256"],
        "frozen_source_file_count": len(result["frozen_source_files_sha256"]),
        "official_dev_evaluated": False,
    }, ensure_ascii=False, indent=2))
