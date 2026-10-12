#!/usr/bin/env python3
"""Stage 1 AFQMC source, quality, overlap, and group-split audit.

This script deliberately does not construct text features, compute similarities,
make predictions, optimize thresholds, or calculate model metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {"train": 34334, "dev": 4316, "test": 3861}
SEED = 2026
SPLITS = ("train", "dev", "test")
TEXT_FIELDS = ("sentence1", "sentence2")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    """Conservative key-only normalization; source text is never rewritten."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def ordered_pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


def pair_group_id(first: str, second: str) -> str:
    """SHA-256 over versioned, length-prefixed sorted normalized UTF-8 strings."""
    left, right = ordered_pair(normalize_text(first), normalize_text(second))
    payload = bytearray(b"AFQMC-pair-v1\0")
    for part in (left.encode("utf-8"), right.encode("utf-8")):
        payload.extend(len(part).to_bytes(8, "big"))
        payload.extend(part)
    return hashlib.sha256(payload).hexdigest()


def invalid_pair_group_id(source_row_id: str) -> str:
    return hashlib.sha256(
        b"AFQMC-invalid-row-v1\0" + source_row_id.encode("utf-8")
    ).hexdigest()


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_cell(value) if isinstance(value, (list, dict, tuple, set)) else value
                             for key, value in row.items()})


def is_punctuation_only(value: str) -> bool:
    stripped = normalize_text(value)
    return bool(stripped) and all(unicodedata.category(char).startswith("P") for char in stripped)


def label_value(record: dict[str, Any] | None) -> str | None:
    if not record or "label" not in record:
        return None
    value = record["label"]
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int)) and str(value) in {"0", "1"}:
        return str(value)
    return None


def add_row_issues(row: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    record = row["record"]
    base = {"source_file": row["source_file"], "source_row_id": row["source_row_id"],
            "pair_group_id": row["pair_group_id"], "label": row["label"] or ""}
    if record is None:
        issues.append({**base, "issue": "json_parse_or_blank_line", "field": "", "text": "",
                       "detail": row["parse_error"]})
        return
    for field in (*TEXT_FIELDS, *(('label',) if row["official_split"] != "test" else ())):
        if field not in record:
            issues.append({**base, "issue": "missing_required_field", "field": field, "text": "",
                           "detail": "required field absent"})
        elif field in TEXT_FIELDS and not isinstance(record[field], str):
            issues.append({**base, "issue": "non_string_text", "field": field, "text": json_cell(record[field]),
                           "detail": f"actual_type={type(record[field]).__name__}"})
    for field in TEXT_FIELDS:
        value = record.get(field)
        if not isinstance(value, str):
            continue
        if value == "":
            issue = "empty_text"
        elif not value.strip():
            issue = "whitespace_only_text"
        elif is_punctuation_only(value):
            issue = "punctuation_only_text"
        elif len(normalize_text(value)) < 2:
            issue = "anomalously_short_text_lt_2_codepoints"
        else:
            continue
        issues.append({**base, "issue": issue, "field": field, "text": value,
                       "detail": f"raw_codepoints={len(value)}"})
    if row["official_split"] != "test":
        if "label" in record and label_value(record) is None:
            issues.append({**base, "issue": "illegal_or_missing_label", "field": "label",
                           "text": json_cell(record.get("label")),
                           "detail": "expected string or integer 0/1; test labels are not inferred"})


def load_archive(archive: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    member_meta: dict[str, dict[str, Any]] = {}
    parse_issues: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure in member {bad_member!r}")
        names = {Path(name).name: name for name in zf.namelist() if not name.endswith("/")}
        for split in SPLITS:
            name = f"{split}.json"
            if name not in names:
                raise RuntimeError(f"Official archive is missing expected member {name!r}; actual={sorted(names)}")
            info = zf.getinfo(names[name])
            raw = zf.read(names[name])
            member_meta[name] = {
                "archive_member": names[name],
                "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "sha256": sha256_bytes(raw),
            }
            content = raw.decode("utf-8-sig")
            lines = content.splitlines()
            for line_number, line in enumerate(lines, start=1):
                source_row_id = f"{split}:{name}:{line_number:06d}"
                record = None
                parse_error = ""
                if not line.strip():
                    parse_error = "blank physical line"
                else:
                    try:
                        parsed = json.loads(line)
                        if not isinstance(parsed, dict):
                            parse_error = f"top-level JSON value is {type(parsed).__name__}, expected object"
                        else:
                            record = parsed
                    except Exception as error:  # retain row-level evidence, do not skip silently
                        parse_error = f"{type(error).__name__}: {error}"
                row = {
                    "official_split": split,
                    "source_file": name,
                    "source_row_id": source_row_id,
                    "line_number": line_number,
                    "record": record,
                    "parse_error": parse_error,
                    "label": label_value(record),
                }
                if record is not None and all(isinstance(record.get(field), str) for field in TEXT_FIELDS):
                    row["pair_group_id"] = pair_group_id(record["sentence1"], record["sentence2"])
                else:
                    row["pair_group_id"] = invalid_pair_group_id(source_row_id)
                split_rows[split].append(row)
                if parse_error:
                    parse_issues.append({"source_file": name, "source_row_id": source_row_id,
                                         "issue": "json_parse_or_blank_line", "detail": parse_error})
    return split_rows, member_meta, parse_issues


def row_is_text_valid(row: dict[str, Any]) -> bool:
    record = row["record"]
    return bool(record) and all(isinstance(record.get(field), str) for field in TEXT_FIELDS)


def exact_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    if not row_is_text_valid(row):
        return None
    return ordered_pair(row["record"]["sentence1"], row["record"]["sentence2"])


def normalized_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    if not row_is_text_valid(row):
        return None
    return ordered_pair(normalize_text(row["record"]["sentence1"]),
                        normalize_text(row["record"]["sentence2"]))


def row_record_content_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    if not row_is_text_valid(row):
        return None
    record = row["record"]
    return (record["sentence1"], record["sentence2"], row["label"])


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None,
                "p75": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(values)
    def at(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"count": len(ordered), "min": ordered[0], "p25": at(.25), "median": at(.5),
            "p75": at(.75), "p90": at(.9), "max": ordered[-1], "mean": statistics.fmean(ordered)}


def length_histogram(values: list[int]) -> dict[str, int]:
    bins = {"0": 0, "1": 0, "2": 0, "3-5": 0, "6-10": 0, "11-20": 0,
            "21-50": 0, "51+": 0}
    for value in values:
        key = ("0" if value == 0 else "1" if value == 1 else "2" if value == 2 else
               "3-5" if value <= 5 else "6-10" if value <= 10 else
               "11-20" if value <= 20 else "21-50" if value <= 50 else "51+")
        bins[key] += 1
    return bins


def group_rows(rows: list[dict[str, Any]], key_function) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = key_function(row)
        if key is not None:
            groups[key].append(row)
    return groups


def duplicate_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text_rows = [row for row in rows if row_is_text_valid(row)]
    full_record_groups = group_rows(text_rows, row_record_content_key)
    exact_order_groups = group_rows(text_rows, lambda r: (r["record"]["sentence1"], r["record"]["sentence2"]))
    exact_unordered_groups = group_rows(text_rows, exact_pair)
    norm_unordered_groups = group_rows(text_rows, normalized_pair)
    normalized_sentence_counts: Counter[str] = Counter()
    exact_sentence_counts: Counter[str] = Counter()
    sentence_pair_keys: dict[str, set[str]] = defaultdict(set)
    for row in text_rows:
        for field in TEXT_FIELDS:
            original = row["record"][field]
            normalized = normalize_text(original)
            exact_sentence_counts[original] += 1
            normalized_sentence_counts[normalized] += 1
            sentence_pair_keys[normalized].add(row["pair_group_id"])
    reversed_groups = 0
    for members in exact_unordered_groups.values():
        orientations = {(r["record"]["sentence1"], r["record"]["sentence2"]) for r in members}
        if len(orientations) > 1 and any((b, a) in orientations for a, b in orientations):
            reversed_groups += 1
    label_conflicts = []
    for key, members in norm_unordered_groups.items():
        labels = sorted({row["label"] for row in members if row["label"] is not None})
        if len(labels) > 1:
            label_conflicts.append({"pair": key, "rows": members, "labels": labels})
    extra_rows = lambda groups: sum(len(members) - 1 for members in groups.values() if len(members) > 1)
    return {
        "parsed_text_pair_rows": len(text_rows),
        "exact_duplicate_record_groups": sum(len(v) > 1 for v in full_record_groups.values()),
        "exact_duplicate_record_extra_rows": extra_rows(full_record_groups),
        "exact_ordered_pair_groups_repeated": sum(len(v) > 1 for v in exact_order_groups.values()),
        "exact_ordered_pair_extra_rows": extra_rows(exact_order_groups),
        "exact_unordered_pair_groups_repeated": sum(len(v) > 1 for v in exact_unordered_groups.values()),
        "exact_unordered_pair_extra_rows": extra_rows(exact_unordered_groups),
        "exact_unordered_pair_groups_containing_both_directions": reversed_groups,
        "normalized_unordered_pair_groups_repeated": sum(len(v) > 1 for v in norm_unordered_groups.values()),
        "normalized_unordered_pair_extra_rows": extra_rows(norm_unordered_groups),
        "unique_exact_sentence_texts": len(exact_sentence_counts),
        "unique_normalized_sentence_texts": len(normalized_sentence_counts),
        "normalized_sentence_texts_reused_in_multiple_distinct_pairs": sum(len(v) > 1 for v in sentence_pair_keys.values()),
        "normalized_sentence_texts_with_repeated_occurrences": sum(count > 1 for count in normalized_sentence_counts.values()),
        "unique_exact_unordered_pairs": len(exact_unordered_groups),
        "unique_normalized_unordered_pairs": len(norm_unordered_groups),
        "label_conflict_groups_normalized_unordered_pair": len(label_conflicts),
        "label_conflict_rows": sum(len(conflict["rows"]) for conflict in label_conflicts),
        "_groups": {"full_record": full_record_groups, "exact_ordered": exact_order_groups,
                    "exact_unordered": exact_unordered_groups, "normalized_unordered": norm_unordered_groups,
                    "label_conflicts": label_conflicts},
    }


def field_summary(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    parsed = [row for row in rows if row["record"] is not None]
    field_names = sorted({key for row in parsed for key in row["record"]})
    fields: dict[str, Any] = {}
    for name in field_names:
        types = Counter(type(row["record"][name]).__name__ for row in parsed if name in row["record"])
        missing = sum(name not in row["record"] for row in parsed)
        fields[name] = {"present": len(parsed) - missing, "missing": missing,
                        "types": dict(sorted(types.items()))}
    label_counts = Counter(row["label"] if row["label"] is not None else "missing_or_illegal"
                           for row in rows if row["record"] is not None)
    legal_label_count = sum(label_counts.get(label, 0) for label in ("0", "1"))
    if split == "test":
        labels = {"label_field_present_rows": fields.get("label", {}).get("present", 0),
                  "unlabeled_rows": len(parsed), "legal_0_1_total": None,
                  "proportions_over_legal_labels": None,
                  "note": "Test has no label field; no labels were inferred or scored."}
    else:
        labels = {"counts": dict(label_counts), "legal_0_1_total": legal_label_count,
                  "proportions_over_legal_labels": {
                      label: (label_counts.get(label, 0) / legal_label_count if legal_label_count else None)
                      for label in ("0", "1")}}
    text_stats: dict[str, Any] = {}
    for field in TEXT_FIELDS:
        values = [row["record"][field] for row in parsed if isinstance(row["record"].get(field), str)]
        lengths = [len(value) for value in values]
        text_stats[field] = {
            "valid_string_count": len(values),
            "empty_count": sum(value == "" for value in values),
            "whitespace_only_count": sum(bool(value) and not value.strip() for value in values),
            "punctuation_only_count": sum(is_punctuation_only(value) for value in values),
            "anomalously_short_lt_2_codepoints_count": sum(0 < len(normalize_text(value)) < 2 for value in values),
            "raw_unicode_codepoint_length": quantiles(lengths),
            "raw_unicode_codepoint_length_histogram": length_histogram(lengths),
        }
    return {"expected_reference_rows": EXPECTED[split], "physical_line_rows": len(rows),
            "json_object_rows": len(parsed), "parse_or_blank_line_failures": len(rows) - len(parsed),
            "fields": fields, "labels": labels, "text_quality": text_stats}


def cross_pair_overlap(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_exact = group_rows(train, exact_pair)
    dev_exact = group_rows(dev, exact_pair)
    train_norm = group_rows(train, normalized_pair)
    dev_norm = group_rows(dev, normalized_pair)
    common_exact = set(train_exact) & set(dev_exact)
    common_norm = set(train_norm) & set(dev_norm)
    detail = []
    train_affected: set[str] = set()
    dev_affected: set[str] = set()
    same_dir_groups = reverse_dir_groups = 0
    for key in sorted(common_norm):
        left_rows, right_rows = train_norm[key], dev_norm[key]
        same = False
        reverse = False
        exact_same = False
        exact_reverse = False
        for tr in left_rows:
            a, b = tr["record"]["sentence1"], tr["record"]["sentence2"]
            for dv in right_rows:
                x, y = dv["record"]["sentence1"], dv["record"]["sentence2"]
                if (a, b) == (x, y):
                    exact_same = True
                if (a, b) == (y, x):
                    exact_reverse = True
                na, nb = normalize_text(a), normalize_text(b)
                nx, ny = normalize_text(x), normalize_text(y)
                if (na, nb) == (nx, ny):
                    same = True
                if (na, nb) == (ny, nx):
                    reverse = True
        same_dir_groups += int(same)
        reverse_dir_groups += int(reverse)
        train_affected.update(row["source_row_id"] for row in left_rows)
        dev_affected.update(row["source_row_id"] for row in right_rows)
        detail.append({
            "pair_group_id": pair_group_id(*key), "normalized_sentence_a": key[0],
            "normalized_sentence_b": key[1], "exact_unordered_pair_overlap": key in common_exact,
            "same_direction_normalized_overlap": same, "reverse_direction_normalized_overlap": reverse,
            "same_direction_exact_overlap": exact_same, "reverse_direction_exact_overlap": exact_reverse,
            "train_row_ids": [row["source_row_id"] for row in left_rows],
            "train_labels": [row["label"] for row in left_rows],
            "dev_row_ids": [row["source_row_id"] for row in right_rows],
            "dev_labels": [row["label"] for row in right_rows],
        })
    stats = {
        "exact_unordered_pair_key_intersection_groups": len(common_exact),
        "normalized_unordered_pair_key_intersection_groups": len(common_norm),
        "normalized_same_direction_overlap_groups": same_dir_groups,
        "normalized_reverse_direction_overlap_groups": reverse_dir_groups,
        "train_rows_affected_by_normalized_pair_overlap": len(train_affected),
        "dev_rows_affected_by_normalized_pair_overlap": len(dev_affected),
        "train_row_ids_affected": sorted(train_affected), "dev_row_ids_affected": sorted(dev_affected),
    }
    return stats, detail


def cross_sentence_overlap(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def map_sentences(rows: list[dict[str, Any]], normalized: bool):
        result: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"raw": set(), "rows": set()})
        for row in rows:
            if not row_is_text_valid(row):
                continue
            for field in TEXT_FIELDS:
                raw = row["record"][field]
                key = normalize_text(raw) if normalized else raw
                result[key]["raw"].add(raw)
                result[key]["rows"].add(row["source_row_id"])
        return result
    t_exact, d_exact = map_sentences(train, False), map_sentences(dev, False)
    t_norm, d_norm = map_sentences(train, True), map_sentences(dev, True)
    common_exact, common_norm = set(t_exact) & set(d_exact), set(t_norm) & set(d_norm)
    detail = []
    for key in sorted(common_norm):
        detail.append({"normalized_sentence": key,
                       "exact_sentence_overlap": key in common_exact,
                       "train_raw_forms": sorted(t_norm[key]["raw"]),
                       "dev_raw_forms": sorted(d_norm[key]["raw"]),
                       "train_row_ids": sorted(t_norm[key]["rows"]),
                       "dev_row_ids": sorted(d_norm[key]["rows"])})
    train_affected = {rid for key in common_norm for rid in t_norm[key]["rows"]}
    dev_affected = {rid for key in common_norm for rid in d_norm[key]["rows"]}
    return ({"exact_single_sentence_key_intersection": len(common_exact),
             "normalized_single_sentence_key_intersection": len(common_norm),
             "train_rows_affected_by_normalized_shared_sentence": len(train_affected),
             "dev_rows_affected_by_normalized_shared_sentence": len(dev_affected)}, detail)


def build_internal_split(train_rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    eligible = [row for row in train_rows if row_is_text_valid(row) and row["label"] in {"0", "1"}]
    if not eligible:
        raise RuntimeError("No structurally valid, legally labeled Train rows; cannot form internal split")
    from sklearn.model_selection import StratifiedGroupKFold

    groups = [row["pair_group_id"] for row in eligible]
    labels = [row["label"] for row in eligible]
    if len(set(labels)) < 2 or len(set(groups)) < 5:
        raise RuntimeError("Insufficient label classes or pair groups for approved 5-fold group split")

    def once() -> tuple[dict[str, str], list[int], list[int]]:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        fit_index, tune_index = next(splitter.split([[0]] * len(eligible), labels, groups))
        fit_groups = {groups[int(index)] for index in fit_index}
        tune_groups = {groups[int(index)] for index in tune_index}
        crossing = fit_groups & tune_groups
        if crossing:
            raise RuntimeError(f"pair group split across fit/tune: {sorted(crossing)[:5]}")
        mapping = {group_id: "fit" for group_id in fit_groups}
        mapping.update({group_id: "tune" for group_id in tune_groups})
        return mapping, list(map(int, fit_index)), list(map(int, tune_index))

    map_a, fit_a, tune_a = once()
    map_b, fit_b, tune_b = once()
    reproducible = map_a == map_b and fit_a == fit_b and tune_a == tune_b
    if not reproducible:
        raise RuntimeError("Repeated split generation is not deterministic")
    row_split = {row["source_row_id"]: map_a[row["pair_group_id"]] for row in eligible}
    # Preserve the grouping of structurally invalid rows if their valid counterpart exists.
    for row in train_rows:
        if row["source_row_id"] not in row_split and row["pair_group_id"] in map_a:
            row_split[row["source_row_id"]] = map_a[row["pair_group_id"]]
    fit_rows = [row for row in eligible if row_split[row["source_row_id"]] == "fit"]
    tune_rows = [row for row in eligible if row_split[row["source_row_id"]] == "tune"]
    group_roles: dict[str, set[str]] = defaultdict(set)
    for row in eligible:
        group_roles[row["pair_group_id"]].add(row_split[row["source_row_id"]])
    no_group_crossing = all(len(roles) == 1 for roles in group_roles.values())
    if not no_group_crossing:
        raise RuntimeError("A normalized unordered pair group crossed fit/tune")
    class_counts = {split: dict(Counter(row["label"] for row in subset))
                    for split, subset in (("fit", fit_rows), ("tune", tune_rows))}
    totals = {"fit": len(fit_rows), "tune": len(tune_rows)}
    ratios = {key: value / len(eligible) for key, value in totals.items()}
    return row_split, {
        "algorithm": "sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026); first fold used as tune, remaining rows fit",
        "target_fit_fraction": 0.8, "target_tune_fraction": 0.2,
        "eligible_train_rows": len(eligible), "unique_pair_groups": len(set(groups)),
        "rows": totals, "row_proportions": ratios, "label_counts": class_counts,
        "actual_tune_fraction_deviation_from_0_20": ratios["tune"] - 0.2,
        "same_pair_group_crosses_internal_sets": not no_group_crossing,
        "repeated_generation_identical": reproducible,
        "conflicted_pair_groups_kept_whole": True,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="stage01_data_audit output directory")
    parser.add_argument("--archive", type=Path, default=None, help="AFQMC official ZIP archive")
    args = parser.parse_args()
    root = args.root.resolve()
    archive = (args.archive or root / "raw" / "afqmc_public.zip").resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        raise FileNotFoundError(f"AFQMC archive not found: {archive}")

    split_rows, member_meta, parse_issues = load_archive(archive)
    all_rows = [row for split in SPLITS for row in split_rows[split]]
    issues: list[dict[str, Any]] = []
    for row in all_rows:
        add_row_issues(row, issues)

    per_split: dict[str, Any] = {}
    duplicate_reports: dict[str, Any] = {}
    all_conflict_rows: list[dict[str, Any]] = []
    duplicate_record_rows: list[dict[str, Any]] = []
    duplicate_pair_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        rows = split_rows[split]
        per_split[split] = field_summary(rows, split)
        summary = duplicate_summaries(rows)
        duplicate_reports[split] = {key: value for key, value in summary.items() if not key.startswith("_")}
        for members in summary["_groups"]["full_record"].values():
            if len(members) > 1:
                duplicate_record_rows.append({"official_split": split,
                                              "source_row_ids": [r["source_row_id"] for r in members],
                                              "sentence1": members[0]["record"]["sentence1"],
                                              "sentence2": members[0]["record"]["sentence2"],
                                              "label": members[0]["label"] or ""})
        for key, members in summary["_groups"]["normalized_unordered"].items():
            if len(members) > 1:
                orientations = {(r["record"]["sentence1"], r["record"]["sentence2"]) for r in members}
                duplicate_pair_rows.append({"official_split": split,
                                            "pair_group_id": pair_group_id(*key),
                                            "normalized_sentence_a": key[0], "normalized_sentence_b": key[1],
                                            "row_count": len(members),
                                            "both_directions_present": any((b, a) in orientations for a, b in orientations),
                                            "source_row_ids": [r["source_row_id"] for r in members],
                                            "labels": [r["label"] for r in members]})
        for conflict in summary["_groups"]["label_conflicts"]:
            members = conflict["rows"]
            all_conflict_rows.append({"official_scope": split, "pair_group_id": pair_group_id(*conflict["pair"]),
                                      "normalized_sentence_a": conflict["pair"][0],
                                      "normalized_sentence_b": conflict["pair"][1],
                                      "labels": conflict["labels"],
                                      "source_files": [r["source_file"] for r in members],
                                      "source_row_ids": [r["source_row_id"] for r in members],
                                      "raw_pairs": [[r["record"]["sentence1"], r["record"]["sentence2"]] for r in members]})

    train_dev_pair_stats, train_dev_pair_detail = cross_pair_overlap(split_rows["train"], split_rows["dev"])
    train_dev_sentence_stats, train_dev_sentence_detail = cross_sentence_overlap(split_rows["train"], split_rows["dev"])
    internal_row_split, split_info = build_internal_split(split_rows["train"])

    manifest = []
    for row in all_rows:
        split = row["official_split"]
        if split == "train":
            internal = internal_row_split.get(row["source_row_id"], "excluded_invalid")
        elif split == "dev":
            internal = "official_dev"
        else:
            internal = "official_test"
        manifest.append({"source_file": row["source_file"], "source_row_id": row["source_row_id"],
                         "official_split": split, "internal_split": internal,
                         "label": row["label"] or "", "pair_group_id": row["pair_group_id"]})
    manifest_ids = [row["source_row_id"] for row in manifest]
    unique_manifest_ids = len(set(manifest_ids)) == len(manifest_ids)
    source_coverage = len(manifest) == sum(len(split_rows[s]) for s in SPLITS) and unique_manifest_ids
    train_coverage = sum(row["official_split"] == "train" for row in manifest) == len(split_rows["train"])
    fit_tune_roles: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        if row["official_split"] == "train" and row["internal_split"] in {"fit", "tune"}:
            fit_tune_roles[row["pair_group_id"]].add(row["internal_split"])
    group_integrity = all(len(roles) == 1 for roles in fit_tune_roles.values())
    dev_isolation = all(row["internal_split"] == "official_dev" for row in manifest if row["official_split"] == "dev")
    train_only_internal = all(row["official_split"] == "train" for row in manifest
                              if row["internal_split"] in {"fit", "tune"})
    split_checks = {
        "manifest_covers_all_physical_jsonl_rows_once": source_coverage,
        "all_official_train_rows_assigned_once_or_explicitly_marked_invalid": train_coverage,
        "same_pair_group_not_split_fit_vs_tune": group_integrity,
        "repeat_generation_identical": split_info["repeated_generation_identical"],
        "official_dev_kept_separate": dev_isolation,
        "only_official_train_in_internal_fit_tune": train_only_internal,
        "all_checks_pass": all((source_coverage, train_coverage, group_integrity,
                                 split_info["repeated_generation_identical"], dev_isolation, train_only_internal)),
    }
    if not split_checks["all_checks_pass"]:
        raise RuntimeError(f"Manifest/split validation failed: {split_checks}")

    audit = {
        "audit_version": "stage01-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "source_url": "https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip",
            "official_readme": "https://github.com/CLUEbenchmark/CLUE/blob/master/README.md",
            "archive_path": str(archive), "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive), "zip_crc_test": "PASS",
            "redistribution_license": "not specified on the checked official CLUE AFQMC README/download page; do not redistribute raw files pending clarification",
        },
        "archive_members": member_meta,
        "expected_reference_counts": EXPECTED,
        "per_split": per_split,
        "duplicates_within_split": duplicate_reports,
        "train_dev_pair_overlap": train_dev_pair_stats,
        "train_dev_shared_sentence_overlap": train_dev_sentence_stats,
        "internal_split": split_info,
        "split_validation": split_checks,
        "issues": {
            "parse_or_blank_line_rows": len(parse_issues),
            "schema_label_text_anomaly_rows_or_fields": len(issues),
            "data_issue_detail_csv": "data_issues.csv",
            "duplicate_record_detail_csv": "duplicate_records.csv",
            "duplicate_pair_detail_csv": "duplicate_pairs.csv",
            "label_conflict_detail_csv": "label_conflicts.csv",
            "train_dev_pair_overlap_detail_csv": "train_dev_pair_overlap.csv",
            "train_dev_shared_sentence_detail_csv": "train_dev_sentence_overlap.csv",
        },
        "constraints": {
            "source_data_modified": False,
            "test_labels_inferred": False,
            "feature_vocabulary_fitted": False,
            "cosine_similarity_computed": False,
            "predictions_threshold_or_model_metrics_computed": False,
            "official_dev_used_for_selection": False,
        },
    }

    (root / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(root / "split_manifest.csv", manifest,
              ["source_file", "source_row_id", "official_split", "internal_split", "label", "pair_group_id"])
    write_csv(root / "data_issues.csv", issues,
              ["source_file", "source_row_id", "pair_group_id", "label", "issue", "field", "text", "detail"])
    write_csv(root / "duplicate_records.csv", duplicate_record_rows,
              ["official_split", "source_row_ids", "sentence1", "sentence2", "label"])
    write_csv(root / "duplicate_pairs.csv", duplicate_pair_rows,
              ["official_split", "pair_group_id", "normalized_sentence_a", "normalized_sentence_b",
               "row_count", "both_directions_present", "source_row_ids", "labels"])
    write_csv(root / "label_conflicts.csv", all_conflict_rows,
              ["official_scope", "pair_group_id", "normalized_sentence_a", "normalized_sentence_b",
               "labels", "source_files", "source_row_ids", "raw_pairs"])
    write_csv(root / "train_dev_pair_overlap.csv", train_dev_pair_detail,
              ["pair_group_id", "normalized_sentence_a", "normalized_sentence_b",
               "exact_unordered_pair_overlap", "same_direction_normalized_overlap",
               "reverse_direction_normalized_overlap", "same_direction_exact_overlap",
               "reverse_direction_exact_overlap", "train_row_ids", "train_labels", "dev_row_ids", "dev_labels"])
    write_csv(root / "train_dev_sentence_overlap.csv", train_dev_sentence_detail,
              ["normalized_sentence", "exact_sentence_overlap", "train_raw_forms", "dev_raw_forms",
               "train_row_ids", "dev_row_ids"])

    archive_relative = archive.relative_to(root.parent).as_posix()
    checksum_lines = [f"{audit['source']['archive_sha256']}  {archive_relative}",
                      "# SHA-256 of uncompressed ZIP members (verify after extracting the named member):"]
    for name, meta in sorted(member_meta.items()):
        checksum_lines.append(f"# {meta['sha256']}  {archive_relative}::{name}  bytes={meta['uncompressed_bytes']}")
    (root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    environment = {"python": sys.version, "platform": platform.platform(),
                   "scikit_learn": package_version("scikit-learn"), "numpy": package_version("numpy"),
                   "pandas": package_version("pandas"), "script_sha256": sha256_file(Path(__file__).resolve())}
    run_record = {"generated_utc": audit["generated_utc"],
                  "command": f"python {Path(__file__).resolve()} --archive {archive}",
                  "working_directory": str(Path.cwd()), "archive_path": str(archive),
                  "environment": environment, "validation": split_checks,
                  "split_manifest_sha256": sha256_file(root / "split_manifest.csv"),
                  "split_regeneration_check": "two independent deterministic generations in this execution were identical",
                  "outputs": ["data_audit.json", "checksums.sha256", "split_manifest.csv", "data_issues.csv",
                              "duplicate_records.csv", "duplicate_pairs.csv", "label_conflicts.csv",
                              "train_dev_pair_overlap.csv", "train_dev_sentence_overlap.csv"]}
    (root / "run_record.json").write_text(json.dumps(run_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "audit_json": str(root / "data_audit.json"),
                      "split_manifest": str(root / "split_manifest.csv"),
                      "rows": {name: per_split[name]["physical_line_rows"] for name in SPLITS},
                      "labels": {name: per_split[name]["labels"] for name in ("train", "dev")},
                      "duplicate_summary": duplicate_reports,
                      "train_dev_pair_overlap": {k: v for k, v in train_dev_pair_stats.items()
                                                 if not k.endswith("row_ids_affected")},
                      "train_dev_sentence_overlap": train_dev_sentence_stats,
                      "internal_split": split_info, "validation": split_checks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"AUDIT_FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
