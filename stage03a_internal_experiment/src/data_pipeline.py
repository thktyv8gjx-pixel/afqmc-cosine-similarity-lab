from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("question text must be a string")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip())


def read_train_manifest(path: Path) -> list[dict[str, str]]:
    """Read only Train manifest records; no official Dev raw text is involved."""
    selected: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_file", "source_row_id", "official_split", "internal_split", "label", "pair_group_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"manifest is missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if row["official_split"] != "train":
                continue
            if row["source_file"] != "train.json" or row["internal_split"] not in {"fit", "tune"}:
                raise ValueError(f"unexpected Train manifest assignment: {row['source_row_id']}")
            if row["source_row_id"] in seen_ids:
                raise ValueError(f"duplicate Train source_row_id: {row['source_row_id']}")
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"illegal Train label for {row['source_row_id']}: {row['label']!r}")
            seen_ids.add(row["source_row_id"])
            selected.append({key: row[key] for key in required})
    return selected


def read_only_train_member(
    archive_path: Path,
    expected_archive_sha256: str,
    expected_train_sha256: str,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], str]:
    """Hash the archive, then open and parse only train.json from the ZIP."""
    actual_archive_hash = sha256_file(archive_path)
    if actual_archive_hash != expected_archive_sha256:
        raise ValueError(f"archive SHA-256 mismatch: {actual_archive_hash}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        info = archive.getinfo("train.json")
        with archive.open(info, "r") as member:
            raw = member.read()
    actual_train_hash = hashlib.sha256(raw).hexdigest()
    if actual_train_hash != expected_train_sha256:
        raise ValueError(f"train.json SHA-256 mismatch: {actual_train_hash}")

    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line in train.json at {line_no}")
        item = json.loads(line.decode("utf-8"))
        if not isinstance(item, dict):
            raise ValueError(f"Train line {line_no} is not a JSON object")
        if not isinstance(item.get("sentence1"), str) or not isinstance(item.get("sentence2"), str):
            raise ValueError(f"Train line {line_no} has invalid sentence fields")
        label = str(item.get("label"))
        if label not in {"0", "1"}:
            raise ValueError(f"Train line {line_no} has illegal label: {label!r}")
        records.append(
            {
                "source_row_id": f"train:train.json:{line_no:06d}",
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "label": int(label),
            }
        )
    if len(records) != expected_rows:
        raise ValueError(f"Train row count mismatch: {len(records)} != {expected_rows}")
    return records, actual_train_hash


def align_internal_train_rows(
    raw_train_rows: list[dict[str, Any]], manifest_train_rows: list[dict[str, str]], expected_rows: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join by stable row ID, never by position after filtering or sorting."""
    manifest_by_id = {item["source_row_id"]: item for item in manifest_train_rows}
    if len(manifest_by_id) != len(manifest_train_rows):
        raise ValueError("Train manifest row IDs are not unique")
    if len(raw_train_rows) != expected_rows or len(manifest_by_id) != expected_rows:
        raise ValueError("Train raw/manifest row counts do not match the frozen count")
    joined_ids: set[str] = set()
    fit: list[dict[str, Any]] = []
    tune: list[dict[str, Any]] = []
    for raw in raw_train_rows:
        row_id = raw["source_row_id"]
        if row_id in joined_ids:
            raise ValueError(f"duplicate raw Train row ID: {row_id}")
        joined_ids.add(row_id)
        meta = manifest_by_id.get(row_id)
        if meta is None:
            raise ValueError(f"raw Train row is absent from manifest: {row_id}")
        if int(meta["label"]) != raw["label"]:
            raise ValueError(f"raw/manifest label mismatch for {row_id}")
        row = {**raw, "internal_split": meta["internal_split"], "pair_group_id": meta["pair_group_id"]}
        (fit if meta["internal_split"] == "fit" else tune).append(row)
    if joined_ids != set(manifest_by_id):
        raise ValueError("raw Train IDs and Train manifest IDs are not a complete one-to-one match")
    group_roles: dict[str, set[str]] = defaultdict(set)
    for row in (*fit, *tune):
        group_roles[row["pair_group_id"]].add(row["internal_split"])
    crossings = [group for group, roles in group_roles.items() if len(roles) != 1]
    if crossings:
        raise ValueError(f"pair_group_id crosses fit/tune: {len(crossings)} groups")
    return fit, tune


def build_unique_fit_corpus(fit_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Unique normalized full questions, with sorted provenance and a canonical digest."""
    provenance: dict[str, set[str]] = defaultdict(set)
    for row in fit_rows:
        for side in ("sentence1", "sentence2"):
            text = normalize_text(row[side])
            provenance[text].add(row["source_row_id"])
    texts = sorted(provenance)
    docs = [
        {"doc_index": index, "normalized_text": text, "source_row_ids": sorted(provenance[text])}
        for index, text in enumerate(texts)
    ]
    digest = hashlib.sha256(b"AFQMC-IDF-unique-normalized-v1\0")
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return docs, digest.hexdigest()


def write_fit_corpus(path: Path, docs: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
    return sha256_file(path)


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["label"]) for row in rows)
    return {str(label): counts.get(str(label), 0) for label in (0, 1)}
