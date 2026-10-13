"""Strict freeze validation and append-only evaluation-attempt evidence."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METHODS = ("count_char2", "tfidf_char2")
PROTECTED_FIELDS = (
    "stage3a_config_sha256", "inputs", "fit_space", "normalization",
    "vectorizer_parameters", "cosine_parameters", "thresholds", "formal_methods",
    "metric_definitions", "official_split_boundary", "environment",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("SHA-256 must be exactly 64 lowercase hexadecimal characters")
    return value


def validate_fingerprint_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if "sha256" in key and isinstance(item, str):
                require_sha256(item)
            if key == "frozen_source_files_sha256":
                if not isinstance(item, dict):
                    raise ValueError("source fingerprint map must be an object")
                for expected in item.values():
                    require_sha256(expected)
            validate_fingerprint_fields(item)
    elif isinstance(value, list):
        for item in value:
            validate_fingerprint_fields(item)


def validate_manifest_digest(freeze: dict[str, Any]) -> None:
    validate_fingerprint_fields(freeze)
    expected = require_sha256(freeze.get("freeze_manifest_sha256"))
    payload = {key: value for key, value in freeze.items() if key != "freeze_manifest_sha256"}
    if digest(canonical_bytes(payload)) != expected:
        raise AssertionError("canonical freeze manifest fingerprint mismatch")


def validate_thresholds(freeze: dict[str, Any]) -> dict[str, float]:
    entries = freeze.get("thresholds")
    if not isinstance(entries, dict) or set(entries) != set(METHODS):
        raise ValueError("threshold map must contain exactly the two registered methods")
    result: dict[str, float] = {}
    for method in METHODS:
        entry = entries[method]
        if not isinstance(entry, dict) or "value" not in entry:
            raise ValueError("each threshold must be an object containing value")
        value = entry["value"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("threshold value must be a finite JSON number; bool and null are forbidden")
        result[method] = float(value)
    return result


def method_consistency(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    checks = {key: v1.get(key) == v2.get(key) for key in PROTECTED_FIELDS}
    threshold_bits = {method: float(v1["thresholds"][method]["value"]).hex() == float(v2["thresholds"][method]["value"]).hex() for method in METHODS}
    return {"checks": checks, "threshold_float64_hex_equal": threshold_bits,
            "passed": all(checks.values()) and all(threshold_bits.values())}


def verify_recovery_reference(freeze: dict[str, Any], root: Path) -> None:
    if freeze.get("freeze_version") != 2:
        raise ValueError("controlled recovery requires freeze version 2")
    recovery = freeze["recovery"]
    path = root / "stage03b_official_validation" / "pre_dev_freeze_manifest.json"
    raw = path.read_bytes()
    if digest(raw) != require_sha256(recovery["v1_manifest_raw_bytes_sha256"]):
        raise AssertionError("v1 manifest raw-byte fingerprint mismatch")
    v1 = json.loads(raw)
    validate_manifest_digest(v1)
    if v1["freeze_manifest_sha256"] != recovery["v1_manifest_canonical_sha256"]:
        raise AssertionError("v1 canonical fingerprint mismatch")
    if not method_consistency(v1, freeze)["passed"]:
        raise AssertionError("registered experimental method changed")


def write_exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class AttemptRecorder:
    """Each event and final record is created once; no prior evidence is replaced."""
    def __init__(self, root: Path, attempt_id: str, metadata: dict[str, Any]):
        if re.fullmatch(r"attempt_\d{3}", attempt_id) is None:
            raise ValueError("invalid attempt identifier")
        self.path = root / attempt_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.attempt_id = attempt_id
        self.stage = "started"
        self.events: list[dict[str, Any]] = []
        self.metadata = metadata
        self.event("started", **metadata)

    def event(self, stage: str, **data: Any) -> None:
        self.stage = stage
        value = {"attempt_id": self.attempt_id, "stage": stage, "recorded_utc": utc_now(), **data}
        write_exclusive_json(self.path / f"{len(self.events):03d}_{stage}.json", value)
        self.events.append(value)

    def finish(self, status: str, **data: Any) -> None:
        write_exclusive_json(self.path / "result.json", {
            "attempt_id": self.attempt_id, "status": status, "last_stage": self.stage,
            "recorded_utc": utc_now(), "dev_read_started": any(e["stage"] == "dev_read_started" for e in self.events),
            **self.metadata, **data,
        })
