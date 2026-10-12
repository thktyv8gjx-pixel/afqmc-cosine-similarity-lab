"""Fail-closed allowlist and staged-content scan for the first public commit."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = ROOT / "scripts" / "public_allowlist.json"
FORBIDDEN_SUFFIXES = {".zip", ".pdf", ".docx", ".jsonl", ".env", ".pem", ".key"}
FORBIDDEN_NAME_PARTS = (
    "split_manifest", "_overlap.csv", "_predictions.csv", "error_cases", "error_examples",
    "fit_corpus", "feature_vocabulary", "manual_case", "data_issues.csv", "duplicate_pairs.csv",
    "duplicate_records.csv", "label_conflicts.csv",
)
SECRET_PATTERNS = {
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "google_api_key": re.compile(rb"\bAIza[0-9A-Za-z\-_]{35}\b"),
    "private_key_header": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "jwt": re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "credential_assignment": re.compile(rb"(?i)(?:api[_-]?key|client[_-]?secret|password|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
    "personal_windows_path": re.compile(rb"(?i)[A-Z]:\\Users\\[^\\\r\n]+"),
    "workspace_absolute_path": re.compile(re.escape("D:\\人工智能数学实验".encode("utf-8")), re.IGNORECASE),
}


def run_git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def main() -> int:
    spec = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    expected = set(spec["files"])
    staged = {part.decode("utf-8") for part in run_git("diff", "--cached", "--name-only", "-z").split(b"\0") if part}
    if not staged:
        raise SystemExit("FAIL: Git index is empty")
    if staged != expected:
        missing = sorted(expected - staged)
        extra = sorted(staged - expected)
        raise SystemExit(f"FAIL: staged allowlist mismatch; missing={missing}; extra={extra}")
    unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode
    if unstaged != 0:
        raise SystemExit("FAIL: tracked working-tree edits remain unstaged")
    whitespace = subprocess.run(["git", "diff", "--cached", "--check"], cwd=ROOT, capture_output=True, text=True)
    if whitespace.returncode != 0:
        raise SystemExit("FAIL: staged whitespace check failed")

    hits: list[str] = []
    for relative in sorted(staged):
        path = Path(relative)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            hits.append(f"forbidden suffix: {relative}")
        if any(part in path.name.lower() for part in FORBIDDEN_NAME_PARTS):
            hits.append(f"forbidden row-level filename: {relative}")
        payload = (ROOT / path).read_bytes()
        for category, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                hits.append(f"content pattern {category}: {relative}")
    if hits:
        raise SystemExit("FAIL: staged safety scan found: " + "; ".join(hits))

    commits = run_git("rev-list", "--all", "--count").decode("ascii").strip()
    if commits != "0":
        raise SystemExit("FAIL: initial public release must not include pre-existing local history")
    print(json.dumps({
        "status": "PASS",
        "staged_file_count": len(staged),
        "allowlist_exact_match": True,
        "secret_pattern_hits": 0,
        "forbidden_row_level_or_source_files": 0,
        "preexisting_commit_count": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"FAIL: Git command failed ({error.cmd[0]})", file=sys.stderr)
        raise SystemExit(2)
