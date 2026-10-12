"""Run recorded pre-Dev math/core/helper tests without evaluating official Dev."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE3B = ROOT / "stage03b_official_validation"
SUITES = [
    ("stage2_math", ROOT / "stage02_math_protocol", ["-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]),
    ("stage3a_implementation", ROOT / "stage03a_internal_experiment", ["-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]),
    ("stage3b_synthetic_helpers", STAGE3B, [str(STAGE3B / "test_official_dev_runner.py")]),
]


def redact_paths(text: str) -> str:
    return re.sub(re.escape(str(ROOT)), "<PROJECT_ROOT>", text, flags=re.IGNORECASE)


def main() -> int:
    results = []
    log_parts = []
    all_passed = True
    for name, cwd, args in SUITES:
        command = [sys.executable, *args]
        proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
        output = redact_paths((proc.stdout or "") + (proc.stderr or ""))
        match = re.search(r"Ran (\d+) tests? in", output)
        count = int(match.group(1)) if match else 0
        passed = proc.returncode == 0 and count > 0 and "FAILED" not in output
        all_passed = all_passed and passed
        results.append({
            "suite": name,
            "test_count": count,
            "exit_code": proc.returncode,
            "passed": passed,
            "official_dev_opened": False,
        })
        display_args = [str(Path(arg).relative_to(ROOT)) if Path(arg).is_absolute() else arg for arg in args]
        log_parts.append(f"===== {name} =====\ncommand: python {' '.join(display_args)}\n{output.rstrip()}\n")
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pre-Dev synthetic math/core/helper tests; Stage 3A data loader test is restricted to the Train member",
        "suites": results,
        "total_tests": sum(item["test_count"] for item in results),
        "all_passed": all_passed,
        "official_dev_raw_text_opened": False,
        "threshold_or_feature_selection_performed": False,
    }
    (STAGE3B / "pre_dev_test_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGE3B / "pre_dev_test_log.txt").write_text("\n".join(log_parts), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
