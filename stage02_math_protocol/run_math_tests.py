"""Run only the small, synthetic Stage 2 math tests and record actual results."""

from __future__ import annotations

import json
import platform
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy
import scipy
import sklearn


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "hand-built vectors and tiny synthetic corpora only; no AFQMC text fitted or scored",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "was_successful": result.wasSuccessful(),
        "failed_tests": [test.id() for test, _ in result.failures + result.errors],
    }
    (ROOT / "math_test_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded actual test run to {ROOT / 'math_test_results.json'}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
