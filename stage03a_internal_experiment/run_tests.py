from __future__ import annotations

import json
import platform
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy
import PIL
import scipy
import sklearn

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(EXPERIMENT_DIR / "tests"), pattern="test_*.py", top_level_dir=str(EXPERIMENT_DIR))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic Stage 3A implementation tests only; no official Dev raw text parsed or scored",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pillow": PIL.__version__,
        },
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "was_successful": result.wasSuccessful(),
        "failed_tests": [test.id() for test, _ in result.failures + result.errors],
    }
    outputs = EXPERIMENT_DIR / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "stage3a_test_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
