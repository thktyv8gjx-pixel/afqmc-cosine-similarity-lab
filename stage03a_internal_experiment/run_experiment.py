from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
SRC = EXPERIMENT_DIR / "src"
sys.path.insert(0, str(SRC))

from pipeline import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run only the authorized AFQMC internal fit/tune experiment.")
    parser.add_argument("--root", type=Path, default=EXPERIMENT_DIR.parent, help="project root containing Stage 1 evidence")
    args = parser.parse_args()
    result = run_experiment(args.root.resolve(), EXPERIMENT_DIR)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote internal experiment outputs under: {EXPERIMENT_DIR / 'outputs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
