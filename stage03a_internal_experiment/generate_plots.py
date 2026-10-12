from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from visualize import generate_all_figures  # noqa: E402
from pipeline import write_json, write_output_checksums  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate only Stage 3A internal-tune plots from saved CSV files.")
    parser.add_argument("--root", type=Path, default=EXPERIMENT_DIR.parent, help="project root (not used to read raw data)")
    parser.parse_args()
    result = generate_all_figures(EXPERIMENT_DIR)
    write_json(EXPERIMENT_DIR / "outputs" / "plot_metadata.json", result)
    write_output_checksums(EXPERIMENT_DIR)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
