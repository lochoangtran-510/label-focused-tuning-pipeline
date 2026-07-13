#!/usr/bin/env python3
"""Create the train/validation/test files consumed by all experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from label_focused.preprocessing import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["neu_esc", "uit_vsfc", "victsd"])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    written = prepare_dataset(args.dataset, args.raw_dir, args.output_dir)
    for split, path in written.items():
        print(f"{split:>10}: {path}")


if __name__ == "__main__":
    main()
