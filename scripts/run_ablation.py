#!/usr/bin/env python3
"""Run the exact Base/Masking/Full ablation configurations from Table 6."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANKS = {"neu_esc": 32, "uit_vsfc": 8, "victsd": 8}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(RANKS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rank = RANKS[args.dataset]
    for method in ("base", "masking", "full"):
        config = ROOT / "configs" / "ablation" / f"{args.dataset}_{method}_r{rank}.yaml"
        command = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--config", str(config), "--data-dir", args.data_dir,
            "--output-root", args.output_root,
        ]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
