#!/usr/bin/env python3
"""Run only the Joint or Dual rank combinations reported in Tables 8–9."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configs(architecture: str, dataset: str) -> list[Path]:
    if architecture == "joint":
        if dataset not in {"neu_esc", "uit_vsfc"}:
            raise ValueError("The paper reports no Joint rank sweep for ViCTSD")
        return [
            ROOT / "configs" / "joint_rank" / f"{dataset}_{method}_r{rank}.yaml"
            for rank in (8, 16, 32, 64)
            for method in ("base", "full")
        ]
    if dataset != "neu_esc":
        raise ValueError("The paper reports the Dual rank sweep only for NEU-ESC")
    pairs = ((8, 8), (16, 16), (32, 32), (64, 64), (16, 64), (64, 16), (32, 8))
    return [
        ROOT / "configs" / "dual_rank" / f"neu_esc_full_r{s}_r{t}.yaml"
        for s, t in pairs
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=["joint", "dual_adapter"])
    parser.add_argument("--dataset", required=True, choices=["neu_esc", "uit_vsfc", "victsd"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for config in configs(args.architecture, args.dataset):
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
