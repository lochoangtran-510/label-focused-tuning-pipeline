#!/usr/bin/env python3
"""Train a reported Joint or Dual Adapter experiment from YAML."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from label_focused.datasets import DATASET_REGISTRY, load_splits
from label_focused.training import (
    ExperimentConfig,
    load_model_and_tokenizer,
    make_prompt_datasets,
    make_trainer,
    train_dual_adapters,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-steps", type=int, help="Override for smoke tests, e.g. 2")
    parser.add_argument("--output-root", help="Override output root")
    return parser.parse_args()


def read_config(path: Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment YAML must contain a mapping")
    experiment = raw.get("experiment", raw)
    allowed = {field.name for field in fields(ExperimentConfig)}
    unknown = set(experiment).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown ExperimentConfig fields: {sorted(unknown)}")
    config = ExperimentConfig(**experiment)
    validate_config(config)
    return config, raw


def validate_config(config: ExperimentConfig) -> None:
    if config.dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    if config.architecture not in {"joint", "dual_adapter"}:
        raise ValueError(f"Unsupported architecture: {config.architecture}")
    if config.epochs != 3:
        raise ValueError("Reported experiments use exactly 3 epochs")
    if config.lora_alpha not in (None, 2 * config.lora_rank):
        raise ValueError("The paper uses LoRA alpha = 2 * rank")
    if config.max_steps != -1:
        raise ValueError(
            "Public reproduction presets train for 3 complete epochs; "
            "set max_steps=-1. Use the CLI --max-steps option only for a smoke test."
        )


def environment_metadata() -> dict[str, Any]:
    import torch
    import transformers
    import peft
    import trl

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=False,
        ).stdout.strip() or None
    except OSError:
        commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = arguments()
    config_path = Path(args.config)
    config, source_yaml = read_config(config_path)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps smoke-test override must be positive")
        config.max_steps = args.max_steps
    if args.output_root:
        config.output_root = args.output_root

    run_root = Path(config.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "resolved_config.json", asdict(config))
    write_json(run_root / "source_config.json", source_yaml)
    write_json(run_root / "environment.json", environment_metadata())

    splits = load_splits(config.dataset, args.data_dir)
    started = time.perf_counter()
    if config.architecture == "dual_adapter":
        adapter_paths = train_dual_adapters(splits, config)
        summary = {"adapter_paths": adapter_paths}
    else:
        spec = DATASET_REGISTRY[config.dataset]
        model, tokenizer = load_model_and_tokenizer(config)
        train_dataset, validation_dataset = make_prompt_datasets(
            splits["train"], splits["validation"], spec, tokenizer, config.max_length
        )
        trainer = make_trainer(
            model, tokenizer, train_dataset, validation_dataset,
            splits["train"], config,
        )
        result = trainer.train()
        best_dir = run_root / "best_model"
        trainer.model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        write_json(run_root / "training_metrics.json", result.metrics)
        write_json(run_root / "trainer_log_history.json", trainer.state.log_history)
        summary = {"adapter_paths": {"joint": str(best_dir)}}

    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["resolved_config"] = asdict(config)
    write_json(run_root / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
