#!/usr/bin/env python3
"""Run Joint or Dual Adapter vLLM inference on a complete test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from label_focused.datasets import load_splits
from label_focused.inference import InferenceConfig, run_inference


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["neu_esc", "uit_vsfc", "victsd"])
    parser.add_argument("--architecture", required=True, choices=["joint", "dual_adapter"])
    parser.add_argument("--model-id", default="Viet-Mistral/Vistral-7B-Chat")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--joint-adapter")
    parser.add_argument("--sentiment-adapter")
    parser.add_argument("--topic-adapter")
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--max-model-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.dataset, args.data_dir)
    config = InferenceConfig(
        dataset=args.dataset,
        architecture=args.architecture,
        model_id=args.model_id,
        joint_adapter=args.joint_adapter,
        sentiment_adapter=args.sentiment_adapter,
        topic_adapter=args.topic_adapter,
        output_dir=str(output_dir),
        max_lora_rank=args.max_lora_rank,
        max_model_length=args.max_model_length,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    predictions, performance = run_inference(splits["test"], splits["train"], config)
    prediction_path = output_dir / "predictions.csv"
    performance_path = output_dir / "performance.json"
    invalid_path = output_dir / "invalid_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    predictions.loc[predictions["any_invalid"]].to_csv(
        invalid_path, index=False, encoding="utf-8-sig"
    )
    performance_path.write_text(
        json.dumps(performance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Predictions : {prediction_path}")
    print(f"Invalid audit: {invalid_path}")
    print(f"Performance : {performance_path}")
    print(f"Full test rows retained: {len(predictions)}")


if __name__ == "__main__":
    main()
