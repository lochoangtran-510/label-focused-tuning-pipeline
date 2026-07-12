#!/usr/bin/env python3
"""Evaluate all test rows using majority fallback for invalid predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from label_focused.datasets import DATASET_REGISTRY, load_splits
from label_focused.evaluation import (
    correct_full_test_predictions,
    evaluate_corrected_predictions,
    majority_labels,
)
from label_focused.prompting import labels_from_row


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["neu_esc", "uit_vsfc", "victsd"])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    args = arguments()
    prediction_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else prediction_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(prediction_path)
    splits = load_splits(args.dataset, args.data_dir)
    spec = DATASET_REGISTRY[args.dataset]
    if len(frame) != len(splits["test"]):
        raise ValueError(
            f"Prediction rows ({len(frame)}) do not match the complete test split "
            f"({len(splits['test'])}); refusing partial-test evaluation."
        )
    expected = [labels_from_row(row, spec) for _, row in splits["test"].iterrows()]
    expected_sentiment = [pair[0] for pair in expected]
    expected_topic = [pair[1] for pair in expected]
    if "true_sentiment" not in frame or "true_topic" not in frame:
        frame["true_sentiment"] = expected_sentiment
        frame["true_topic"] = expected_topic
    elif (
        frame["true_sentiment"].astype(str).tolist() != expected_sentiment
        or frame["true_topic"].astype(str).tolist() != expected_topic
    ):
        raise ValueError(
            "Ground-truth labels or row order do not match the selected test split"
        )
    majority_sentiment, majority_topic = majority_labels(splits["train"], spec)
    corrected = correct_full_test_predictions(
        frame, spec, majority_sentiment, majority_topic
    )
    metrics = evaluate_corrected_predictions(corrected, spec)
    metrics["fallback_policy"] = {
        "source": "training split",
        "sentiment": majority_sentiment,
        "topic": majority_topic,
        "applied_independently_per_task": True,
        "test_rows_removed": 0,
    }

    corrected.to_csv(
        output_dir / "predictions_corrected.csv", index=False, encoding="utf-8-sig"
    )
    corrected.loc[corrected["any_invalid"]].to_csv(
        output_dir / "invalid_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(metrics["sentiment"]["classification_report"]).T.to_csv(
        output_dir / "sentiment_classification_report.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(metrics["topic"]["classification_report"]).T.to_csv(
        output_dir / "topic_classification_report.csv", encoding="utf-8-sig"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(json_safe(metrics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for task in ("sentiment", "topic"):
        result = metrics[task]
        print(
            f"{task.title():10} n={result['n_samples']} "
            f"Acc={result['accuracy']:.4f} "
            f"Macro-F1={result['macro_f1']:.4f} "
            f"Weighted-F1={result['weighted_f1']:.4f}"
        )
    print(f"Invalid counts: {metrics['invalid_counts']}")
    print("Test rows removed: 0")


if __name__ == "__main__":
    main()
