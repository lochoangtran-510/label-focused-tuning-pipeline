#!/usr/bin/env python3
"""Evaluate every test row; invalid predictions count as errors without imputation."""

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
    prepare_full_test_predictions,
    evaluate_full_test_predictions,
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
    frame = pd.read_csv(prediction_path, keep_default_na=False)
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
    for column, truth in [("true_sentiment", expected_sentiment), ("true_topic", expected_topic)]:
        if column not in frame:
            frame[column] = truth
        elif frame[column].astype(str).tolist() != truth:
            raise ValueError("Ground-truth labels or row order do not match the selected test split")
    if "text" in frame and frame["text"].astype(str).tolist() != splits["test"][spec.text_column].astype(str).tolist():
        raise ValueError("Prediction text or row order does not match the selected test split")
    # Preparation uses original raw prediction columns when present, including
    # files produced by the former imputation policy. Never score old replacements.
    corrected = prepare_full_test_predictions(frame, spec)
    metrics = evaluate_full_test_predictions(corrected, spec)
    metrics["evaluation_policy"] = {
        "fallback_enabled": False,
        "fallback": "disabled",
        "invalid_sentinel": "PARSE_ERROR",
        "invalid_predictions": "count_as_errors",
        "test_rows_removed": 0,
    }
    # Historical filename retained for consumers; values are normalization-only,
    # with invalid labels represented by PARSE_ERROR and no label imputation.

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
