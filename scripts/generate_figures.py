#!/usr/bin/env python3
"""Generate confusion matrices and error-distribution figures from predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from label_focused.datasets import DATASET_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--prefix", default="model")
    args = parser.parse_args()

    spec = DATASET_REGISTRY[args.dataset]
    frame = pd.read_csv(args.predictions)
    required = {
        "true_sentiment", "true_topic",
        "pred_sentiment_corrected", "pred_topic_corrected",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing: {sorted(missing)}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    for axis, task, labels in (
        (axes[0], "sentiment", spec.sentiment_labels),
        (axes[1], "topic", spec.topic_labels),
    ):
        matrix = confusion_matrix(
            frame[f"true_{task}"], frame[f"pred_{task}_corrected"], labels=labels
        )
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
            ax=axis, colorbar=False, cmap="Blues", values_format="d"
        )
        axis.set_title(task.title())
        axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    confusion_path = output / f"{args.prefix}_confusion_matrices.pdf"
    fig.savefig(confusion_path, bbox_inches="tight")
    plt.close(fig)

    sent_ok = frame["true_sentiment"] == frame["pred_sentiment_corrected"]
    topic_ok = frame["true_topic"] == frame["pred_topic_corrected"]
    categories = pd.Series("Both correct", index=frame.index)
    categories.loc[~sent_ok & topic_ok] = "Sentiment only error"
    categories.loc[sent_ok & ~topic_ok] = "Topic only error"
    categories.loc[~sent_ok & ~topic_ok] = "Both tasks wrong"
    counts = categories.value_counts().reindex(
        ["Both correct", "Sentiment only error", "Topic only error", "Both tasks wrong"],
        fill_value=0,
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    counts.plot.bar(ax=axis, color=["#4c956c", "#f4a261", "#e9c46a", "#e76f51"])
    axis.set_ylabel("Samples")
    axis.set_title(f"Error distribution — {spec.name}")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    error_path = output / f"{args.prefix}_error_analysis.pdf"
    figure.savefig(error_path, bbox_inches="tight")
    plt.close(figure)
    counts.rename("samples").to_csv(output / f"{args.prefix}_error_analysis.csv")
    print(confusion_path)
    print(error_path)


if __name__ == "__main__":
    main()
