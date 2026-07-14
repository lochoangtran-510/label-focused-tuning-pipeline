"""Full-test correction and reproducible classification metrics."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(
    truth: Iterable[str], predictions: Iterable[str], labels: list[str] | tuple[str, ...]
) -> dict:
    truth, predictions = list(truth), list(predictions)
    return {
        "n_samples": len(truth),
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            truth, predictions, labels=labels, average="weighted", zero_division=0
        ),
        "classification_report": classification_report(
            truth, predictions, labels=labels, output_dict=True, zero_division=0
        ),
    }


def majority_labels(train_frame: pd.DataFrame, spec: Any) -> tuple[str, str]:
    """Calculate fallbacks strictly from the training split."""
    from .prompting import labels_from_row

    mapped = [labels_from_row(row, spec) for _, row in train_frame.iterrows()]
    if not mapped:
        raise ValueError("Cannot determine majority labels from an empty training split")
    sentiments, topics = zip(*mapped)
    return (
        pd.Series(sentiments).mode(dropna=True).iloc[0],
        pd.Series(topics).mode(dropna=True).iloc[0],
    )


def correct_full_test_predictions(
    frame: pd.DataFrame,
    spec: Any,
    majority_sentiment: str,
    majority_topic: str,
) -> pd.DataFrame:
    """Replace invalid labels independently; never remove test rows."""
    required = {"pred_sentiment", "pred_topic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")

    result = frame.copy()
    result["pred_sentiment_original"] = (
        result["pred_sentiment"].fillna("").astype(str).str.strip()
    )
    result["pred_topic_original"] = (
        result["pred_topic"].fillna("").astype(str).str.strip()
    )
    sentiment_lookup = {label.casefold(): label for label in spec.sentiment_labels}
    topic_lookup = {label.casefold(): label for label in spec.topic_labels}
    canonical_sentiment = result["pred_sentiment_original"].str.casefold().map(
        sentiment_lookup
    )
    canonical_topic = result["pred_topic_original"].str.casefold().map(topic_lookup)
    result["sentiment_invalid"] = canonical_sentiment.isna()
    result["topic_invalid"] = canonical_topic.isna()
    result["any_invalid"] = result["sentiment_invalid"] | result["topic_invalid"]
    result["pred_sentiment_corrected"] = canonical_sentiment.fillna(
        majority_sentiment
    )
    result["pred_topic_corrected"] = canonical_topic.fillna(majority_topic)
    if len(result) != len(frame):
        raise AssertionError("Full-test correction changed the number of rows")
    return result


def evaluate_corrected_predictions(frame: pd.DataFrame, spec: Any) -> dict[str, Any]:
    required = {
        "true_sentiment", "true_topic",
        "pred_sentiment_corrected", "pred_topic_corrected",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Corrected frame is missing columns: {sorted(missing)}")
    return {
        "test_size": len(frame),
        "invalid_counts": {
            "sentiment": int(frame["sentiment_invalid"].sum()),
            "topic": int(frame["topic_invalid"].sum()),
            "rows_with_any_invalid": int(frame["any_invalid"].sum()),
        },
        "sentiment": classification_metrics(
            frame["true_sentiment"],
            frame["pred_sentiment_corrected"],
            spec.sentiment_labels,
        ),
        "topic": classification_metrics(
            frame["true_topic"],
            frame["pred_topic_corrected"],
            spec.topic_labels,
        ),
    }
