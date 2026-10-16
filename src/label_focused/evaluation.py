"""Full-test metrics: invalid predictions remain errors, without imputation."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


INVALID_LABEL = "PARSE_ERROR"


def _metric_inputs(truth, predictions, labels):
    truth, predictions, labels = list(truth), list(predictions), list(labels)
    if not truth or len(truth) != len(predictions):
        raise ValueError("Truth and predictions must have the same non-zero length")
    if (
        not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
        or INVALID_LABEL in labels
    ):
        raise ValueError("Business labels must be unique non-empty strings, excluding PARSE_ERROR")
    allowed = set(labels)
    if any(not isinstance(value, str) or value not in allowed for value in truth):
        raise ValueError("Ground-truth labels must belong to the business label set")
    if any(
        not isinstance(value, str) or value not in allowed | {INVALID_LABEL}
        for value in predictions
    ):
        raise ValueError("Predictions must be business labels or PARSE_ERROR")
    return truth, predictions, labels


def classification_metrics(
    truth: Iterable[str], predictions: Iterable[str], labels: list[str] | tuple[str, ...]
) -> dict:
    truth, predictions, labels = _metric_inputs(truth, predictions, labels)
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


def prepare_full_test_predictions(frame: pd.DataFrame, spec: Any) -> pd.DataFrame:
    """Normalize each task independently and preserve every prediction row.

    Empty, missing, or out-of-set predictions become ``PARSE_ERROR`` and count
    as wrong. Existing case-insensitive canonicalization is retained. Original
    values and raw-output columns remain available for auditing; when original
    columns already exist, they take precedence over previously corrected data.
    The ``*_corrected`` columns are compatibility aliases for the normalized
    predictions and no longer mean majority-label replacement.
    """
    required = {"pred_sentiment", "pred_topic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")

    result = frame.copy()
    for task, labels in (
        ("sentiment", spec.sentiment_labels), ("topic", spec.topic_labels)
    ):
        original = f"pred_{task}_original"
        if original not in result:
            result[original] = result[f"pred_{task}"].copy()
        lookup = {label.casefold(): label for label in labels}
        values = result[original].fillna("").astype(str).str.strip()
        canonical = values.str.casefold().map(lookup)
        result[f"{task}_invalid"] = canonical.isna()
        normalized = canonical.fillna(INVALID_LABEL)
        result[f"pred_{task}"] = normalized
        result[f"pred_{task}_corrected"] = normalized
    result["any_invalid"] = result["sentiment_invalid"] | result["topic_invalid"]
    if len(result) != len(frame):
        raise AssertionError("Prediction normalization changed the number of test rows")
    return result


def evaluate_full_test_predictions(frame: pd.DataFrame, spec: Any) -> dict[str, Any]:
    """Score all rows, keeping PARSE_ERROR outside the business-class macro mean."""
    required = {"true_sentiment", "true_topic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    frame = prepare_full_test_predictions(frame, spec)
    return {
        "test_size": len(frame),
        "invalid_counts": {
            "sentiment": int(frame["sentiment_invalid"].sum()),
            "topic": int(frame["topic_invalid"].sum()),
            "rows_with_any_invalid": int(frame["any_invalid"].sum()),
        },
        "sentiment": classification_metrics(
            frame["true_sentiment"],
            frame["pred_sentiment"],
            spec.sentiment_labels,
        ),
        "topic": classification_metrics(
            frame["true_topic"],
            frame["pred_topic"],
            spec.topic_labels,
        ),
    }


def confusion_matrix_full_test(
    truth: Iterable[str], predictions: Iterable[str], labels: list[str] | tuple[str, ...]
):
    """Return a confusion matrix and axes that account for every test sample.

    Business classes retain their configured order. A PARSE_ERROR prediction
    column (and matching empty truth row) is included only when needed, so
    plotting with a fixed business label list cannot silently drop failures.
    """
    truth, predictions, labels = _metric_inputs(truth, predictions, labels)
    display_labels = labels + ([INVALID_LABEL] if INVALID_LABEL in predictions else [])
    matrix = confusion_matrix(truth, predictions, labels=display_labels)
    if int(matrix.sum()) != len(truth):
        raise AssertionError("Confusion matrix omitted test samples")
    return matrix, display_labels
