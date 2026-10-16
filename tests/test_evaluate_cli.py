"""CLI regression: rescore raw originals, never previously imputed labels."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from label_focused.datasets import DATASET_REGISTRY
from label_focused.prompting import labels_from_row


def cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluation_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frames(dataset):
    spec = DATASET_REGISTRY[dataset]
    official = pd.DataFrame({
        spec.text_column: ["first test row", "second test row"],
        spec.sentiment_column: [0, 1],
        spec.topic_column: list(spec.topic_labels[:2]) if dataset == "victsd" else [0, 1],
    })
    labels = [labels_from_row(row, spec) for _, row in official.iterrows()]
    predictions = pd.DataFrame({
        "text": official[spec.text_column],
        "true_sentiment": [pair[0] for pair in labels],
        "true_topic": [pair[1] for pair in labels],
        # Simulate a historical artifact whose normalized/corrected columns were
        # already replaced with correct-looking majority labels.
        "pred_sentiment": [pair[0] for pair in labels],
        "pred_topic": [pair[1] for pair in labels],
        "pred_sentiment_original": ["outside_set", spec.sentiment_labels[1].lower()],
        "pred_topic_original": [spec.topic_labels[0].lower(), "PARSE_ERROR"],
        "pred_sentiment_corrected": [pair[0] for pair in labels],
        "pred_topic_corrected": [pair[1] for pair in labels],
    })
    return official, predictions


@pytest.mark.parametrize("dataset", ["neu_esc", "uit_vsfc", "victsd"])
def test_cli_scores_originals_and_keeps_all_rows(monkeypatch, tmp_path, dataset):
    cli = cli_module()
    official, frame = frames(dataset)
    source = tmp_path / "predictions.csv"; frame.to_csv(source, index=False)
    output = tmp_path / "metrics"
    monkeypatch.setattr(cli, "arguments", lambda: SimpleNamespace(dataset=dataset, predictions=str(source), output_dir=str(output), data_dir="unused"))
    # No training split is provided: evaluation must have no dependence on it.
    monkeypatch.setattr(cli, "load_splits", lambda *args: {"test": official})
    cli.main()
    prepared = pd.read_csv(output / "predictions_corrected.csv")
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(prepared) == len(official) == metrics["test_size"] == 2
    assert prepared["pred_sentiment"].tolist() == ["PARSE_ERROR", DATASET_REGISTRY[dataset].sentiment_labels[1]]
    assert prepared["pred_topic"].tolist() == [DATASET_REGISTRY[dataset].topic_labels[0], "PARSE_ERROR"]
    assert prepared["pred_sentiment_corrected"].tolist() == prepared["pred_sentiment"].tolist()
    assert prepared["pred_topic_corrected"].tolist() == prepared["pred_topic"].tolist()
    assert metrics["sentiment"]["accuracy"] == metrics["topic"]["accuracy"] == 0.5
    assert len(pd.read_csv(output / "invalid_predictions.csv")) == 2
    assert "fallback_policy" not in metrics and "majority_fallback" not in metrics
    assert metrics["evaluation_policy"]["fallback_enabled"] is False
    assert metrics["evaluation_policy"]["invalid_sentinel"] == "PARSE_ERROR"


@pytest.mark.parametrize("violation", ["partial", "truth_order", "text_order", "one_truth_column_wrong"])
def test_cli_rejects_partial_or_misaligned_predictions(monkeypatch, tmp_path, violation):
    cli = cli_module(); official, frame = frames("uit_vsfc")
    if violation == "partial": frame = frame.iloc[:1]
    elif violation == "truth_order": frame["true_sentiment"] = frame["true_sentiment"].iloc[::-1].to_numpy()
    elif violation == "text_order": frame["text"] = frame["text"].iloc[::-1].to_numpy()
    else:
        frame = frame.drop(columns="true_topic")
        frame["true_sentiment"] = "wrong"
    source = tmp_path / "predictions.csv"; frame.to_csv(source, index=False)
    monkeypatch.setattr(cli, "arguments", lambda: SimpleNamespace(dataset="uit_vsfc", predictions=str(source), output_dir=str(tmp_path / "out"), data_dir="unused"))
    monkeypatch.setattr(cli, "load_splits", lambda *args: {"test": official})
    with pytest.raises(ValueError, match="complete test split|row order"):
        cli.main()
