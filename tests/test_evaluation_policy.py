import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from label_focused.datasets import DATASET_REGISTRY, prepare_frame
from label_focused.evaluation import (
    INVALID_LABEL,
    classification_metrics,
    confusion_matrix_full_test,
    evaluate_full_test_predictions,
    prepare_full_test_predictions,
)
from label_focused.prompting import parse_prediction, parse_single_prediction


def test_joint_parser_preserves_valid_task_when_other_key_is_missing():
    assert parse_prediction('{"Sentiment": "Positive"}') == (
        "Positive",
        "PARSE_ERROR",
    )


def test_dual_parser_returns_error_for_malformed_json():
    assert parse_single_prediction("not json", "topic") == "PARSE_ERROR"


@pytest.mark.parametrize("dataset", ["neu_esc", "uit_vsfc", "victsd"])
def test_invalid_predictions_count_as_wrong_without_dropping_rows(dataset):
    spec = DATASET_REGISTRY[dataset]
    sentiment, topic = spec.sentiment_labels[0], spec.topic_labels[0]
    frame = pd.DataFrame(
        {
            "true_sentiment": [sentiment] * 3,
            "true_topic": [topic] * 3,
            "pred_sentiment": [sentiment, None, "Unknown"],
            "pred_topic": ["PARSE_ERROR", topic, topic],
            "raw_output": ["partial result", "missing sentiment", "invalid label"],
        },
        index=[8, 11, 19],
    )
    original = frame.copy(deep=True)
    prepared = prepare_full_test_predictions(frame, spec)
    pd.testing.assert_frame_equal(frame, original)
    assert prepared.index.tolist() == [8, 11, 19]
    assert len(prepared) == 3
    assert prepared["pred_sentiment"].tolist() == [sentiment, INVALID_LABEL, INVALID_LABEL]
    assert prepared["pred_topic"].tolist() == [INVALID_LABEL, topic, topic]
    assert prepared["pred_sentiment_original"].tolist() == [sentiment, None, "Unknown"]
    assert prepared["raw_output"].tolist() == frame["raw_output"].tolist()
    for task in ("sentiment", "topic"):
        assert prepared[f"pred_{task}_corrected"].equals(prepared[f"pred_{task}"])
    metrics = evaluate_full_test_predictions(prepared, spec)
    assert metrics["test_size"] == 3
    assert metrics["sentiment"]["n_samples"] == metrics["topic"]["n_samples"] == 3
    assert metrics["sentiment"]["accuracy"] == pytest.approx(1 / 3)
    assert metrics["topic"]["accuracy"] == pytest.approx(2 / 3)
    assert metrics["invalid_counts"] == {
        "sentiment": 2, "topic": 1, "rows_with_any_invalid": 3,
    }
    pd.testing.assert_frame_equal(prepare_full_test_predictions(prepared, spec), prepared)


def test_generated_labels_are_canonicalized_case_insensitively():
    spec = DATASET_REGISTRY["uit_vsfc"]
    frame = pd.DataFrame(
        {"pred_sentiment": ["positive"], "pred_topic": ["LECTURER"]}
    )
    prepared = prepare_full_test_predictions(frame, spec)
    assert prepared.loc[0, "pred_sentiment_corrected"] == "Positive"
    assert prepared.loc[0, "pred_topic_corrected"] == "Lecturer"
    assert not prepared.loc[0, "any_invalid"]


@pytest.mark.parametrize("dataset,sentiment,topic", [
    ("neu_esc", "Neutral", "Other"),
    ("uit_vsfc", "Positive", "Lecturer"),
    ("victsd", "Non-toxic", "TheGioi"),
])
def test_invalid_result_cannot_get_credit_from_a_matching_majority_class(dataset, sentiment, topic):
    spec = DATASET_REGISTRY[dataset]
    frame = pd.DataFrame(
        {
            "true_sentiment": [sentiment, sentiment], "true_topic": [topic, topic],
            "pred_sentiment": [INVALID_LABEL, sentiment], "pred_topic": ["", topic],
        }
    )
    metrics = evaluate_full_test_predictions(frame, spec)
    # Imputing these labels would turn the failed row into a correct prediction.
    assert metrics["sentiment"]["accuracy"] == 0.5
    assert metrics["topic"]["accuracy"] == 0.5


@pytest.mark.parametrize("dataset", ["neu_esc", "uit_vsfc", "victsd"])
def test_all_valid_metrics_are_unchanged(dataset):
    spec = DATASET_REGISTRY[dataset]
    n = max(len(spec.sentiment_labels), len(spec.topic_labels))
    sentiments = [spec.sentiment_labels[index % len(spec.sentiment_labels)] for index in range(n)]
    topics = [spec.topic_labels[index % len(spec.topic_labels)] for index in range(n)]
    frame = pd.DataFrame({
        "true_sentiment": sentiments, "true_topic": topics,
        "pred_sentiment": sentiments, "pred_topic": topics,
    })
    metrics = evaluate_full_test_predictions(frame, spec)
    assert metrics["invalid_counts"] == {"sentiment": 0, "topic": 0, "rows_with_any_invalid": 0}
    for task, labels, values in [
        ("sentiment", spec.sentiment_labels, sentiments), ("topic", spec.topic_labels, topics)
    ]:
        assert metrics[task]["accuracy"] == accuracy_score(values, values)
        assert metrics[task]["macro_f1"] == f1_score(values, values, labels=labels, average="macro")
        assert metrics[task]["weighted_f1"] == 1


def test_re_evaluation_uses_original_outputs_instead_of_old_imputed_labels():
    spec = DATASET_REGISTRY["neu_esc"]
    frame = pd.DataFrame({
        "true_sentiment": ["Neutral"], "true_topic": ["Other"],
        "pred_sentiment": ["Neutral"], "pred_topic": ["Other"],
        "pred_sentiment_corrected": ["Neutral"], "pred_topic_corrected": ["Other"],
        "pred_sentiment_original": ["Unknown"], "pred_topic_original": [""],
    })
    prepared = prepare_full_test_predictions(frame, spec)
    assert prepared.loc[0, "pred_sentiment_corrected"] == INVALID_LABEL
    assert prepared.loc[0, "pred_topic_corrected"] == INVALID_LABEL
    assert prepared.loc[0, "pred_sentiment_original"] == "Unknown"
    metrics = evaluate_full_test_predictions(prepared, spec)
    assert metrics["sentiment"]["accuracy"] == metrics["topic"]["accuracy"] == 0


def test_macro_f1_uses_fixed_business_classes_and_all_invalid_rows_count():
    labels = ["A", "B"]
    metrics = classification_metrics(["A", "B"], ["A", INVALID_LABEL], labels)
    assert metrics["n_samples"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5  # PARSE_ERROR is not a third macro class.
    assert metrics["classification_report"]["B"]["support"] == 1
    assert metrics["classification_report"]["B"]["recall"] == 0
    all_invalid = classification_metrics(["A", "B"], [INVALID_LABEL] * 2, labels)
    assert all_invalid["accuracy"] == all_invalid["macro_f1"] == 0


def test_confusion_matrix_keeps_invalid_predictions_in_the_full_test_count():
    matrix, display_labels = confusion_matrix_full_test(
        ["A", "B", "A"], ["A", INVALID_LABEL, INVALID_LABEL], ["A", "B"]
    )
    assert display_labels == ["A", "B", INVALID_LABEL]
    assert matrix.sum() == 3
    assert matrix[:, -1].sum() == 2
    assert matrix[-1].sum() == 0
    assert matrix[0, 0] == matrix[0, 2] == matrix[1, 2] == 1
    valid_matrix, valid_labels = confusion_matrix_full_test(["A", "B"], ["A", "B"], ["A", "B"])
    assert valid_labels == ["A", "B"]
    assert valid_matrix.shape == (2, 2)
    assert valid_matrix.sum() == 2


@pytest.mark.parametrize("truth,predictions", [
    ([INVALID_LABEL], ["A"]), (["A"], ["Unknown"]), (["A"], []), ([], []),
])
def test_metric_and_confusion_matrix_reject_invalid_inputs(truth, predictions):
    for function in (classification_metrics, confusion_matrix_full_test):
        with pytest.raises(ValueError):
            function(truth, predictions, ["A", "B"])


def test_victsd_separator_preprocessing():
    spec = DATASET_REGISTRY["victsd"]
    frame = pd.DataFrame(
        [{"Title": "Tiêu đề", "Comment": "Bình luận", "Toxicity": 0, "Topic": "TheGioi"}]
    )
    assert prepare_frame(frame, spec).iloc[0]["Text"] == "Tiêu đề [SEP] Bình luận"
