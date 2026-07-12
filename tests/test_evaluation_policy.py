import pandas as pd

from label_focused.datasets import DATASET_REGISTRY, prepare_frame
from label_focused.evaluation import (
    correct_full_test_predictions,
    majority_labels,
)
from label_focused.prompting import parse_prediction, parse_single_prediction


def test_joint_parser_preserves_valid_task_when_other_key_is_missing():
    assert parse_prediction('{"Sentiment": "Positive"}') == (
        "Positive",
        "PARSE_ERROR",
    )


def test_dual_parser_returns_error_for_malformed_json():
    assert parse_single_prediction("not json", "topic") == "PARSE_ERROR"


def test_invalid_predictions_are_corrected_without_dropping_rows():
    spec = DATASET_REGISTRY["neu_esc"]
    frame = pd.DataFrame(
        {
            "pred_sentiment": ["Positive", "PARSE_ERROR", ""],
            "pred_topic": ["PARSE_ERROR", "Academic", "Unknown"],
        }
    )
    corrected = correct_full_test_predictions(frame, spec, "Neutral", "Academic")
    assert len(corrected) == 3
    assert corrected["pred_sentiment_corrected"].tolist() == [
        "Positive", "Neutral", "Neutral"
    ]
    assert corrected["pred_topic_corrected"].tolist() == [
        "Academic", "Academic", "Academic"
    ]


def test_majority_is_calculated_from_training_split():
    spec = DATASET_REGISTRY["uit_vsfc"]
    train = pd.DataFrame(
        {
            "sentence": ["a", "b", "c"],
            "sentiment": [2, 2, 0],
            "topic": [0, 0, 3],
        }
    )
    assert majority_labels(train, spec) == ("Positive", "Lecturer")


def test_victsd_separator_preprocessing():
    spec = DATASET_REGISTRY["victsd"]
    frame = pd.DataFrame(
        [{"Title": "Tiêu đề", "Comment": "Bình luận", "Toxicity": 0, "Topic": "TheGioi"}]
    )
    assert prepare_frame(frame, spec).iloc[0]["Text"] == "Tiêu đề [SEP] Bình luận"
