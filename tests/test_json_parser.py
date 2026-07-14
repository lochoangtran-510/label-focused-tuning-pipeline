from label_focused.datasets import DATASET_REGISTRY
from label_focused.prompting import (
    build_zero_shot_prompt,
    parse_prediction,
    parse_single_prediction,
)


def test_joint_parser_fails_each_missing_key_independently():
    assert parse_prediction('{"Sentiment":"Neutral"}') == (
        "Neutral", "PARSE_ERROR"
    )


def test_dual_parser_rejects_invalid_json():
    assert parse_single_prediction("invalid", "sentiment") == "PARSE_ERROR"


def test_zero_shot_prompt_has_no_one_shot_example():
    class Tokenizer:
        def apply_chat_template(self, messages, **_):
            return messages

    spec = DATASET_REGISTRY["neu_esc"]
    messages = build_zero_shot_prompt("new input", spec, Tokenizer())
    assert len(messages) == 2
    assert spec.one_shot_text not in messages[1]["content"]
    assert '"Sentiment": "Nhãn"' in messages[0]["content"]
    assert messages[1]["content"].endswith("Trích xuất JSON:")


def test_victsd_zero_shot_uses_notebook_toxicity_contract():
    class Tokenizer:
        def apply_chat_template(self, messages, **_):
            return messages

    spec = DATASET_REGISTRY["victsd"]
    messages = build_zero_shot_prompt("new input", spec, Tokenizer())
    assert '"Toxicity": "Nhãn"' in messages[0]["content"]
    assert "Sentiment và Topic" not in messages[0]["content"]


def test_victsd_parser_reads_toxicity_key():
    assert parse_prediction(
        '{"Toxicity": "Non-toxic", "Topic": "TheGioi"}',
        sentiment_key="Toxicity",
    ) == ("Non-toxic", "TheGioi")
