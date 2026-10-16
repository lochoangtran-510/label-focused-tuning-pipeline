import sys
from types import SimpleNamespace

import pytest
import torch

from label_focused.completion_collator import (
    TaskAwareAlphaCollator,
    make_full_sequence_collator,
    validate_completion_masking,
)


class PieceTokenizer:
    """Deterministic contextual pieces; no model download is needed."""

    eos_token = "</s>"
    all_special_tokens = ["</s>"]

    def __init__(self):
        self.pieces = {}
        self.ids = {}
        self.add(["ignored prompt", "[/INST]"])

    def add(self, pieces):
        result = []
        for piece in pieces:
            if piece not in self.ids:
                token_id = len(self.ids) + 10
                self.ids[piece] = token_id
                self.pieces[token_id] = piece
            result.append(self.ids[piece])
        return result

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(token_id)] for token_id in ids)

    def encode(self, text, **_kwargs):
        if text == "Others":
            return [999]  # isolated encoding deliberately differs from O + thers
        return [self.ids[text]]


class PreparedCollator:
    """Pad pre-masked examples to exercise preservation of the inner policy."""

    def __call__(self, features):
        width = max(len(feature["input_ids"]) for feature in features)
        return {
            key: torch.tensor([
                feature[key] + [padding] * (width - len(feature[key]))
                for feature in features
            ])
            for key, padding in [("input_ids", 0), ("attention_mask", 0), ("labels", -100)]
        }


def feature(tokenizer, pieces, *, full_sequence=False):
    prompt = tokenizer.add(["ignored prompt", "[/INST]"])
    response = tokenizer.add(pieces)
    ids = prompt + response
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": ids.copy() if full_sequence else [-100] * len(prompt) + response,
    }


@pytest.mark.parametrize("scope", ["terminal_only", "all_label_tokens"])
def test_contextual_spans_keep_masks_and_weight_only_label_content(scope):
    tokenizer = PieceTokenizer()
    example = feature(tokenizer, [
        '{"Sentiment": "', "Ne", "ut", "ral", '", "Topic": "', "O", "thers", '"}',
    ])
    inner = PreparedCollator()
    collator = TaskAwareAlphaCollator(
        inner, tokenizer, {"Sentiment": {"Neutral": 3}, "Topic": {"Others": 5}},
        alpha_scope=scope,
    )
    batch = collator([example])
    original = inner([example])
    for key in ("input_ids", "labels", "attention_mask"):
        torch.testing.assert_close(batch[key], original[key])
    expected = torch.ones_like(batch["alpha_weights"])
    sentiment = [3, 4, 5] if scope == "all_label_tokens" else [5]
    topic = [7, 8] if scope == "all_label_tokens" else [8]
    expected[0, sentiment], expected[0, topic] = 3, 5
    torch.testing.assert_close(batch["alpha_weights"], expected)
    assert 999 not in example["input_ids"]


def test_shared_label_and_token_id_can_have_different_field_weights():
    tokenizer = PieceTokenizer()
    example = feature(tokenizer, ['{"Sentiment": "', "Shared", '", "Topic": "', "Shared", '"}'])
    collator = TaskAwareAlphaCollator(
        PreparedCollator(), tokenizer,
        {"Sentiment": {"Shared": 2}, "Topic": {"Shared": 7}},
    )
    batch = collator([example])
    assert batch["input_ids"][0, 3] == batch["input_ids"][0, 5]
    assert batch["alpha_weights"][0, 3] == 2
    assert batch["alpha_weights"][0, 5] == 7


@pytest.mark.parametrize("field", ["Sentiment", "Topic"])
def test_dual_adapter_one_field_json_is_supported(field):
    tokenizer = PieceTokenizer()
    example = feature(tokenizer, [f'{{"{field}": "', "Known", '"}', "</s>"])
    collator = TaskAwareAlphaCollator(PreparedCollator(), tokenizer, {field: {"Known": 4}})
    batch = collator([example])
    assert batch["alpha_weights"][0, 3] == 4
    assert batch["alpha_weights"][0, 4:].eq(1).all()


def test_full_sequence_alpha_only_weights_response_and_uses_last_marker():
    tokenizer = PieceTokenizer()
    example = feature(tokenizer, ['{"Topic": "', "Known", '"}'], full_sequence=True)
    # Include an earlier example response in the unmasked prompt.
    earlier = tokenizer.add(["[/INST]", '{"Topic": "', "Known", '"}'])
    example = {
        key: earlier + values if key != "attention_mask" else [1] * len(earlier) + values
        for key, values in example.items()
    }
    collator = TaskAwareAlphaCollator(PreparedCollator(), tokenizer, {"Topic": {"Known": 6}})
    batch = collator([example])
    torch.testing.assert_close(batch["labels"][0], torch.tensor(example["input_ids"]))
    expected = torch.ones_like(batch["alpha_weights"])
    expected[0, -2] = 6
    torch.testing.assert_close(batch["alpha_weights"], expected)


def test_padding_does_not_change_per_example_alpha():
    tokenizer = PieceTokenizer()
    short = feature(tokenizer, ['{"Topic": "', "Known", '"}'])
    long = feature(tokenizer, ['{"Topic": "', "Long", " label", '"}'])
    collator = TaskAwareAlphaCollator(
        PreparedCollator(), tokenizer, {"Topic": {"Known": 2, "Long label": 3}},
        alpha_scope="all_label_tokens",
    )
    mixed = collator([short, long])
    for row, example in enumerate([short, long]):
        single = collator([example])
        length = len(example["input_ids"])
        torch.testing.assert_close(mixed["alpha_weights"][row, :length], single["alpha_weights"][0])
        assert mixed["labels"][row, length:].eq(-100).all()
        assert mixed["alpha_weights"][row, length:].eq(1).all()


@pytest.mark.parametrize("text", [
    '{"Topic": "Unknown"}',
    '{"Topic": "Known"',
    '{"Topic": "Known", "Extra": "Known"}',
    '{"Topic": "Known", "Topic": "Known"}',
    '{"Topic": 1}',
    '{"Topic": "Known"} trailing text',
])
def test_invalid_target_json_or_unknown_label_fails(text):
    tokenizer = PieceTokenizer()
    collator = TaskAwareAlphaCollator(PreparedCollator(), tokenizer, {"Topic": {"Known": 2}})
    with pytest.raises(ValueError):
        collator([feature(tokenizer, [text])])


def test_two_fields_in_one_token_are_rejected():
    tokenizer = PieceTokenizer()
    collator = TaskAwareAlphaCollator(
        PreparedCollator(), tokenizer, {"Sentiment": {"A": 2}, "Topic": {"B": 3}}
    )
    with pytest.raises(ValueError, match="share a token"):
        collator([feature(tokenizer, ['{"Sentiment": "A", "Topic": "B"}'])])


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_class_weights_fail_before_training(weight):
    with pytest.raises(ValueError, match="finite and positive"):
        TaskAwareAlphaCollator(PreparedCollator(), PieceTokenizer(), {"Topic": {"Known": weight}})


def test_unstable_prefix_decode_and_discontinuous_masks_fail():
    tokenizer = PieceTokenizer()
    example = feature(tokenizer, ['{"Topic": "', "Known", '"}'])
    collator = TaskAwareAlphaCollator(PreparedCollator(), tokenizer, {"Topic": {"Known": 2}})
    original_decode = tokenizer.decode
    tokenizer.decode = lambda ids, **kwargs: "!" if len(ids) == 1 else original_decode(ids, **kwargs)
    with pytest.raises(ValueError, match="prefix decode"):
        collator([example])
    tokenizer.decode = original_decode
    example["labels"][3] = -100
    with pytest.raises(ValueError, match="contiguous"):
        collator([example])


def test_full_sequence_ce_preserves_real_eos_when_it_is_also_padding(monkeypatch):
    class PaddingCollator:
        def __init__(self, tokenizer, **_kwargs):
            self.tokenizer = tokenizer

        def __call__(self, features):
            width = max(len(item["input_ids"]) for item in features)
            return {
                "input_ids": torch.tensor([
                    item["input_ids"] + [9] * (width - len(item["input_ids"])) for item in features
                ]),
                "attention_mask": torch.tensor([
                    item["attention_mask"] + [0] * (width - len(item["attention_mask"])) for item in features
                ]),
            }

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(DataCollatorWithPadding=PaddingCollator))
    collator = make_full_sequence_collator(SimpleNamespace(pad_token_id=9, eos_token_id=9))
    batch = collator([
        {"input_ids": [1, 2, 9], "attention_mask": [1, 1, 1], "labels": [-100] * 3},
        {"input_ids": [1, 3, 4, 9], "attention_mask": [1] * 4},
    ])
    torch.testing.assert_close(batch["labels"], torch.tensor([[1, 2, 9, -100], [1, 3, 4, 9]]))
    assert "alpha_weights" not in batch


def test_completion_masking_reports_only_active_response_tokens():
    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": [10, 11, 17, 18], "attention_mask": [1, 1, 1, 1]}

        def decode(self, ids):
            return " ".join(map(str, ids.tolist()))

    class Collator:
        def __call__(self, _features):
            return {
                "input_ids": torch.tensor([[10, 11, 17, 18]]),
                "labels": torch.tensor([[-100, -100, 17, 18]]),
            }

    result = validate_completion_masking(Collator(), Tokenizer(), "prompt")
    assert result["masked_tokens"] == 2
    assert result["active_tokens"] == 2
    assert result["active_text"] == "17 18"
