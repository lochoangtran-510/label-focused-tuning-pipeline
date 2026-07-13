import torch

from label_focused.completion_collator import validate_completion_masking


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
