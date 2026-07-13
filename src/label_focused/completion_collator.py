"""Completion-only masking used by the Label-Focused Tuning Pipeline."""

from __future__ import annotations

from typing import Any


def make_completion_collator(tokenizer: Any):
    """Mask every token before the Vistral assistant response marker."""
    from trl import DataCollatorForCompletionOnlyLM

    response_template_ids = tokenizer.encode("[/INST]", add_special_tokens=False)
    if not response_template_ids:
        raise ValueError("Tokenizer did not encode the [/INST] response marker")
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )


def validate_completion_masking(collator: Any, tokenizer: Any, prompt: str) -> dict:
    """Fail early if no assistant completion tokens contribute to loss."""
    feature = tokenizer(prompt, truncation=True, add_special_tokens=False)
    batch = collator([feature])
    labels = batch["labels"][0]
    active = labels.ne(-100)
    active_count = int(active.sum().item())
    if active_count == 0:
        raise RuntimeError("Completion masking found no assistant response tokens")
    decoded = tokenizer.decode(batch["input_ids"][0][active])
    return {
        "masked_tokens": int((~active).sum().item()),
        "active_tokens": active_count,
        "active_text": decoded,
    }
