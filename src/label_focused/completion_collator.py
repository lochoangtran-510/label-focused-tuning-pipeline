"""Completion-only masking used by the Label-Focused Tuning Pipeline."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import json
import math
import re
from typing import Any

import torch


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _label_token_spans(
    token_ids: Sequence[int], tokenizer: Any, fields: Collection[str]
) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    """Locate label characters in the actual completion tokenization."""
    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        raise ValueError("Completion has no active tokens")
    decode_kwargs = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoded = tokenizer.decode(ids, **decode_kwargs)
    start = decoded.find("{")
    if start < 0 or decoded[:start].strip():
        raise ValueError("The masked completion must start with its target JSON")
    try:
        target, end = json.JSONDecoder(object_pairs_hook=_unique_json_object).raw_decode(
            decoded, start
        )
    except json.JSONDecodeError as error:
        raise ValueError("Incomplete target JSON; check truncation and completion masking") from error
    if not isinstance(target, dict) or set(target) != set(fields):
        raise ValueError(f"Target JSON must contain exactly these fields: {sorted(fields)}")
    if any(not isinstance(value, str) or not value for value in target.values()):
        raise ValueError("Target JSON labels must be non-empty strings")

    # A completion may end in EOS/chat special tokens, but not another answer.
    tail = decoded[end:].strip()
    special_tokens = set(getattr(tokenizer, "all_special_tokens", ()) or ())
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        special_tokens.add(eos)
    allowed_suffixes = sorted((value for value in special_tokens if value), key=len, reverse=True)
    while tail:
        suffix = next((value for value in allowed_suffixes if tail.startswith(value)), None)
        if suffix is None:
            raise ValueError("Completion contains text after its target JSON")
        tail = tail[len(suffix):].lstrip()

    # Decoding a label in isolation can produce different IDs (e.g. Others).
    # Prefix boundaries preserve the tokenization already used for training.
    boundaries = [0]
    for stop in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:stop], **decode_kwargs)
        if not decoded.startswith(prefix) or len(prefix) < boundaries[-1]:
            raise ValueError("Tokenizer prefix decode is unstable; cannot assign alpha safely")
        boundaries.append(len(prefix))

    field_pattern = "|".join(re.escape(field) for field in fields)
    pattern = re.compile(
        rf'"(?P<field>{field_pattern})"\s*:\s*"(?P<label>[^"\\]*)"'
    )
    spans = {}
    occupied = set()
    for match in pattern.finditer(decoded[start:end]):
        field, label = match.group("field"), match.group("label")
        if field in spans or target[field] != label:
            raise ValueError("Cannot identify unique unescaped target label spans")
        left, right = start + match.start("label"), start + match.end("label")
        positions = tuple(
            index for index in range(len(ids))
            if boundaries[index + 1] > boundaries[index]
            and boundaries[index + 1] > left and boundaries[index] < right
        )
        if not positions or positions != tuple(range(positions[0], positions[-1] + 1)):
            raise ValueError(f"No contiguous token span for {field}={label}")
        if occupied.intersection(positions):
            raise ValueError("Different label fields share a token; independent alpha is ambiguous")
        occupied.update(positions)
        spans[field] = positions
    if set(spans) != set(fields):
        raise ValueError("Cannot identify all target label spans")
    return target, spans


class TaskAwareAlphaCollator:
    """Add target-position alpha while preserving the wrapped completion mask.

    Fields have separate class tables even when names or token IDs overlap.
    Terminal-only alpha is the method default; weighting every label token is
    an explicit experimental option. JSON syntax and other tokens keep alpha 1.
    A full-sequence inner collator is also supported: its prompt labels remain
    active, while alpha is assigned only after the last Vistral response marker.
    """

    def __init__(
        self,
        inner: Any,
        tokenizer: Any,
        weights_by_field: Mapping[str, Mapping[str, float]],
        alpha_scope: str = "terminal_only",
    ):
        if alpha_scope not in {"terminal_only", "all_label_tokens"}:
            raise ValueError("alpha_scope must be terminal_only or all_label_tokens")
        if not isinstance(weights_by_field, Mapping) or not weights_by_field:
            raise ValueError("weights_by_field must contain at least one field/class table")
        self.weights_by_field = {}
        for field, class_weights in weights_by_field.items():
            if not isinstance(field, str) or not field or not isinstance(class_weights, Mapping):
                raise ValueError("Each JSON field must have its own class-weight mapping")
            if not class_weights:
                raise ValueError(f"The class-weight table for {field} is empty")
            table = {}
            for label, weight in class_weights.items():
                if not isinstance(label, str) or not label:
                    raise ValueError("Class labels must be non-empty strings")
                try:
                    value = float(weight)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid alpha for {field}={label}") from error
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Alpha for {field}={label} must be finite and positive")
                table[label] = value
            self.weights_by_field[field] = table
        self.inner = inner
        self.tokenizer = tokenizer
        self.alpha_scope = alpha_scope
        self._cache = {}

    def _response_ids(self, ids):
        decoded = self.tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded.lstrip().startswith("{"):
            return ids, 0
        marker = self.tokenizer.encode("[/INST]", add_special_tokens=False)
        if not marker:
            raise ValueError("Tokenizer did not encode the [/INST] response marker")
        width = len(marker)
        ends = [
            start + width for start in range(len(ids) - width + 1)
            if list(ids[start:start + width]) == list(marker)
        ]
        if not ends:
            raise ValueError("Active labels contain neither target JSON nor a response marker")
        offset = ends[-1]
        return ids[offset:], offset

    def __call__(self, features):
        batch = self.inner(features)
        labels = batch["labels"]
        if labels.ndim != 2:
            raise ValueError("Completion labels must have shape [batch, tokens]")
        alpha = torch.ones_like(labels, dtype=torch.float32)
        for row in range(labels.size(0)):
            active = torch.where(labels[row] != -100)[0]
            if active.numel() == 0:
                raise ValueError("A sample has no active completion tokens")
            if active.numel() > 1 and not torch.all(active[1:] - active[:-1] == 1).item():
                raise ValueError("Completion tokens must be contiguous; packed examples are unsupported")
            ids, offset = self._response_ids(tuple(labels[row, active].tolist()))
            if ids not in self._cache:
                self._cache[ids] = _label_token_spans(
                    ids, self.tokenizer, self.weights_by_field.keys()
                )
            target, spans = self._cache[ids]
            for field, table in self.weights_by_field.items():
                label = target[field]
                if label not in table:
                    raise ValueError(f"Missing class weight for {field}={label}")
                relative = spans[field]
                if self.alpha_scope == "terminal_only":
                    relative = relative[-1:]
                alpha[row, active[[offset + index for index in relative]]] = table[label]
        batch["alpha_weights"] = alpha
        return batch


class FullSequenceCECollator:
    """Create full-sequence labels, masking padding by attention rather than ID."""

    def __init__(self, tokenizer: Any):
        from transformers import DataCollatorWithPadding

        self.pad = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    def __call__(self, features):
        if not features:
            raise ValueError("Cannot collate an empty batch")
        model_features = [
            {key: value for key, value in feature.items() if key in {"input_ids", "attention_mask"}}
            for feature in features
        ]
        batch = self.pad(model_features)
        labels = batch["input_ids"].clone()
        if labels.shape != batch["attention_mask"].shape:
            raise ValueError("input_ids and attention_mask shapes must match")
        labels[batch["attention_mask"] == 0] = -100
        if (labels[:, 1:] != -100).sum(dim=1).eq(0).any().item():
            raise ValueError("A sample has no causal targets after padding")
        batch["labels"] = labels
        return batch


def make_full_sequence_collator(tokenizer: Any):
    """Full-prompt CE baseline: preserve real EOS even when EOS is the pad ID."""
    return FullSequenceCECollator(tokenizer)


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
