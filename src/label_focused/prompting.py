"""Unified prompt construction for all supported datasets."""

from __future__ import annotations

import json
from typing import Any

from .datasets import DatasetSpec


def _map_label(value: Any, mapping: dict[str, str], field: str) -> str:
    key = str(value).strip()
    if key in mapping:
        return mapping[key]
    if key in mapping.values():
        return key
    raise ValueError(f"Unknown {field} label: {value!r}")


def labels_from_row(row: Any, spec: DatasetSpec) -> tuple[str, str]:
    sentiment = _map_label(
        row[spec.sentiment_column], spec.sentiment_map, spec.sentiment_column
    )
    topic = _map_label(row[spec.topic_column], spec.topic_map, spec.topic_column)
    return sentiment, topic


def _messages(text: str, spec: DatasetSpec, target: dict | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": spec.system_prompt},
        {
            "role": "user",
            "content": f'{spec.one_shot}\n\n{spec.input_label}: "{text}"',
        },
    ]
    if target is not None:
        messages.append(
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        )
    return messages


def build_training_prompt(row: Any, spec: DatasetSpec, tokenizer: Any) -> str:
    sentiment, topic = labels_from_row(row, spec)
    target = {"Sentiment": sentiment, "Topic": topic}
    return tokenizer.apply_chat_template(
        _messages(str(row[spec.text_column]), spec, target),
        tokenize=False,
        add_generation_prompt=False,
    )


def build_inference_prompt(text: str, spec: DatasetSpec, tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        _messages(str(text), spec), tokenize=False, add_generation_prompt=True
    )


def build_zero_shot_prompt(text: str, spec: DatasetSpec, tokenizer: Any) -> str:
    """Build the Vistral baseline prompt without the one-shot exemplar."""
    messages = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": f'{spec.input_label}: "{text}"'},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _single_task_metadata(spec: DatasetSpec, task: str) -> tuple[str, str, str]:
    if task == "sentiment":
        labels = ", ".join(spec.sentiment_labels)
        system = (
            "Phân tích cảm xúc (Sentiment) của đoạn văn bản.\n"
            f"Nhãn hợp lệ: {labels}.\n"
            "Chỉ trả về JSON với đúng một khóa: Sentiment. Không giải thích thêm."
        )
        return "Sentiment", spec.one_shot_sentiment, system
    if task == "topic":
        labels = ", ".join(spec.topic_labels)
        system = (
            "Phân loại chủ đề (Topic) của đoạn văn bản.\n"
            f"Nhãn hợp lệ: {labels}.\n"
            "Chỉ trả về JSON với đúng một khóa: Topic. Không giải thích thêm."
        )
        return "Topic", spec.one_shot_topic, system
    raise ValueError("task must be 'sentiment' or 'topic'")


def build_single_task_prompt(
    row: Any, spec: DatasetSpec, tokenizer: Any, task: str, training: bool = True
) -> str:
    """Build the independent one-key prompt used by a Dual Adapter."""
    key, example_label, system = _single_task_metadata(spec, task)
    example = (
        f'Ví dụ:\n{spec.input_label}: "{spec.one_shot_text}"\n'
        f'Output: {{"{key}": "{example_label}"}}'
    )
    text = str(row[spec.text_column]) if not isinstance(row, str) else row
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f'{example}\n\n{spec.input_label}: "{text}"',
        },
    ]
    if training:
        sentiment, topic = labels_from_row(row, spec)
        label = sentiment if task == "sentiment" else topic
        messages.append(
            {"role": "assistant", "content": json.dumps({key: label}, ensure_ascii=False)}
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=not training
    )


def parse_prediction(raw: str) -> tuple[str, str]:
    """Extract both labels independently from the last JSON object."""
    start, end = raw.rfind("{"), raw.rfind("}")
    if start < 0 or end < start:
        return "PARSE_ERROR", "PARSE_ERROR"
    try:
        value = json.loads(raw[start : end + 1])
        if not isinstance(value, dict):
            return "PARSE_ERROR", "PARSE_ERROR"
        sentiment = str(value.get("Sentiment", "PARSE_ERROR")).strip()
        topic = str(value.get("Topic", "PARSE_ERROR")).strip()
        return sentiment, topic
    except (json.JSONDecodeError, TypeError):
        return "PARSE_ERROR", "PARSE_ERROR"


def parse_single_prediction(raw: str, task: str) -> str:
    """Extract a one-key Dual Adapter prediction without hiding parse errors."""
    key = {"sentiment": "Sentiment", "topic": "Topic"}.get(task)
    if key is None:
        raise ValueError("task must be 'sentiment' or 'topic'")
    start, end = raw.rfind("{"), raw.rfind("}")
    if start < 0 or end < start:
        return "PARSE_ERROR"
    try:
        value = json.loads(raw[start : end + 1])
        if not isinstance(value, dict):
            return "PARSE_ERROR"
        return str(value.get(key, "PARSE_ERROR")).strip()
    except (json.JSONDecodeError, TypeError):
        return "PARSE_ERROR"
