"""Reusable pipeline for multitask Vietnamese text classification."""

from .datasets import (
    DATASET_REGISTRY,
    DatasetSpec,
    load_huggingface_splits,
    load_splits,
)
from .prompting import (
    build_inference_prompt,
    build_single_task_prompt,
    build_training_prompt,
    build_zero_shot_prompt,
)

__all__ = [
    "DATASET_REGISTRY",
    "DatasetSpec",
    "build_inference_prompt",
    "build_single_task_prompt",
    "build_training_prompt",
    "build_zero_shot_prompt",
    "load_splits",
    "load_huggingface_splits",
]
