"""vLLM inference for Joint and Dual Adapter architectures."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .datasets import DATASET_REGISTRY
from .evaluation import majority_labels, correct_full_test_predictions
from .prompting import (
    build_inference_prompt,
    build_single_task_prompt,
    build_zero_shot_prompt,
    labels_from_row,
    parse_prediction,
    parse_single_prediction,
)


@dataclass
class InferenceConfig:
    dataset: str
    architecture: str
    model_id: str = "Viet-Mistral/Vistral-7B-Chat"
    joint_adapter: str | None = None
    sentiment_adapter: str | None = None
    topic_adapter: str | None = None
    output_dir: str = "outputs/inference"
    max_lora_rank: int = 64
    max_model_length: int = 1024
    max_new_tokens: int = 30
    gpu_memory_utilization: float = 0.90


class VramMonitor:
    """Sample device memory through NVML while generation is running."""

    def __init__(self, device_index: int = 0, interval: float = 0.1):
        self.device_index = device_index
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = False

    def __enter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.available = True
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        except Exception:
            self.available = False
        return self

    def _sample(self):
        while not self._stop.is_set():
            used = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
            self.peak_bytes = max(self.peak_bytes, int(used))
            self._stop.wait(self.interval)

    def __exit__(self, *_):
        if self.available:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=2)
            self._pynvml.nvmlShutdown()

    @property
    def peak_gb(self) -> float | None:
        return self.peak_bytes / 1024**3 if self.available else None


def _validate_config(config: InferenceConfig) -> None:
    if config.architecture == "zero_shot":
        return
    if config.architecture == "joint":
        if not config.joint_adapter:
            raise ValueError("Joint inference requires joint_adapter")
    elif config.architecture == "dual_adapter":
        if not config.sentiment_adapter or not config.topic_adapter:
            raise ValueError(
                "Dual inference requires sentiment_adapter and topic_adapter"
            )
    else:
        raise ValueError("architecture must be 'zero_shot', 'joint', or 'dual_adapter'")


def _token_counts(outputs: list[Any]) -> tuple[int, int]:
    input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    return input_tokens, output_tokens


def run_inference(
    test_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    config: InferenceConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate, correct, and retain all test predictions."""
    _validate_config(config)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    spec = DATASET_REGISTRY[config.dataset]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, trust_remote_code=True
    )
    uses_lora = config.architecture != "zero_shot"
    llm = LLM(
        model=config.model_id,
        enable_lora=uses_lora,
        max_lora_rank=config.max_lora_rank,
        dtype="half",
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_length,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        max_tokens=config.max_new_tokens,
        temperature=0.0,
        stop=["\n"],
    )

    true_labels = [labels_from_row(row, spec) for _, row in test_frame.iterrows()]
    texts = test_frame[spec.text_column].astype(str).tolist()
    if config.architecture in {"zero_shot", "joint"}:
        prompt_builder = (
            build_zero_shot_prompt
            if config.architecture == "zero_shot"
            else build_inference_prompt
        )
        joint_prompts = [prompt_builder(text, spec, tokenizer) for text in texts]
    else:
        sentiment_prompts = [
            build_single_task_prompt(text, spec, tokenizer, "sentiment", training=False)
            for text in texts
        ]
        topic_prompts = [
            build_single_task_prompt(text, spec, tokenizer, "topic", training=False)
            for text in texts
        ]
    all_outputs: list[Any] = []
    started = time.perf_counter()
    with VramMonitor() as monitor:
        if config.architecture in {"zero_shot", "joint"}:
            request = (
                None
                if config.architecture == "zero_shot"
                else LoRARequest("joint", 1, str(config.joint_adapter))
            )
            generate_kwargs = {"use_tqdm": True}
            if request is not None:
                generate_kwargs["lora_request"] = request
            outputs = llm.generate(joint_prompts, sampling, **generate_kwargs)
            all_outputs.extend(outputs)
            raw_joint = [output.outputs[0].text for output in outputs]
            parsed = [parse_prediction(raw) for raw in raw_joint]
            raw_sentiment = raw_joint
            raw_topic = raw_joint
        else:
            sentiment_outputs = llm.generate(
                sentiment_prompts,
                sampling,
                lora_request=LoRARequest(
                    "sentiment", 1, str(config.sentiment_adapter)
                ),
                use_tqdm=True,
            )
            topic_outputs = llm.generate(
                topic_prompts,
                sampling,
                lora_request=LoRARequest("topic", 2, str(config.topic_adapter)),
                use_tqdm=True,
            )
            all_outputs.extend(sentiment_outputs)
            all_outputs.extend(topic_outputs)
            raw_sentiment = [output.outputs[0].text for output in sentiment_outputs]
            raw_topic = [output.outputs[0].text for output in topic_outputs]
            parsed = list(
                zip(
                    [parse_single_prediction(raw, "sentiment") for raw in raw_sentiment],
                    [parse_single_prediction(raw, "topic") for raw in raw_topic],
                )
            )
    elapsed = time.perf_counter() - started

    input_tokens, output_tokens = _token_counts(all_outputs)
    predictions = pd.DataFrame(
        {
            "text": texts,
            "true_sentiment": [pair[0] for pair in true_labels],
            "true_topic": [pair[1] for pair in true_labels],
            "pred_sentiment": [pair[0] for pair in parsed],
            "pred_topic": [pair[1] for pair in parsed],
            "raw_sentiment_output": raw_sentiment,
            "raw_topic_output": raw_topic,
        }
    )
    majority_sentiment, majority_topic = majority_labels(train_frame, spec)
    corrected = correct_full_test_predictions(
        predictions, spec, majority_sentiment, majority_topic
    )
    performance = {
        "architecture": config.architecture,
        "dataset": config.dataset,
        "test_size": len(corrected),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(corrected) / elapsed,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "input_tokens_per_second": input_tokens / elapsed,
        "output_tokens_per_second": output_tokens / elapsed,
        "overall_tokens_per_second": (input_tokens + output_tokens) / elapsed,
        "peak_vram_gb": monitor.peak_gb,
        "majority_fallback": {
            "sentiment": majority_sentiment,
            "topic": majority_topic,
        },
        "invalid_counts": {
            "sentiment": int(corrected["sentiment_invalid"].sum()),
            "topic": int(corrected["topic_invalid"].sum()),
            "rows_with_any_invalid": int(corrected["any_invalid"].sum()),
        },
        "parse_error_counts": {
            "sentiment": int((corrected["pred_sentiment_original"] == "PARSE_ERROR").sum()),
            "topic": int((corrected["pred_topic_original"] == "PARSE_ERROR").sum()),
        },
    }
    return corrected, performance
