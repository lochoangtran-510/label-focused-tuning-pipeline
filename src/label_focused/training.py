"""Shared model, dataset, masking, focal-loss, and trainer construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import gc
from pathlib import Path
from typing import Any
import json

import pandas as pd
import torch

from .datasets import DATASET_REGISTRY, DatasetSpec
from .completion_collator import make_completion_collator, validate_completion_masking
from .losses import flatten_label_token_weights, label_focused_loss
from .prompting import build_single_task_prompt, build_training_prompt, labels_from_row


@dataclass
class ExperimentConfig:
    dataset: str = "neu_esc"
    model_id: str = "Viet-Mistral/Vistral-7B-Chat"
    output_root: str = "outputs"
    lora_rank: int = 32
    lora_alpha: int | None = None
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    epochs: int = 3
    max_steps: int = -1
    batch_size: int = 6
    gradient_accumulation: int = 4
    max_length: int = 1024
    use_masking: bool = True
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    seed: int = 42
    architecture: str = "joint"
    sentiment_lora_rank: int | None = None
    topic_lora_rank: int | None = None

    @property
    def output_dir(self) -> str:
        if self.use_masking and self.use_focal_loss:
            method = "full"
        elif self.use_masking:
            method = "masking"
        elif self.use_focal_loss:
            method = "focal"
        else:
            method = "base"
        rank_name = f"r{self.lora_rank}"
        if self.architecture == "dual_adapter":
            sentiment_rank = self.sentiment_lora_rank or self.lora_rank
            topic_rank = self.topic_lora_rank or self.lora_rank
            rank_name = f"r{sentiment_rank}_r{topic_rank}"
        return str(
            Path(self.output_root)
            / self.dataset
            / self.architecture
            / f"{method}_{rank_name}"
        )


def load_model_and_tokenizer(config: ExperimentConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def make_prompt_datasets(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    spec: DatasetSpec,
    tokenizer: Any,
    max_length: int,
):
    from datasets import Dataset

    def prompts(frame: pd.DataFrame) -> list[str]:
        return [
            build_training_prompt(row, spec, tokenizer)
            for _, row in frame.iterrows()
        ]

    train_prompts = prompts(train_frame)
    lengths = [
        len(ids)
        for ids in tokenizer(train_prompts, add_special_tokens=False)["input_ids"]
    ]
    kept_prompts = [p for p, length in zip(train_prompts, lengths) if length <= max_length]
    validation_prompts = prompts(validation_frame)
    print(f"Training samples: {len(kept_prompts)}/{len(train_prompts)} after length filter")
    return (
        Dataset.from_dict({"text": kept_prompts}),
        Dataset.from_dict({"text": validation_prompts}),
    )


def make_single_task_datasets(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    spec: DatasetSpec,
    tokenizer: Any,
    max_length: int,
    task: str,
):
    """Create the Sentiment or Topic dataset for one independent adapter."""
    from datasets import Dataset

    def prompts(frame: pd.DataFrame) -> list[str]:
        return [
            build_single_task_prompt(row, spec, tokenizer, task, training=True)
            for _, row in frame.iterrows()
        ]

    train_prompts = prompts(train_frame)
    lengths = [
        len(ids)
        for ids in tokenizer(train_prompts, add_special_tokens=False)["input_ids"]
    ]
    kept = [prompt for prompt, length in zip(train_prompts, lengths) if length <= max_length]
    validation_prompts = prompts(validation_frame)
    print(f"{task}: {len(kept)}/{len(train_prompts)} training samples after filter")
    return Dataset.from_dict({"text": kept}), Dataset.from_dict({"text": validation_prompts})


def sample_weights(
    frame: pd.DataFrame, spec: DatasetSpec, task: str | None = None
) -> dict[str, float]:
    """Inverse-frequency alpha weights, normalized separately for both tasks."""
    pairs = [labels_from_row(row, spec) for _, row in frame.iterrows()]
    result: dict[str, float] = {}
    columns = list(zip(*pairs))
    if task == "sentiment":
        columns = columns[:1]
    elif task == "topic":
        columns = columns[1:]
    for values in columns:
        counts = Counter(values)
        total, classes = sum(counts.values()), len(counts)
        result.update({label: total / (classes * count) for label, count in counts.items()})
    return result


def label_token_map(
    spec: DatasetSpec, tokenizer: Any, task: str | None = None
) -> dict[str, list[int]]:
    if task == "sentiment":
        labels = spec.sentiment_labels
    elif task == "topic":
        labels = spec.topic_labels
    else:
        labels = (*spec.sentiment_labels, *spec.topic_labels)
    result = {}
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Tokenizer produced no IDs for label {label!r}")
        result[label] = [token_ids[-1]]
    return result


def make_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    train_frame: pd.DataFrame,
    config: ExperimentConfig,
    task: str | None = None,
):
    from peft import LoraConfig, get_peft_model
    from transformers import EarlyStoppingCallback, TrainingArguments
    from trl import SFTTrainer

    spec = DATASET_REGISTRY[config.dataset]
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha or 2 * config.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    collator = None
    if config.use_masking:
        collator = make_completion_collator(tokenizer)
        masking_check = validate_completion_masking(
            collator, tokenizer, train_dataset[0]["text"]
        )
        print(f"Completion masking check: {masking_check}")

    output_dir = str(Path(config.output_dir) / task) if task else config.output_dir
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2 * config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        optim="adamw_torch_fused",
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="tensorboard",
        group_by_length=True,
        seed=config.seed,
    )

    common = dict(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=args,
        dataset_text_field="text",
        max_seq_length=config.max_length,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )
    if not config.use_focal_loss:
        return SFTTrainer(**common)

    class FocalTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            self.gamma = kwargs.pop("focal_gamma")
            self.alpha = kwargs.pop("sample_weights")
            self.label_ids = kwargs.pop("label_token_ids")
            self.token_weights = flatten_label_token_weights(
                self.label_ids, self.alpha
            )
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            outputs = model(**inputs)
            loss = label_focused_loss(
                outputs.logits, labels, self.gamma, self.token_weights
            )
            return (loss, outputs) if return_outputs else loss

    return FocalTrainer(
        **common,
        focal_gamma=config.focal_gamma,
        sample_weights=sample_weights(train_frame, spec, task),
        label_token_ids=label_token_map(spec, tokenizer, task),
    )


def train_dual_adapters(
    splits: dict[str, pd.DataFrame], config: ExperimentConfig
) -> dict[str, str]:
    """Train two fully independent LoRA adapters sequentially.

    Sequential loading keeps peak memory close to Joint Training and guarantees
    that no parameters are shared between the two tasks.
    """
    dual_config = replace(config, architecture="dual_adapter")
    spec = DATASET_REGISTRY[dual_config.dataset]
    saved: dict[str, str] = {}
    for task in ("sentiment", "topic"):
        print(f"\n=== Training independent {task} adapter ===")
        task_rank = (
            dual_config.sentiment_lora_rank
            if task == "sentiment"
            else dual_config.topic_lora_rank
        )
        task_config = replace(
            dual_config,
            lora_rank=task_rank or dual_config.lora_rank,
        )
        model, tokenizer = load_model_and_tokenizer(task_config)
        train_dataset, validation_dataset = make_single_task_datasets(
            splits["train"],
            splits["validation"],
            spec,
            tokenizer,
            task_config.max_length,
            task,
        )
        trainer = make_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            train_frame=splits["train"],
            config=task_config,
            task=task,
        )
        train_result = trainer.train()
        best_dir = Path(task_config.output_dir) / task / "best_model"
        trainer.model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        run_dir = best_dir.parent
        (run_dir / "training_metrics.json").write_text(
            json.dumps(train_result.metrics, default=str, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "trainer_log_history.json").write_text(
            json.dumps(trainer.state.log_history, default=str, indent=2) + "\n",
            encoding="utf-8",
        )
        saved[task] = str(best_dir)

        del trainer, model, tokenizer, train_dataset, validation_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return saved
