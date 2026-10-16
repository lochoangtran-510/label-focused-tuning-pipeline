"""Shared training with independent masking, class alpha, and focal switches."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .datasets import DATASET_REGISTRY, DatasetSpec
from .completion_collator import (
    TaskAwareAlphaCollator, make_completion_collator, make_full_sequence_collator,
    validate_completion_masking,
)
from .losses import label_focused_loss
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
    use_class_alpha: bool | None = None
    alpha_scope: str = "terminal_only"
    protocol: str = "paper"
    seed: int = 42
    architecture: str = "joint"
    sentiment_lora_rank: int | None = None
    topic_lora_rank: int | None = None

    @property
    def class_alpha_enabled(self) -> bool:
        """Omitted alpha retains the legacy focal-implies-alpha configuration."""
        return self.use_focal_loss if self.use_class_alpha is None else self.use_class_alpha

    @property
    def loss_gamma(self) -> float:
        return float(self.focal_gamma) if self.use_focal_loss else 0.0

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
            rank_name = (
                f"r{self.sentiment_lora_rank or self.lora_rank}"
                f"_r{self.topic_lora_rank or self.lora_rank}"
            )
        legacy = (
            self.protocol == "paper" and self.alpha_scope == "terminal_only"
            and self.class_alpha_enabled == self.use_focal_loss and self.seed == 42
            and (not self.use_focal_loss or self.focal_gamma == 2.0) and self.max_steps == -1
        )
        if legacy:
            run_name = f"{method}_{rank_name}"
        else:
            objective = "focal" if self.use_focal_loss else "ce"
            mask = "mask" if self.use_masking else "full_sequence"
            alpha = f"alpha_{self.alpha_scope}" if self.class_alpha_enabled else "no_alpha"
            run_name = (
                f"{mask}_{objective}_{alpha}_{rank_name}_len{self.max_length}"
                f"_seed{self.seed}{'_gamma' + str(self.loss_gamma) if self.use_focal_loss else ''}"
                f"{'_steps' + str(self.max_steps) if self.max_steps > 0 else ''}_{self.protocol}"
            )
        return str(Path(self.output_root) / self.dataset / self.architecture / run_name)


def load_model_and_tokenizer(config: ExperimentConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def _make_datasets(train_frame, validation_frame, spec, tokenizer, max_length, task=None):
    from datasets import Dataset

    if train_frame.empty or validation_frame.empty:
        raise ValueError("Train and validation must both contain examples")

    def prompts(frame):
        return [
            build_training_prompt(row, spec, tokenizer) if task is None else
            build_single_task_prompt(row, spec, tokenizer, task, training=True)
            for _, row in frame.iterrows()
        ]

    def lengths(texts):
        return [len(ids) for ids in tokenizer(
            texts, add_special_tokens=False, truncation=False, padding=False,
        )["input_ids"]]

    train_prompts = prompts(train_frame)
    train_lengths = lengths(train_prompts)
    kept = [i for i, length in enumerate(train_lengths) if length <= max_length]
    if not kept:
        raise ValueError("No training examples remain after the length filter")
    validation_prompts = prompts(validation_frame)
    validation_lengths = lengths(validation_prompts)
    too_long = [i for i, length in enumerate(validation_lengths) if length > max_length]
    if too_long:
        raise ValueError(
            f"{len(too_long)} validation examples exceed max_length={max_length}; "
            f"positions {too_long[:10]}. Refusing response truncation or split changes."
        )
    train_dataset = Dataset.from_dict({"text": [train_prompts[i] for i in kept]})
    eval_dataset = Dataset.from_dict({"text": validation_prompts})
    # Keep the public two-return API while carrying the exact rows used for alpha.
    train_dataset._label_focused_filtered_frame = train_frame.iloc[kept].copy()
    train_dataset._label_focused_data_manifest = {
        "raw_train_examples": len(train_frame), "filtered_train_examples": len(kept),
        "removed_train_examples": len(train_frame) - len(kept),
        "validation_examples": len(validation_frame), "max_length": max_length,
        "add_special_tokens": False, "packing": False,
        "kept_train_positions": kept,
        "train_text_sha256": hashlib.sha256(
            json.dumps([train_prompts[i] for i in kept], ensure_ascii=False).encode()
        ).hexdigest(),
    }
    print(f"{task or 'Joint'} training samples: {len(kept)}/{len(train_frame)} after length filter")
    return train_dataset, eval_dataset


def make_prompt_datasets(train_frame, validation_frame, spec, tokenizer, max_length):
    return _make_datasets(train_frame, validation_frame, spec, tokenizer, max_length)


def make_single_task_datasets(train_frame, validation_frame, spec, tokenizer, max_length, task):
    if task not in {"sentiment", "topic"}:
        raise ValueError(f"Unsupported task: {task}")
    return _make_datasets(train_frame, validation_frame, spec, tokenizer, max_length, task)


def filtered_train_frame(train_dataset: Any) -> pd.DataFrame:
    """Return exactly the retained frame; never fall back to unfiltered class counts."""
    frame = getattr(train_dataset, "_label_focused_filtered_frame", None)
    if frame is None:
        raise ValueError("Build the training dataset with make_prompt_datasets or make_single_task_datasets")
    if len(frame) != len(train_dataset):
        raise ValueError("Filtered train metadata no longer matches the training dataset")
    return frame.copy()


def _field_labels(spec, task=None):
    if task == "sentiment":
        return {"Sentiment": spec.sentiment_labels}
    if task == "topic":
        return {"Topic": spec.topic_labels}
    if task is not None:
        raise ValueError(f"Unsupported task: {task}")
    return {"Sentiment": spec.sentiment_labels, "Topic": spec.topic_labels}


def class_counts(frame, spec, task=None):
    pairs = [labels_from_row(row, spec) for _, row in frame.iterrows()]
    return {
        field: dict(Counter(pair[0 if field == "Sentiment" else 1] for pair in pairs))
        for field in _field_labels(spec, task)
    }


def sample_weights(frame, spec, task=None) -> dict[str, dict[str, float]]:
    """Balanced weights N/(K*n_c), separately by field, from retained train only."""
    counts_by_field = class_counts(frame, spec, task)
    result = {}
    for field, expected in _field_labels(spec, task).items():
        counts = counts_by_field[field]
        missing = set(expected) - set(counts)
        if missing:
            raise ValueError(f"Filtered train is missing {field} classes: {sorted(missing)}")
        total = sum(counts.values())
        result[field] = {name: total / (len(expected) * counts[name]) for name in expected}
    return result


def tokenizer_fingerprint(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialized = backend.to_str() if backend is not None else None
    payload = {
        "backend": serialized,
        "vocabulary": None if serialized else getattr(tokenizer, "get_vocab", lambda: {})(),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "padding_side": getattr(tokenizer, "padding_side", None),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def ensure_new_run(path, allowed_files=()):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(item.name not in allowed_files for item in path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing run: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_trainer(model, tokenizer, train_dataset, eval_dataset, train_frame,
                 config: ExperimentConfig, task=None):
    from peft import LoraConfig, get_peft_model
    from transformers import EarlyStoppingCallback, TrainingArguments, set_seed
    from trl import SFTTrainer

    if config.alpha_scope not in {"terminal_only", "all_label_tokens"}:
        raise ValueError(f"Unsupported alpha_scope: {config.alpha_scope}")
    if not len(train_dataset) or not len(eval_dataset):
        raise ValueError("Train and validation datasets cannot be empty")
    retained = filtered_train_frame(train_dataset)
    spec = DATASET_REGISTRY[config.dataset]
    weights = sample_weights(retained, spec, task) if config.class_alpha_enabled else {}
    output_dir = Path(config.output_dir) / task if task else Path(config.output_dir)
    # CLI creates only these metadata files before trainer construction.
    ensure_new_run(output_dir, allowed_files={"resolved_config.json", "source_config.json", "environment.json"})
    inner = make_completion_collator(tokenizer) if config.use_masking else make_full_sequence_collator(tokenizer)
    collator = TaskAwareAlphaCollator(inner, tokenizer, weights, alpha_scope=config.alpha_scope) if weights else inner
    if config.use_masking:
        print(f"Completion masking check: {validate_completion_masking(inner, tokenizer, train_dataset[0]['text'])}")

    peft_config = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha or 2 * config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.lora_dropout, bias="none", task_type="CAUSAL_LM",
    )
    # Reset immediately before random LoRA initialization for paired ablations.
    set_seed(config.seed)
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=config.epochs, max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=2 * config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation, optim="adamw_torch_fused",
        learning_rate=config.learning_rate, weight_decay=0.01, max_grad_norm=1.0,
        lr_scheduler_type="cosine", warmup_ratio=0.05, bf16=True, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        evaluation_strategy="steps", eval_steps=200, save_strategy="steps", save_steps=200,
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=10, report_to="tensorboard", group_by_length=True,
        seed=config.seed, data_seed=config.seed, label_names=["labels"], prediction_loss_only=True,
        remove_unused_columns=False,
    )

    class ObjectiveTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            model_inputs = dict(inputs)
            alpha = model_inputs.pop("alpha_weights", None)
            if config.class_alpha_enabled and alpha is None:
                raise ValueError("Class alpha is enabled but the collator did not provide alpha_weights")
            if not config.class_alpha_enabled and alpha is not None:
                raise ValueError("Class alpha is disabled but alpha_weights were provided")
            labels = model_inputs.pop("labels")
            outputs = model(**model_inputs)
            loss = label_focused_loss(outputs.logits, labels, config.loss_gamma, alpha_weights=alpha)
            outputs["loss"] = loss
            return (loss, outputs) if return_outputs else loss

    trainer = ObjectiveTrainer(
        model=model, train_dataset=train_dataset, eval_dataset=eval_dataset,
        tokenizer=tokenizer, args=args, dataset_text_field="text", max_seq_length=config.max_length,
        packing=False, dataset_kwargs={"add_special_tokens": False}, data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )
    manifest = {
        "config": asdict(config), "task": task or "joint",
        "data": getattr(train_dataset, "_label_focused_data_manifest", {}),
        "class_counts": class_counts(retained, spec, task), "class_weights_by_field": weights,
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "semantics": {
            "completion_only_masking": config.use_masking,
            "class_alpha_enabled": config.class_alpha_enabled, "alpha_scope": config.alpha_scope,
            "focal_gamma": config.loss_gamma, "normalization": "mean_active_shifted_tokens",
            "alpha_source": "length_filtered_train_only",
            "padding_mask": (
                "trl_completion_mask_and_pad_id_mask" if config.use_masking
                else "attention_mask_equals_zero"
            ),
            "true_eos_supervised": (
                not config.use_masking or tokenizer.pad_token_id != tokenizer.eos_token_id
            ),
            "add_special_tokens": False, "packing": False,
            "checkpoint_selection": "validation_loss_within_run",
        },
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return trainer


def train_dual_adapters(splits: dict[str, pd.DataFrame], config: ExperimentConfig) -> dict[str, str]:
    """Train independent adapters sequentially, with each task's exact filtered alpha."""
    dual_config = replace(config, architecture="dual_adapter")
    spec = DATASET_REGISTRY[dual_config.dataset]
    saved = {}
    for task in ("sentiment", "topic"):
        task_rank = dual_config.sentiment_lora_rank if task == "sentiment" else dual_config.topic_lora_rank
        task_config = replace(
            dual_config, lora_rank=task_rank or dual_config.lora_rank,
            sentiment_lora_rank=dual_config.sentiment_lora_rank or dual_config.lora_rank,
            topic_lora_rank=dual_config.topic_lora_rank or dual_config.lora_rank,
        )
        model, tokenizer = load_model_and_tokenizer(task_config)
        train_dataset, validation_dataset = make_single_task_datasets(
            splits["train"], splits["validation"], spec, tokenizer, task_config.max_length, task,
        )
        trainer = make_trainer(
            model, tokenizer, train_dataset, validation_dataset,
            filtered_train_frame(train_dataset), task_config, task=task,
        )
        result = trainer.train()
        best_dir = Path(task_config.output_dir) / task / "best_model"
        trainer.model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        run_dir = best_dir.parent
        (run_dir / "training_metrics.json").write_text(json.dumps(result.metrics, default=str, indent=2) + "\n", encoding="utf-8")
        (run_dir / "trainer_log_history.json").write_text(json.dumps(trainer.state.log_history, default=str, indent=2) + "\n", encoding="utf-8")
        saved[task] = str(best_dir)
        del trainer, model, tokenizer, train_dataset, validation_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return saved
