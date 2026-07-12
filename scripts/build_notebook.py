"""Build the clean public notebook without carrying private Colab metadata."""

import json
from pathlib import Path


class _V4:
    @staticmethod
    def new_notebook():
        return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}

    @staticmethod
    def _cell(cell_type, source):
        cell = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(True)}
        if cell_type == "code":
            cell.update({"execution_count": None, "outputs": []})
        return cell

    @classmethod
    def new_markdown_cell(cls, source):
        return cls._cell("markdown", source)

    @classmethod
    def new_code_cell(cls, source):
        return cls._cell("code", source)


class _NotebookBuilder:
    v4 = _V4()

    @staticmethod
    def write(notebook, target):
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


nbf = _NotebookBuilder()


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Label-Focused Tuning Pipeline

Reproducible joint multitask fine-tuning for **NEU-ESC**, **UIT-VSFC**, and **ViCTSD**.
The training pipeline is shared; dataset loading, labels, preprocessing, and prompts are selected through one configuration value."""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Installation

Run this notebook from the repository root. On Google Colab, clone the repository first. Never paste an access token into a notebook; authenticate with `huggingface-cli login` or a Colab secret only if the selected model requires it."""
    ),
    nbf.v4.new_code_cell("%pip install -q -r requirements.txt"),
    nbf.v4.new_markdown_cell("## 2. Experiment configuration"),
    nbf.v4.new_code_cell(
        """import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from label_focused.datasets import DATASET_REGISTRY, load_splits
from label_focused.training import (
    ExperimentConfig,
    load_model_and_tokenizer,
    make_prompt_datasets,
    make_trainer,
    train_dual_adapters,
)

# Change only this value to run another dataset:
DATASET_NAME = "neu_esc"  # "neu_esc", "uit_vsfc", or "victsd"
ARCHITECTURE = "joint"    # "joint" or "dual_adapter"

config = ExperimentConfig(
    dataset=DATASET_NAME,
    architecture=ARCHITECTURE,
    lora_rank=32,
    # For asymmetric Dual Adapter runs, set both fields explicitly, e.g. 32/8:
    sentiment_lora_rank=None,
    topic_lora_rank=None,
    use_masking=True,
    use_focal_loss=True,
)
spec = DATASET_REGISTRY[config.dataset]
print(config)
print(spec.system_prompt)
print("\\n" + spec.one_shot)"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Load data

- `UIT-VSFC` is downloaded from Hugging Face automatically.
- Place NEU-ESC files in `data/neu_esc/{train,validation,test}.csv`.
- Place ViCTSD files in `data/victsd/{train,validation,test}.csv`.

The raw datasets are not redistributed by this repository; obtain them from their original sources."""
    ),
    nbf.v4.new_code_cell(
        """splits = load_splits(config.dataset, data_dir=ROOT / "data")
for split, frame in splits.items():
    print(f"{split:>10}: {len(frame):,} samples; columns={list(frame.columns)}")"""
    ),
    nbf.v4.new_markdown_cell("## 4. Train Joint or Dual Adapter architecture"),
    nbf.v4.new_code_cell(
        """if config.architecture == "joint":
    model, tokenizer = load_model_and_tokenizer(config)
    train_dataset, validation_dataset = make_prompt_datasets(
        splits["train"], splits["validation"], spec, tokenizer, config.max_length
    )
    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        train_frame=splits["train"],
        config=config,
    )
    trainer.train()
    best_dir = Path(config.output_dir) / "best_model"
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Saved Joint adapter to {best_dir}")
elif config.architecture == "dual_adapter":
    adapter_paths = train_dual_adapters(splits, config)
    print(adapter_paths)
else:
    raise ValueError(f"Unknown architecture: {config.architecture}")"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Experiments reported in the paper

The code is reusable, but the paper did not run every combination on every dataset.
For the reported ablation, use exactly:

| Configuration | `use_masking` | `use_focal_loss` |
|---|---:|---:|
| Base | `False` | `False` |
| Masking only | `True` | `False` |
| Full pipeline | `True` | `True` |

Dataset/rank scope:

- Ablation: NEU-ESC at r=32; UIT-VSFC and ViCTSD at r=8.
- Joint rank sensitivity (r=8, 16, 32, 64): NEU-ESC and UIT-VSFC only.
- Dual rank sensitivity: NEU-ESC only, using 8/8, 16/16, 32/32, 64/64, 16/64, 64/16, and 32/8.
- Inference efficiency: UIT-VSFC only.
- Detailed error analysis: NEU-ESC only."""
    ),
]

target = Path("notebooks/label_focused_pipeline.ipynb")
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, target)
print(target)
