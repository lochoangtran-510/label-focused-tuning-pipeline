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

Colab quickstart for **NEU-ESC**, **UIT-VSFC**, and **ViCTSD**. The notebook is
only a thin interface to the same Python CLI used on Linux."""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Installation

Never paste an access token into a notebook. Authenticate with a Colab secret
only if the selected model requires it."""
    ),
    nbf.v4.new_code_cell(
        """!git clone https://github.com/lochoangtran-510/label-focused-tuning-pipeline.git
%cd label-focused-tuning-pipeline
%pip install -q -r requirements.txt"""
    ),
    nbf.v4.new_markdown_cell("## 2. Prepare one dataset"),
    nbf.v4.new_code_cell(
        """# UIT-VSFC downloads automatically. For NEU-ESC or ViCTSD, first
# place the official source CSVs under data/raw/<dataset>/.
DATASET = "uit_vsfc"  # neu_esc | uit_vsfc | victsd
!python scripts/prepare_data.py --dataset {DATASET}"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Smoke test

The override checks the pipeline only. Remove `--max-steps 2` for the public
three-epoch schedule."""
    ),
    nbf.v4.new_code_cell(
        """CONFIG = "configs/ablation/uit_vsfc_full_r8.yaml"
!python scripts/train.py --config {CONFIG} --max-steps 2 --output-root outputs/smoke"""
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

target = Path("notebooks/colab_quickstart.ipynb")
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, target)
print(target)
