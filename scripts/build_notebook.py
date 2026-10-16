"""Build the clean public quickstart with deterministic notebook cell IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(cell_type: str, source: str) -> dict:
    result = {
        "cell_type": cell_type,
        "id": hashlib.sha256(f"{cell_type}\n{source}".encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(True),
    }
    if cell_type == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "cells": [
            cell("markdown", """# Label-Focused Tuning Pipeline

Colab quickstart for **NEU-ESC**, **UIT-VSFC**, and **ViCTSD**. This is a thin
interface to the Linux CLI. Corrected task-aware alpha assignment defaults to
the terminal token; weighting every label token is a separate opt-in experiment.
Historical paper results are archived results, not metrics reproduced by this update."""),
            cell("markdown", """## 1. Installation

Use a GPU runtime for training. Training dependencies pin Transformers 4.46.3
and Datasets 2.21.0, matching the corrected standalone training notebooks.
Inference uses its separate environment in `requirements-inference.txt`.

NEU-ESC is gated: accept the Hugging Face dataset conditions and authenticate
privately before preparing it. Never paste an access token into saved notebook cells."""),
            cell("code", """!git clone https://github.com/lochoangtran-510/label-focused-tuning-pipeline.git
%cd label-focused-tuning-pipeline
%pip install -q -r requirements.txt"""),
            cell("markdown", "## 2. Prepare one dataset"),
            cell("code", """# Choose one dataset. NEU-ESC requires prior access approval.
DATASET = "uit_vsfc"  # neu_esc | uit_vsfc | victsd
if DATASET == "neu_esc":
    from google.colab import userdata
    from huggingface_hub import login
    login(token=userdata.get("HF_TOKEN"), add_to_git_credential=False)
!python scripts/prepare_data.py --dataset {DATASET}"""),
            cell("markdown", """## 3. Choose one corrected configuration and smoke test

`use_focal_loss` and `use_class_alpha` are independent. Set `alpha_scope` to
`terminal_only` or explicitly opt into `all_label_tokens`. JSON formatting tokens
retain weight 1; completion masking excludes prompt loss.

| Method | `use_masking` | `use_focal_loss` | `use_class_alpha` |
|---|---:|---:|---:|
| Base CE | false | false | false |
| Masked CE | true | false | false |
| Masked CE + alpha | true | false | true |
| Masked focal without alpha | true | true | false |
| Full pipeline | true | true | true |

The configurations below are additional runs, outside the historical paper matrix.
Recent NEU runs use r=8, length=2024 and max_steps=2200; UIT-VSFC and ViCTSD
use r=8, length=1024 and three epochs. All use seed 42. The two-step override
checks execution only and cannot supply research results."""),
            cell("code", """MAX_LENGTH = 2024 if DATASET == "neu_esc" else 1024
ALPHA_SCOPE = "terminal_only"  # opt-in alternative: "all_label_tokens"
CONFIG = (
    f"configs/additional/{DATASET}_mask_alpha_ce_"
    f"{ALPHA_SCOPE}_r8_len{MAX_LENGTH}.yaml"
)
# Alternative: masked focal WITHOUT class alpha
# CONFIG = f"configs/additional/{DATASET}_mask_focal_no_alpha_r8_len{MAX_LENGTH}.yaml"
# ViCTSD base CE: no completion masking, alpha, or focal modulation
# CONFIG = "configs/additional/victsd_base_ce_r8_len1024.yaml"

import yaml
from pathlib import Path
settings = yaml.safe_load(Path(CONFIG).read_text())["experiment"]
print({key: settings[key] for key in (
    "dataset", "protocol", "use_masking", "use_focal_loss", "use_class_alpha",
    "alpha_scope", "lora_rank", "max_length", "epochs", "max_steps", "seed"
)})
!python scripts/train.py --config {CONFIG} --max-steps 2 --output-root outputs/smoke"""),
            cell("markdown", """## 4. Optionally train the selected configuration

Enable one full run only after the smoke test passes. Use a new output root to
separate smoke tests, independent runs, and archived results. This notebook does
not start a rank sweep or all-label experiment matrix automatically."""),
            cell("code", """RUN_FULL_TRAINING = False
if RUN_FULL_TRAINING:
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "scripts/train.py", "--config", CONFIG,
        "--output-root", "outputs/revised_runs",
    ], check=True)"""),
            cell("markdown", """## 5. Historical paper preset scope

The original `configs/ablation`, `configs/joint_rank`, and `configs/dual_rank`
files preserve the paper ranks and three-epoch schedules. Their full pipeline
explicitly enables focal modulation AND class alpha at terminal label positions.
The current implementation uses corrected task-aware position assignment;
running a historical preset does not establish that archived scores are reproduced.

- Ablation: NEU-ESC at r=32; UIT-VSFC and ViCTSD at r=8.
- Joint rank sensitivity (r=8, 16, 32, 64): NEU-ESC and UIT-VSFC only.
- Dual rank sensitivity: NEU-ESC only, using 8/8, 16/16, 32/32, 64/64, 16/64, 64/16, and 32/8.
- Inference efficiency: UIT-VSFC only. Detailed error analysis: NEU-ESC only.

Additional CE + alpha and all-label-token runs live in `configs/additional`;
they are not included in the paper matrix runners by default."""),
        ],
    }


def main() -> None:
    target = ROOT / "notebooks" / "colab_quickstart.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
