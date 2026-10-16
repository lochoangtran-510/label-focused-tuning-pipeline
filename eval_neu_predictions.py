#!/usr/bin/env python3
"""Backward-compatible entry point for full-test evaluation.

The original Colab-specific implementation has moved to
``scripts/evaluate.py``. Invalid task predictions remain ``PARSE_ERROR`` and
count as wrong; no labels are imputed and no test samples are removed.

Example:
    python eval_neu_predictions.py \
      --dataset neu_esc \
      --predictions outputs/neu_esc/experiment/predictions.csv
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import main


if __name__ == "__main__":
    main()
