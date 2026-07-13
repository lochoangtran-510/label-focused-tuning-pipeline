# Data preparation

Raw and processed datasets are intentionally excluded from version control.
The preparation command downloads all three datasets from Hugging Face and
preserves their official splits. Dataset revisions are pinned in
`src/label_focused/datasets.py` and `configs/datasets/` for reproducibility:

```bash
python scripts/prepare_data.py --dataset neu_esc
python scripts/prepare_data.py --dataset uit_vsfc
python scripts/prepare_data.py --dataset victsd
```

Sources:

- `hung20gg/NEU-ESC` (gated)
- `uitnlp/vietnamese_students_feedback`
- `tarudesu/ViCTSD`

Before preparing NEU-ESC, accept the conditions on its Hugging Face page and
authenticate with `hf auth login`. In Colab, keep `HF_TOKEN` in a private
secret; never paste it into the notebook or commit it.

Generated layout:

```text
data/
├── raw/                # optional offline fallback; never committed
└── processed/
    ├── neu_esc/{train,validation,test}.csv
    ├── uit_vsfc/{train,validation,test}.csv
    └── victsd/{train,validation,test}.csv
```

For offline use, existing NEU-ESC or ViCTSD source CSVs under
`data/raw/<dataset>/` are accepted as a fallback. When complete processed CSVs
already exist, training and evaluation use them without downloading again.

ViCTSD preprocessing concatenates the article title and comment with ` [SEP] `.
The preparation command preserves the official split; it does not randomly
resplit the datasets.

Original publications:

- UIT-VSFC: <https://doi.org/10.1109/KSE.2018.8573337>
- NEU-ESC: <https://doi.org/10.48550/arXiv.2506.23524>
- ViCTSD: <https://doi.org/10.1007/978-3-030-79457-6_49>
