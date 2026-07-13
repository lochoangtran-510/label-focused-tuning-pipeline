# Data preparation

Raw datasets are intentionally excluded from version control. Obtain each dataset
from its original source and preserve its license.

Place source files under `data/raw/`, then create the processed splits:

```bash
python scripts/prepare_data.py --dataset neu_esc
python scripts/prepare_data.py --dataset uit_vsfc
python scripts/prepare_data.py --dataset victsd
```

Expected layout:

```text
data/
├── raw/
│   ├── neu_esc/        # train_set.csv, val_set.csv, test_set.csv accepted
│   └── victsd/         # ViCTSD_train/valid/test.csv accepted
└── processed/
    ├── neu_esc/{train,validation,test}.csv
    ├── uit_vsfc/{train,validation,test}.csv
    └── victsd/{train,validation,test}.csv
```

UIT-VSFC is loaded directly from
`uitnlp/vietnamese_students_feedback` through Hugging Face Datasets.

ViCTSD preprocessing concatenates the article title and comment with ` [SEP] `.
The preparation command preserves the official split; it does not randomly
resplit the datasets.

Original publications:

- UIT-VSFC: <https://doi.org/10.1109/KSE.2018.8573337>
- NEU-ESC: <https://doi.org/10.48550/arXiv.2506.23524>
- ViCTSD: <https://doi.org/10.1007/978-3-030-79457-6_49>
