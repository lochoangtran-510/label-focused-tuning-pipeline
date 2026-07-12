# Data preparation

Raw datasets are intentionally excluded from version control. Obtain each dataset
from its original source and preserve its license.

Expected local layout:

```text
data/
├── neu_esc/
│   ├── train.csv       # text, sentiment, classification
│   ├── validation.csv
│   └── test.csv
└── victsd/
    ├── train.csv       # Title, Comment, Toxicity, Topic
    ├── validation.csv
    └── test.csv
```

UIT-VSFC is loaded directly from
`uitnlp/vietnamese_students_feedback` through Hugging Face Datasets.
