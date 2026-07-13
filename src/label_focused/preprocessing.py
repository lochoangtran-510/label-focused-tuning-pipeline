"""Prepare the three datasets without redistributing their raw content."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .datasets import DATASET_REGISTRY, prepare_frame


SPLIT_ALIASES = {
    "neu_esc": {
        "train": ("train.csv", "train_set.csv"),
        "validation": ("validation.csv", "val.csv", "val_set.csv"),
        "test": ("test.csv", "test_set.csv"),
    },
    "victsd": {
        "train": ("train.csv", "ViCTSD_train.csv"),
        "validation": ("validation.csv", "valid.csv", "ViCTSD_valid.csv"),
        "test": ("test.csv", "ViCTSD_test.csv"),
    },
}


def _find_split(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of these files exist in {root}: {list(names)}")


def prepare_dataset(
    dataset_name: str,
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    spec = DATASET_REGISTRY[dataset_name]
    destination = Path(output_dir) / dataset_name
    destination.mkdir(parents=True, exist_ok=True)

    if dataset_name == "uit_vsfc":
        from datasets import load_dataset

        dataset = load_dataset("uitnlp/vietnamese_students_feedback")
        validation_key = "validation" if "validation" in dataset else "valid"
        frames = {
            "train": dataset["train"].to_pandas(),
            "validation": dataset[validation_key].to_pandas(),
            "test": dataset["test"].to_pandas(),
        }
    else:
        source = Path(raw_dir) / dataset_name
        frames = {
            split: pd.read_csv(_find_split(source, aliases))
            for split, aliases in SPLIT_ALIASES[dataset_name].items()
        }

    written = {}
    for split, frame in frames.items():
        prepared = prepare_frame(frame, spec)
        path = destination / f"{split}.csv"
        prepared.to_csv(path, index=False, encoding="utf-8")
        written[split] = path
    return written
