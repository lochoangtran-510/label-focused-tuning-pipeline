"""Prepare the three datasets without redistributing their raw content."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .datasets import DATASET_REGISTRY, load_huggingface_splits, prepare_frame


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

    split_aliases = SPLIT_ALIASES.get(dataset_name)
    source = Path(raw_dir) / dataset_name
    has_local_copy = split_aliases is not None and all(
        any((source / name).exists() for name in names)
        for names in split_aliases.values()
    )
    if has_local_copy:
        source = Path(raw_dir) / dataset_name
        frames = {
            split: pd.read_csv(_find_split(source, names))
            for split, names in split_aliases.items()
        }
    else:
        frames = load_huggingface_splits(dataset_name)

    written = {}
    for split, frame in frames.items():
        prepared = prepare_frame(frame, spec)
        path = destination / f"{split}.csv"
        prepared.to_csv(path, index=False, encoding="utf-8")
        written[split] = path
    return written
