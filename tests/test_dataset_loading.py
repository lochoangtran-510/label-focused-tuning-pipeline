import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from label_focused.datasets import (
    DATASET_REGISTRY,
    load_huggingface_splits,
    prepare_frame,
)


class _FakeSplit:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame.copy()


def test_neu_huggingface_columns_are_normalized_case_insensitively():
    frame = pd.DataFrame(
        {"Text": ["x"], "Sentiment": [0], "Classification": [2]}
    )
    prepared = prepare_frame(frame, DATASET_REGISTRY["neu_esc"])
    assert {"text", "sentiment", "classification"}.issubset(prepared.columns)


def test_neu_loader_pins_official_files_and_normalizes_splits(monkeypatch):
    calls = []
    frame = pd.DataFrame(
        {"Text": ["x"], "Sentiment": [0], "Classification": [2]}
    )

    def fake_load_dataset(dataset_id, **kwargs):
        calls.append((dataset_id, kwargs))
        return {
            "train": _FakeSplit(frame),
            "validation": _FakeSplit(frame),
            "test": _FakeSplit(frame),
        }

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )
    splits = load_huggingface_splits("neu_esc")

    assert calls == [
        (
            "hung20gg/NEU-ESC",
            {
                "revision": "daf543ad1992153cd2be9fec3cb59aa0fc714147",
                "data_files": {
                    "train": "train_set.csv",
                    "validation": "val_set.csv",
                    "test": "test_set.csv",
                }
            },
        )
    ]
    assert set(splits) == {"train", "validation", "test"}
    assert splits["train"].columns.tolist() == [
        "text", "sentiment", "classification"
    ]


def test_neu_gated_error_has_authentication_instructions(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        raise PermissionError("gated")

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )
    with pytest.raises(RuntimeError, match="hf auth login"):
        load_huggingface_splits("neu_esc")
