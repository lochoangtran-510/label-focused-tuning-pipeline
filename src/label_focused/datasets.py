"""Dataset-specific loading, labels, preprocessing, and prompt metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    text_column: str
    sentiment_column: str
    topic_column: str
    sentiment_map: dict[str, str]
    topic_map: dict[str, str]
    sentiment_labels: tuple[str, ...]
    topic_labels: tuple[str, ...]
    input_label: str
    one_shot_text: str
    one_shot_sentiment: str
    one_shot_topic: str
    preprocess: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    system_prompt_override: str | None = None

    @property
    def system_prompt(self) -> str:
        if self.system_prompt_override:
            return self.system_prompt_override
        sentiments = ", ".join(self.sentiment_labels)
        topics = ", ".join(self.topic_labels)
        return (
            "Thực hiện đồng thời hai nhiệm vụ:\n"
            f"(1) Cảm xúc (Sentiment): {sentiments}.\n"
            f"(2) Chủ đề (Topic): {topics}.\n"
            "Chỉ trả về JSON với đúng hai khóa: Sentiment và Topic. "
            "Không giải thích thêm."
        )

    @property
    def one_shot(self) -> str:
        return (
            f'Ví dụ:\n{self.input_label}: "{self.one_shot_text}"\n'
            f'Output: {{"Sentiment": "{self.one_shot_sentiment}", '
            f'"Topic": "{self.one_shot_topic}"}}'
        )


def _numeric_map(labels: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, label in enumerate(labels):
        result[str(index)] = label
        result[f"{index}.0"] = label
    return result


def _prepare_victsd(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Title", "Comment", "Toxicity", "Topic"}
    _require_columns(frame, required, "ViCTSD")
    result = frame.copy()
    title = result["Title"].fillna("").astype(str).str.strip()
    comment = result["Comment"].fillna("").astype(str).str.strip()
    result["Text"] = [
        f"{t} [SEP] {c}" if t else c for t, c in zip(title, comment)
    ]
    return result


NEU_SENTIMENTS = ("Neutral", "Positive", "Negative", "Toxic")
NEU_TOPICS = (
    "Spam", "News", "Academic", "Other", "Service",
    "Jobs & Recruitment", "Personal Affairs", "Social Affairs",
    "Help & Share", "Club & Events",
)
UIT_SENTIMENTS = ("Negative", "Neutral", "Positive")
UIT_TOPICS = ("Lecturer", "Curriculum", "Facility", "Others")
VICTSD_SENTIMENTS = ("Non-toxic", "Toxic")
VICTSD_TOPICS = (
    "KinhDoanh", "OtoXemay", "TheGioi", "TheThao", "KhoaHoc",
    "ThoiSu", "PhapLuat", "GiaoDuc", "SucKhoe", "GiaiTri",
)

DATASET_REGISTRY = {
    "neu_esc": DatasetSpec(
        name="NEU-ESC", text_column="text", sentiment_column="sentiment",
        topic_column="classification", sentiment_map=_numeric_map(NEU_SENTIMENTS),
        topic_map=_numeric_map(NEU_TOPICS), sentiment_labels=NEU_SENTIMENTS,
        topic_labels=NEU_TOPICS, input_label="Phản hồi",
        one_shot_text=("bạn ơi ngành này có phù hợp với tuổi trung niên đến già không "
                       "bạn hay đến độ tuổi đó mức độ đào thải sẽ cao bạn nhỉ ."),
        one_shot_sentiment="Neutral", one_shot_topic="Academic",
    ),
    "uit_vsfc": DatasetSpec(
        name="UIT-VSFC", text_column="sentence", sentiment_column="sentiment",
        topic_column="topic", sentiment_map=_numeric_map(UIT_SENTIMENTS),
        topic_map=_numeric_map(UIT_TOPICS), sentiment_labels=UIT_SENTIMENTS,
        topic_labels=UIT_TOPICS, input_label="Phản hồi",
        one_shot_text=("giảng viên giảng dạy nhiệt tình , có đầy đủ các hoạt động "
                       "để sinh viên rèn luyện và học tập ."),
        one_shot_sentiment="Positive", one_shot_topic="Lecturer",
    ),
    "victsd": DatasetSpec(
        name="ViCTSD", text_column="Text", sentiment_column="Toxicity",
        topic_column="Topic", sentiment_map=_numeric_map(VICTSD_SENTIMENTS),
        topic_map={label: label for label in VICTSD_TOPICS},
        sentiment_labels=VICTSD_SENTIMENTS, topic_labels=VICTSD_TOPICS,
        input_label="Bình luận",
        one_shot_text=("Biden dẫn Trump 13 điểm [SEP] Cuộc bầu cử ở Mỹ là "
                       "công khai và công bằng nhất."),
        one_shot_sentiment="Non-toxic", one_shot_topic="TheGioi",
        preprocess=_prepare_victsd,
        system_prompt_override=(
            "Thực hiện đồng thời hai nhiệm vụ:\n"
            "(1) Sentiment: Non-toxic hoặc Toxic.\n"
            "(2) Topic: KinhDoanh, OtoXemay, TheGioi, TheThao, KhoaHoc, "
            "ThoiSu, PhapLuat, GiaoDuc, SucKhoe, GiaiTri.\n"
            "Chỉ trả về JSON với đúng hai khóa: Sentiment và Topic. "
            "Không giải thích thêm."
        ),
    ),
}


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def prepare_frame(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    result = spec.preprocess(frame) if spec.preprocess else frame.copy()
    _require_columns(
        result,
        {spec.text_column, spec.sentiment_column, spec.topic_column},
        spec.name,
    )
    return result


def load_splits(
    dataset_name: str,
    data_dir: str | Path = "data",
) -> dict[str, pd.DataFrame]:
    """Load train/validation/test frames using one stable public interface."""
    spec = DATASET_REGISTRY[dataset_name]
    if dataset_name == "uit_vsfc":
        from datasets import load_dataset

        dataset = load_dataset("uitnlp/vietnamese_students_feedback")
        validation_key = "validation" if "validation" in dataset else "valid"
        raw = {
            "train": dataset["train"].to_pandas(),
            "validation": dataset[validation_key].to_pandas(),
            "test": dataset["test"].to_pandas(),
        }
    else:
        root = Path(data_dir) / dataset_name
        raw = {
            split: pd.read_csv(root / f"{split}.csv")
            for split in ("train", "validation", "test")
        }
    return {key: prepare_frame(value, spec) for key, value in raw.items()}
