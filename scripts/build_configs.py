#!/usr/bin/env python3
"""Generate historical paper presets and explicitly opt-in revised runs.

The original matrix retains its ranks and three-epoch schedule. Additional
experiments use a separate protocol/path and are never swept by default.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"

COMMON = {
    "model_id": "Viet-Mistral/Vistral-7B-Chat",
    "output_root": "outputs",
    "lora_alpha": None,
    "lora_dropout": 0.05,
    "learning_rate": 0.0001,
    "epochs": 3,
    "max_steps": -1,
    "batch_size": 6,
    "gradient_accumulation": 4,
    "max_length": 1024,
    "focal_gamma": 2.0,
    "alpha_scope": "terminal_only",
    "protocol": "paper",
    "seed": 42,
}


def write(relative: str, **values) -> None:
    path = CONFIG_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"experiment": {**COMMON, **values}}
    lines = ["experiment:"]
    for key, value in document["experiment"].items():
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, str):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        lines.append(f"  {key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ablation_ranks = {"neu_esc": 32, "uit_vsfc": 8, "victsd": 8}
    methods = {
        "base": (False, False),
        "masking": (True, False),
        "full": (True, True),
    }
    for dataset, rank in ablation_ranks.items():
        for name, (masking, focal) in methods.items():
            write(
                f"ablation/{dataset}_{name}_r{rank}.yaml",
                dataset=dataset,
                architecture="joint",
                lora_rank=rank,
                use_masking=masking,
                use_focal_loss=focal,
                use_class_alpha=focal,
            )

    for dataset in ("neu_esc", "uit_vsfc"):
        for rank in (8, 16, 32, 64):
            for name, masking, focal in (
                ("base", False, False), ("full", True, True)
            ):
                write(
                    f"joint_rank/{dataset}_{name}_r{rank}.yaml",
                    dataset=dataset,
                    architecture="joint",
                    lora_rank=rank,
                    use_masking=masking,
                    use_focal_loss=focal,
                    use_class_alpha=focal,
                )

    for sentiment_rank, topic_rank in (
        (8, 8), (16, 16), (32, 32), (64, 64), (16, 64), (64, 16), (32, 8)
    ):
        write(
            f"dual_rank/neu_esc_full_r{sentiment_rank}_r{topic_rank}.yaml",
            dataset="neu_esc",
            architecture="dual_adapter",
            lora_rank=sentiment_rank,
            sentiment_lora_rank=sentiment_rank,
            topic_lora_rank=topic_rank,
            use_masking=True,
            use_focal_loss=True,
            use_class_alpha=True,
        )

    # Corrected notebook protocols are separate from the paper's preset matrix.
    # In particular, recent NEU notebooks use r=8 / len=2024 / max_steps=2800;
    # the historical NEU presets above remain len=1024 / three epochs.
    for dataset in ("neu_esc", "uit_vsfc", "victsd"):
        max_length = 2024 if dataset == "neu_esc" else 1024
        revised = {
            "dataset": dataset,
            "architecture": "joint",
            "lora_rank": 8,
            "protocol": "revision",
            "max_length": max_length,
            "max_steps": 2800 if dataset == "neu_esc" else -1,
            "use_masking": True,
        }
        for scope in ("terminal_only", "all_label_tokens"):
            write(
                f"additional/{dataset}_mask_alpha_ce_{scope}_r8_len{max_length}.yaml",
                **revised,
                use_focal_loss=False,
                focal_gamma=0.0,
                use_class_alpha=True,
                alpha_scope=scope,
            )
        write(
            f"additional/{dataset}_mask_focal_no_alpha_r8_len{max_length}.yaml",
            **revised,
            use_focal_loss=True,
            use_class_alpha=False,
        )
    write(
        "additional/victsd_base_ce_r8_len1024.yaml",
        dataset="victsd",
        architecture="joint",
        lora_rank=8,
        protocol="revision",
        use_masking=False,
        use_focal_loss=False,
        focal_gamma=0.0,
        use_class_alpha=False,
    )

    print(f"Generated {len(list(CONFIG_ROOT.rglob('*.yaml')))} presets")


if __name__ == "__main__":
    main()
