"""Token-level alpha-balanced focal loss after completion masking."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as functional


def label_focused_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    token_weights: dict[int, float],
) -> torch.Tensor:
    """Compute Eq. (3–5) only where completion labels are not ``-100``."""
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    active = shifted_labels.ne(-100)
    active_logits = shifted_logits[active]
    active_labels = shifted_labels[active]
    if active_labels.numel() == 0:
        raise RuntimeError("No completion tokens available for focal loss")

    cross_entropy = functional.cross_entropy(
        active_logits, active_labels, reduction="none"
    )
    probability = torch.exp(-cross_entropy)
    alpha = torch.ones_like(cross_entropy)
    for token_id, weight in token_weights.items():
        alpha[active_labels == token_id] = weight
    focal_modulation = (1.0 - probability) ** gamma
    return (alpha * focal_modulation * cross_entropy).mean()


def flatten_label_token_weights(
    label_tokens: dict[str, Iterable[int]], class_weights: dict[str, float]
) -> dict[int, float]:
    result: dict[int, float] = {}
    for label, token_ids in label_tokens.items():
        for token_id in token_ids:
            result[int(token_id)] = float(class_weights.get(label, 1.0))
    return result
