"""Causal CE/focal loss with alpha aligned to target positions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def label_focused_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    alpha_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean weighted focal CE over non-ignored shifted target tokens.

    ``alpha_weights`` has the same shape as ``labels`` and is shifted with it.
    It is supplied by the field-aware collator, never looked up by token ID.
    With ``gamma=0`` and no alpha, this is ordinary causal cross-entropy.
    The denominator is the number of active targets, not the sum of alpha.
    """
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [batch, tokens, vocab] and matching labels")
    if alpha_weights is not None:
        if not isinstance(alpha_weights, torch.Tensor):
            raise TypeError(
                "alpha_weights must be a tensor aligned to labels; "
                "global token-ID weight dictionaries are not supported"
            )
        if alpha_weights.shape != labels.shape:
            raise ValueError("alpha_weights must have the same shape as labels")
        if not torch.isfinite(alpha_weights).all().item() or not alpha_weights.gt(0).all().item():
            raise ValueError("alpha_weights must be finite and strictly positive")

    shifted_labels = labels[:, 1:].to(logits.device)
    active = shifted_labels.ne(-100)
    # Select before casting to avoid a full-sequence float32 logits copy.
    active_logits = logits[:, :-1, :][active].float()
    active_labels = shifted_labels[active]
    if active_labels.numel() == 0:
        raise ValueError("No non-ignored causal targets are available for loss")

    cross_entropy = functional.cross_entropy(
        active_logits, active_labels, reduction="none"
    )
    if alpha_weights is None:
        alpha = torch.ones_like(cross_entropy)
    else:
        alpha = alpha_weights[:, 1:].to(
            device=logits.device, dtype=cross_entropy.dtype
        )[active]
    if gamma == 0:
        return (alpha * cross_entropy).mean()
    # p_t must come from the unweighted CE, before applying class alpha.
    probability = torch.exp(-cross_entropy)
    return (alpha * (1.0 - probability).pow(gamma) * cross_entropy).mean()
