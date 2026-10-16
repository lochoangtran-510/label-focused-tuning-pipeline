import math

import pytest
import torch
import torch.nn.functional as functional

from label_focused.losses import label_focused_loss


def test_default_and_gamma_zero_match_standard_causal_ce():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 5, 7, generator=generator)
    labels = torch.tensor([[-100, -100, 2, 3, 4], [0, 1, 5, -100, -100]])
    expected = functional.cross_entropy(
        logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=-100
    )
    torch.testing.assert_close(label_focused_loss(logits, labels, gamma=0), expected)
    torch.testing.assert_close(
        label_focused_loss(logits, labels, gamma=0, alpha_weights=torch.ones_like(labels)),
        expected,
    )


@pytest.mark.parametrize("gamma", [0.0, 2.0])
def test_position_alpha_is_shifted_and_does_not_change_the_mean_denominator(gamma):
    logits = torch.linspace(-1, 1, 2 * 5 * 7).reshape(2, 5, 7).requires_grad_()
    labels = torch.tensor([[-100, -100, 2, 3, 4], [-100, 2, 3, -100, -100]])
    alpha = torch.ones_like(labels, dtype=torch.float32)
    # The same target ID has different alpha at different task-aware positions.
    alpha[0, 2] = 3.0
    alpha[1, 1] = 0.5
    terms = []
    for row in range(labels.size(0)):
        for target_position in range(1, labels.size(1)):
            target = int(labels[row, target_position])
            if target == -100:
                continue
            ce = -functional.log_softmax(logits[row, target_position - 1], dim=-1)[target]
            terms.append(alpha[row, target_position] * (1 - torch.exp(-ce)).pow(gamma) * ce)
    actual = label_focused_loss(logits, labels, gamma, alpha)
    torch.testing.assert_close(actual, torch.stack(terms).mean())
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0].count_nonzero() == 0  # predicts an ignored target
    assert logits.grad[:, -1].count_nonzero() == 0  # discarded by causal shift
    assert logits.grad[1, 2:].count_nonzero() == 0  # predicts padding


def test_focal_probability_comes_from_ce_before_alpha():
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    labels = torch.tensor([[-100, 1]])
    alpha = torch.tensor([[1.0, 3.0]])
    expected = torch.tensor(3.0 * (1 - 0.5) ** 2 * math.log(2))
    torch.testing.assert_close(label_focused_loss(logits, labels, 2, alpha), expected)


def test_ignored_logits_do_not_affect_loss_but_full_sequence_prompt_does():
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(1, 5, 7, generator=generator)
    masked = torch.tensor([[-100, -100, 2, 3, 4]])
    changed = logits.clone()
    changed[0, 0, 1] += 20
    torch.testing.assert_close(
        label_focused_loss(logits, masked, 2), label_focused_loss(changed, masked, 2)
    )
    full_sequence = torch.tensor([[0, 1, 2, 3, 4]])
    assert not torch.isclose(
        label_focused_loss(logits, full_sequence, 0),
        label_focused_loss(changed, full_sequence, 0),
    )


@pytest.mark.parametrize("gamma", [-1, float("nan"), float("inf")])
def test_invalid_gamma_fails(gamma):
    with pytest.raises(ValueError, match="gamma"):
        label_focused_loss(torch.zeros(1, 2, 3), torch.tensor([[0, 1]]), gamma)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_alpha_fails(weight):
    alpha = torch.tensor([[1.0, weight]])
    with pytest.raises(ValueError, match="strictly positive"):
        label_focused_loss(torch.zeros(1, 2, 3), torch.tensor([[0, 1]]), 0, alpha)


def test_global_token_dictionary_is_rejected_and_empty_targets_fail():
    logits, labels = torch.zeros(1, 2, 3), torch.tensor([[-100, 1]])
    with pytest.raises(TypeError, match="global token-ID"):
        label_focused_loss(logits, labels, 2, {1: 3.0})
    with pytest.raises(ValueError, match="same shape"):
        label_focused_loss(logits, labels, 0, torch.ones(1, 1))
    with pytest.raises(ValueError, match="No non-ignored"):
        label_focused_loss(logits, torch.full_like(labels, -100), 0)
