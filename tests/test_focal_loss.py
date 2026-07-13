import torch

from label_focused.losses import label_focused_loss


def test_focal_loss_ignores_masked_prompt_tokens():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 7)
    labels = torch.tensor([[-100, -100, 2, 3]])
    original = label_focused_loss(logits, labels, gamma=2.0, token_weights={3: 2.0})
    changed = logits.clone()
    changed[:, 0, :] = 1000  # predicts the masked region only
    assert torch.allclose(
        original,
        label_focused_loss(changed, labels, gamma=2.0, token_weights={3: 2.0}),
    )
