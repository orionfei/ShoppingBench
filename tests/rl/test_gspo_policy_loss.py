from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_gspo, get_policy_loss_fn


def _config(low=3e-4, high=4e-4):
    return SimpleNamespace(clip_ratio=low, clip_ratio_low=low, clip_ratio_high=high)


def test_gspo_is_registered_and_identity_ratio_has_sequence_balanced_gradient():
    old_log_prob = torch.zeros(2, 4)
    log_prob = torch.zeros(2, 4, requires_grad=True)
    advantages = torch.tensor([[1.0] * 4, [-1.0] * 4])
    response_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0] * 4])

    loss, clipfrac, ppo_kl, lower_clipfrac = compute_policy_loss_gspo(
        old_log_prob, log_prob, advantages, response_mask, config=_config()
    )
    loss.backward()

    assert get_policy_loss_fn("gspo") is compute_policy_loss_gspo
    assert loss.item() == pytest.approx(0.0)
    assert clipfrac.item() == pytest.approx(0.0)
    assert ppo_kl.item() == pytest.approx(0.0)
    assert lower_clipfrac.item() == pytest.approx(0.0)
    # Each response contributes equal total gradient even though lengths differ.
    assert log_prob.grad[0].sum().item() == pytest.approx(-0.5)
    assert log_prob.grad[1].sum().item() == pytest.approx(0.5)
    assert log_prob.grad[0, 2:].abs().sum().item() == pytest.approx(0.0)


def test_gspo_uses_geometric_mean_ratio_and_sequence_level_clipping():
    old_log_prob = torch.zeros(2, 3)
    # The first sequence is above the positive-advantage upper clip. The
    # second is below the negative-advantage lower clip.
    log_prob = torch.tensor(
        [[5e-4, 5e-4, 0.0], [-5e-4, -5e-4, -5e-4]], requires_grad=True
    )
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    loss, clipfrac, _, _ = compute_policy_loss_gspo(
        old_log_prob, log_prob, advantages, response_mask, config=_config()
    )
    loss.backward()

    expected = (-1.0004 + 0.9997) / 2
    assert loss.item() == pytest.approx(expected, abs=1e-7)
    assert clipfrac.item() == pytest.approx(1.0)
    assert log_prob.grad.abs().sum().item() == pytest.approx(0.0)


def test_gspo_ignores_padding_when_computing_sequence_ratio():
    old_log_prob = torch.zeros(1, 4)
    log_prob = torch.tensor([[2e-4, 4e-4, 10.0, -10.0]], requires_grad=True)
    advantages = torch.ones(1, 4)
    response_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    loss, clipfrac, _, _ = compute_policy_loss_gspo(
        old_log_prob, log_prob, advantages, response_mask, config=_config()
    )
    loss.backward()

    assert loss.item() == pytest.approx(-torch.exp(torch.tensor(3e-4)).item())
    assert clipfrac.item() == pytest.approx(0.0)
    assert log_prob.grad[0, 2:].abs().sum().item() == pytest.approx(0.0)
