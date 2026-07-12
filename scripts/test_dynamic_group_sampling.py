#!/usr/bin/env python3
"""Unit tests for outcome-only DAPO group filtering."""

import numpy as np
import pytest

from verl.trainer.ppo.ray_trainer import mixed_group_selection


def test_only_mixed_groups_are_selected():
    uids = np.repeat(np.array(["fail", "mixed", "success"], dtype=object), 8)
    outcomes = np.array([0] * 8 + [0, 0, 0, 0, 1, 1, 1, 1] + [1] * 8)
    indices, states = mixed_group_selection(uids, outcomes, group_size=8)
    assert states == {"fail": "all_fail", "mixed": "mixed", "success": "all_success"}
    assert indices.tolist() == list(range(8, 16))


def test_mixed_group_cap_keeps_complete_groups():
    uids = np.repeat(np.array(["a", "b", "c"], dtype=object), 8)
    outcomes = np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1]), 3)
    indices, _states = mixed_group_selection(uids, outcomes, group_size=8, max_groups=2)
    assert indices.tolist() == list(range(16))
    assert set(uids[indices]) == {"a", "b"}


def test_malformed_group_is_rejected():
    with pytest.raises(ValueError, match="expected 8"):
        mixed_group_selection(["a"] * 7, [0] * 7, group_size=8)
