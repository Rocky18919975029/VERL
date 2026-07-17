# Copyright 2026 The verl team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.hpf_utils import (
    build_hpf_transition_prefix_plan,
    configure_hpf_transition_behavior_log_probs,
    estimate_hpf_transition_return,
)
from verl.workers.utils.losses import _hpf_transition_combine_policy_losses


def test_transition_prefix_plan_pairs_current_and_next_cut_requests():
    plan = build_hpf_transition_prefix_plan(
        [
            list(range(8)),
            list(range(5)),
            list(range(4)),
            list(range(2)),
        ],
        current_horizon=4,
        next_horizon=6,
        max_response_length=10,
    )

    assert plan.current_prefix_ids == [list(range(4)), list(range(4)), list(range(4)), list(range(2))]
    assert plan.next_prefix_ids == [list(range(6)), list(range(5)), list(range(4)), list(range(2))]
    np.testing.assert_array_equal(plan.current_needs_suffix, [True, True, False, False])
    np.testing.assert_array_equal(plan.next_needs_suffix, [True, False, False, False])
    # Requests are source-major so the two cuts from one sampled high-temp
    # prefix stay adjacent in the combined low-temperature batch.
    np.testing.assert_array_equal(plan.request_source_rows, [0, 0, 1])
    np.testing.assert_array_equal(plan.request_cut_indices, [0, 1, 0])


def test_transition_prefix_plan_omits_suffix_requests_at_final_cut():
    plan = build_hpf_transition_prefix_plan(
        [list(range(6))],
        current_horizon=6,
        next_horizon=6,
        max_response_length=6,
    )

    assert not plan.current_needs_suffix.any()
    assert not plan.next_needs_suffix.any()
    assert len(plan.request_source_rows) == 0


def test_transition_prefix_plan_rejects_reversed_cuts():
    with pytest.raises(ValueError, match="next_horizon must be at least current_horizon"):
        build_hpf_transition_prefix_plan(
            [[1, 2, 3]],
            current_horizon=4,
            next_horizon=2,
            max_response_length=8,
        )


def test_transition_return_estimate_aligns_paired_rollouts():
    current = DataProto.from_single_dict(
        {
            "dummy": torch.zeros(2, 1),
            "hpf_transition_pair_uid": np.array(["a", "b"], dtype=object),
        }
    )
    next_cut = DataProto.from_single_dict(
        {
            "dummy": torch.zeros(2, 1),
            "hpf_transition_pair_uid": np.array(["b", "a"], dtype=object),
        }
    )
    estimate = estimate_hpf_transition_return(
        current,
        next_cut,
        current_reward_tensor=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        next_reward_tensor=torch.tensor([[5.0, 0.0], [3.0, 0.0]]),
    )

    assert estimate.current_mean == pytest.approx(1.5)
    assert estimate.next_mean == pytest.approx(4.0)
    assert estimate.delta_mean == pytest.approx(2.5)


def test_transition_behavior_log_probs_reuse_vllm_values_without_recompute():
    current = DataProto.from_single_dict({"old_log_probs": torch.tensor([[1.0]])})
    next_cut = DataProto.from_single_dict({"old_log_probs": torch.tensor([[2.0]])})

    def fail_if_called(_batch):
        raise AssertionError("recompute_fn must not be called in rollout reuse mode")

    source = configure_hpf_transition_behavior_log_probs(
        current,
        next_cut,
        reuse_rollout_log_probs=True,
        recompute_fn=fail_if_called,
    )

    assert source == "vllm_tree_rollout"
    assert current.batch["old_log_probs"].item() == 1.0
    assert next_cut.batch["old_log_probs"].item() == 2.0


def test_transition_behavior_log_probs_recompute_both_mixed_temperature_cuts():
    current = DataProto.from_single_dict({"old_log_probs": torch.tensor([[1.0]])})
    next_cut = DataProto.from_single_dict({"old_log_probs": torch.tensor([[2.0]])})
    calls = []

    def recompute(batch):
        calls.append(batch)
        value = 10.0 + len(calls)
        return DataProto.from_single_dict({"old_log_probs": torch.tensor([[value]])})

    source = configure_hpf_transition_behavior_log_probs(
        current,
        next_cut,
        reuse_rollout_log_probs=False,
        recompute_fn=recompute,
    )

    assert source == "actor_mixed_temperature_forward"
    assert calls == [current, next_cut]
    assert current.batch["old_log_probs"].item() == 11.0
    assert next_cut.batch["old_log_probs"].item() == 12.0


def test_transition_policy_loss_uses_current_once_when_hinge_is_inactive():
    current = torch.tensor(0.2, requires_grad=True)
    next_cut = torch.tensor(0.1, requires_grad=True)
    loss, info = _hpf_transition_combine_policy_losses(
        current,
        next_cut,
        static_offset=0.3,
        lambda_trans=0.5,
        dp_size=1,
        dp_group=None,
    )
    loss.backward()

    assert info["active"].item() == 0
    assert info["objective"].item() == pytest.approx(0.2)
    assert current.grad.item() == pytest.approx(1.0)
    assert next_cut.grad.item() == pytest.approx(0.0)


def test_transition_policy_loss_adds_next_minus_current_gradient_when_active():
    current = torch.tensor(0.2, requires_grad=True)
    next_cut = torch.tensor(0.1, requires_grad=True)
    loss, info = _hpf_transition_combine_policy_losses(
        current,
        next_cut,
        static_offset=-0.3,
        lambda_trans=0.5,
        dp_size=1,
        dp_group=None,
    )
    loss.backward()

    assert info["active"].item() == 1
    assert info["penalty"].item() == pytest.approx(0.2)
    assert info["objective"].item() == pytest.approx(0.3)
    assert current.grad.item() == pytest.approx(0.5)
    assert next_cut.grad.item() == pytest.approx(0.5)
