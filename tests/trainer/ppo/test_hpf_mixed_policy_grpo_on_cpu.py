# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.hpf_utils import build_hpf_mixed_policy_grpo_batch
from verl.utils.torch_functional import temperature_policy_kl_from_logits


def test_mixed_policy_grpo_uses_prefix_and_one_horizon_suffix_window():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    advantages = torch.tensor([[2.0] * 6, [-1.0] * 6])
    batch = DataProto.from_single_dict(
        {
            "response_mask": response_mask,
            "advantages": advantages,
            "returns": advantages.clone(),
        }
    )
    leader_log_probs = torch.full((2, 6), -1.0)
    follower_log_probs = torch.full((2, 6), -3.0)

    mixed = build_hpf_mixed_policy_grpo_batch(
        batch=batch,
        round_index=1,
        progressive_block_size=2,
        max_response_length=6,
        leader_old_log_probs=leader_log_probs,
        follower_old_log_probs=follower_log_probs,
    )

    expected_prefix = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    expected_suffix = torch.tensor([[0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0]])
    expected_update = expected_prefix.bool() | expected_suffix.bool()
    assert torch.equal(mixed.prefix_mask.bool(), expected_prefix.bool())
    assert torch.equal(mixed.suffix_mask.bool(), expected_suffix.bool())
    assert torch.equal(mixed.batch.batch["hpf_pg_mask"].bool(), expected_update)
    assert torch.equal(
        mixed.batch.batch["old_log_probs"],
        torch.tensor(
            [
                [-1.0, -1.0, -3.0, -3.0, -3.0, -3.0],
                [-1.0, -1.0, -3.0, -3.0, -3.0, -3.0],
            ]
        ),
    )
    assert torch.equal(mixed.batch.batch["advantages"], advantages * expected_update)


def test_mixed_policy_grpo_rejects_missing_standard_advantages():
    batch = DataProto.from_single_dict({"response_mask": torch.ones(1, 4, dtype=torch.long)})
    log_probs = torch.zeros(1, 4)

    with pytest.raises(ValueError, match="advantages are required"):
        build_hpf_mixed_policy_grpo_batch(
            batch=batch,
            round_index=1,
            progressive_block_size=2,
            max_response_length=4,
            leader_old_log_probs=log_probs,
            follower_old_log_probs=log_probs,
        )


def test_mixed_policy_grpo_can_train_the_full_suffix_tail():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    advantages = torch.tensor([[2.0] * 6, [-1.0] * 6])
    batch = DataProto.from_single_dict(
        {
            "response_mask": response_mask,
            "advantages": advantages,
            "returns": advantages.clone(),
        }
    )
    leader_log_probs = torch.full((2, 6), -1.0)
    follower_log_probs = torch.full((2, 6), -3.0)

    mixed = build_hpf_mixed_policy_grpo_batch(
        batch=batch,
        round_index=1,
        progressive_block_size=2,
        max_response_length=6,
        leader_old_log_probs=leader_log_probs,
        follower_old_log_probs=follower_log_probs,
        full_suffix_tail=True,
    )

    expected_prefix = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    expected_suffix = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0]])
    expected_update = response_mask.bool()
    assert torch.equal(mixed.prefix_mask.bool(), expected_prefix.bool())
    assert torch.equal(mixed.suffix_mask.bool(), expected_suffix.bool())
    assert torch.equal(mixed.batch.batch["hpf_pg_mask"].bool(), expected_update)
    assert torch.equal(mixed.batch.batch["advantages"], advantages * expected_update)
    assert mixed.metrics["hpf/mixed_policy_grpo_full_suffix_tail"] == 1.0


def test_mixed_policy_bridge_window_is_independent_of_pg_suffix_window():
    response_mask = torch.ones(1, 8, dtype=torch.long)
    advantages = torch.ones(1, 8)
    batch = DataProto.from_single_dict(
        {
            "response_mask": response_mask,
            "advantages": advantages,
            "returns": advantages.clone(),
        }
    )
    log_probs = torch.zeros(1, 8)

    mixed = build_hpf_mixed_policy_grpo_batch(
        batch=batch,
        round_index=1,
        progressive_block_size=2,
        max_response_length=8,
        leader_old_log_probs=log_probs,
        follower_old_log_probs=log_probs,
        bridge_window_size=4,
    )

    expected_pg = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    expected_bridge = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    assert torch.equal(mixed.batch.batch["hpf_pg_mask"].bool(), expected_pg)
    assert torch.equal(mixed.batch.batch["hpf_bridge_kl_mask"].bool(), expected_bridge)
    assert torch.equal(mixed.bridge_mask.bool(), expected_bridge)


def test_temperature_policy_kl_matches_direct_distribution_computation():
    logits = torch.tensor([[1.0, -0.5, 0.25], [0.2, 0.4, -0.3]], requires_grad=True)
    low_temperature = 0.25
    high_temperature = 1.0

    actual = temperature_policy_kl_from_logits(logits, low_temperature, high_temperature)
    low_log_probs = torch.log_softmax(logits / low_temperature, dim=-1)
    high_log_probs = torch.log_softmax(logits / high_temperature, dim=-1)
    expected = torch.sum(low_log_probs.exp() * (low_log_probs - high_log_probs), dim=-1)

    assert torch.allclose(actual, expected, atol=1e-6)
    actual.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
