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
from verl.trainer.ppo.hpf_utils import (
    build_hpf_mixed_policy_grpo_batch,
    compute_hpf_clipped_grpo_surrogate,
)


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


def test_transition_surrogate_is_centered_at_behavior_policy():
    old_log_probs = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]])
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    mask = torch.ones_like(advantages, dtype=torch.bool)

    value = compute_hpf_clipped_grpo_surrogate(
        log_probs=old_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        mask=mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
    )

    assert value.item() == pytest.approx(0.0)


def test_transition_surrogate_applies_ppo_clipping():
    old_log_probs = torch.zeros(1, 2)
    log_probs = torch.log(torch.tensor([[2.0, 0.5]]))
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.ones_like(advantages, dtype=torch.bool)

    value = compute_hpf_clipped_grpo_surrogate(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        mask=mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
    )

    # Positive advantage is capped at 1.2; negative advantage uses the worse clipped value -0.8.
    assert value.item() == pytest.approx((1.2 - 0.8) / 2)
