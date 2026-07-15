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
