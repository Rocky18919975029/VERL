# Copyright 2026 The verl team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from verl import DataProto


@dataclass
class HPFMaskedBatch:
    batch: DataProto
    metrics: dict[str, float]


def _normalize_group_scores(
    scores: torch.Tensor, group_ids: np.ndarray, epsilon: float, std_normalize: bool
) -> torch.Tensor:
    out = torch.zeros_like(scores, dtype=torch.float32)
    unique_ids = np.unique(group_ids)
    for group_id in unique_ids:
        idx_np = np.nonzero(group_ids == group_id)[0]
        if len(idx_np) <= 1:
            continue
        idx = torch.as_tensor(idx_np, device=scores.device, dtype=torch.long)
        group_scores = scores[idx].float()
        centered = group_scores - group_scores.mean()
        if std_normalize:
            std = group_scores.std(unbiased=True)
            if torch.isfinite(std) and std > 0:
                centered = centered / (std + epsilon)
            else:
                centered = torch.zeros_like(centered)
        out[idx] = centered
    return out


def _group_ids(*arrays: np.ndarray) -> np.ndarray:
    return np.array(["::".join(map(str, values)) for values in zip(*arrays, strict=True)], dtype=object)


def _sequence_scores(token_level_rewards: torch.Tensor) -> torch.Tensor:
    return token_level_rewards.sum(dim=-1).float()


def _make_prefix_suffix_masks(
    response_mask: torch.Tensor, prefix_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    response_len = response_mask.shape[-1]
    positions = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
    prefix_lengths = prefix_lengths.to(device=response_mask.device).long().clamp(min=0, max=response_len).unsqueeze(1)
    prefix_mask = (positions < prefix_lengths) & response_mask.bool()
    suffix_mask = (positions >= prefix_lengths) & response_mask.bool()
    return prefix_mask.to(response_mask.dtype), suffix_mask.to(response_mask.dtype)


def _clone_for_masked_update(
    batch: DataProto,
    mask: torch.Tensor,
    scalar_advantages: torch.Tensor,
    old_log_probs: torch.Tensor | None = None,
) -> DataProto:
    update_batch = batch.select(
        batch_keys=list(batch.batch.keys()),
        non_tensor_batch_keys=list(batch.non_tensor_batch.keys()),
        meta_info_keys=list(batch.meta_info.keys()),
        deepcopy=True,
    )
    if old_log_probs is not None:
        update_batch.batch["old_log_probs"] = old_log_probs.to(device=mask.device, dtype=torch.float32)
    update_batch.batch["response_mask"] = mask
    update_batch.batch["advantages"] = scalar_advantages.unsqueeze(-1).to(mask.device) * mask
    update_batch.batch["returns"] = update_batch.batch["advantages"]
    return update_batch


def build_hpf_masked_batches(
    batch: DataProto,
    round_index: int,
    progressive_block_size: int,
    max_response_length: int,
    epsilon: float = 1e-6,
    std_normalize: bool = True,
    follower_old_log_probs: torch.Tensor | None = None,
    leader_old_log_probs: torch.Tensor | None = None,
) -> tuple[HPFMaskedBatch | None, HPFMaskedBatch]:
    """Build suffix/follower and prefix/leader masked update batches.

    This builds the follower suffix update and leader prefix update for HPF-RLVR.
    When tree rollout metadata is available, follower advantages are normalized
    within each sampled prefix and leader prefix rewards use any-correct over the
    suffixes under that prefix. Without tree metadata, it falls back to the
    earlier masked full-response smoke behavior.
    """
    if "response_mask" not in batch.batch:
        raise ValueError("response_mask is required before building HPF masks")
    if "token_level_rewards" not in batch.batch:
        raise ValueError("token_level_rewards is required before building HPF masks")
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("uid is required before building HPF advantages")

    response_mask = batch.batch["response_mask"]
    device = response_mask.device
    response_len = response_mask.shape[-1]
    horizon = min(int(round_index) * int(progressive_block_size), int(max_response_length), response_len)
    prefix_lengths = torch.full((response_mask.shape[0],), horizon, dtype=torch.long, device=device)
    prefix_lengths = torch.minimum(prefix_lengths, response_mask.sum(dim=-1).long())
    prefix_mask, suffix_mask = _make_prefix_suffix_masks(response_mask, prefix_lengths)

    rewards = _sequence_scores(batch.batch["token_level_rewards"])
    correct = (rewards > 0).float()
    uid = batch.non_tensor_batch["uid"]

    problem_ids = batch.non_tensor_batch.get("hpf_problem_uid", uid)
    prefix_group_ids = batch.non_tensor_batch.get("hpf_prefix_uid")
    has_tree_groups = prefix_group_ids is not None
    if prefix_group_ids is None:
        prefix_ids = np.arange(len(uid), dtype=object)
        prefix_group_ids = _group_ids(uid, prefix_ids)

    follower_adv = _normalize_group_scores(rewards, prefix_group_ids, epsilon, std_normalize)

    leader_reward = torch.zeros_like(correct)
    for prefix_group_id in np.unique(prefix_group_ids):
        idx_np = np.nonzero(prefix_group_ids == prefix_group_id)[0]
        idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
        leader_reward[idx] = correct[idx].max()
    leader_adv = _normalize_group_scores(leader_reward, problem_ids, epsilon, std_normalize)

    suffix_nonempty = suffix_mask.sum(dim=-1) > 0
    follower_batch = None
    if bool(suffix_nonempty.any()):
        follower_update_batch = _clone_for_masked_update(
            batch, suffix_mask, follower_adv, old_log_probs=follower_old_log_probs
        )[
            suffix_nonempty.detach().cpu().numpy()
        ]
        follower_batch = HPFMaskedBatch(
            batch=follower_update_batch,
            metrics={
                "hpf/follower_batch_size": float(len(follower_update_batch)),
                "hpf/follower_nonempty_frac": float(suffix_nonempty.float().mean().item()),
                "hpf/follower_adv_mean": float(follower_adv.mean().item()),
                "hpf/follower_adv_std": float(follower_adv.std(unbiased=True).item()),
            },
        )

    leader_batch = HPFMaskedBatch(
        batch=_clone_for_masked_update(
            batch,
            prefix_mask,
            leader_adv,
            old_log_probs=leader_old_log_probs,
        ),
        metrics={
            "hpf/enabled": 1.0,
            "hpf/round_index": float(round_index),
            "hpf/horizon_tokens": float(horizon),
            "hpf/prefix_tokens_mean": float(prefix_mask.sum(dim=-1).float().mean().item()),
            "hpf/suffix_tokens_mean": float(suffix_mask.sum(dim=-1).float().mean().item()),
            "hpf/suffix_empty_frac": float((~suffix_nonempty).float().mean().item()),
            "hpf/leader_adv_mean": float(leader_adv.mean().item()),
            "hpf/leader_adv_std": float(leader_adv.std(unbiased=True).item()),
            "hpf/leader_prefix_any_correct_rate": float(leader_reward.mean().item()),
            "hpf/minimal_grouping": 0.0 if has_tree_groups else 1.0,
            "hpf/leader_prefix_groups": float(len(np.unique(prefix_group_ids))),
        },
    )
    return follower_batch, leader_batch
