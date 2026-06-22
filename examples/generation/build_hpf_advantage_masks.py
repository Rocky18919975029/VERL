#!/usr/bin/env python3
"""Build offline HPF follower/leader advantages and mask diagnostics.

This script consumes *_rollouts.parquet files produced by
hpf_progressive_rollout_sample.py. It does not train the model. The goal is to
validate the batch structure that HPF-RLVR training will need:

- suffix/follower advantage: GRPO-style normalization within each prefix group
- leader prefix reward: any-correct over suffixes for the same prefix
- leader advantage: GRPO-style normalization across prefixes of the same problem
- prefix/suffix loss mask lengths and non-empty fractions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="HPF rollout output dir containing shard_*/*_rollouts.parquet.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>/hpf_advantage_masks.")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--no-std-normalize",
        action="store_true",
        help="Use mean-centering only instead of GRPO-style std normalization.",
    )
    return parser.parse_args()


def load_rollouts(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("shard_*/*_rollouts.parquet"))
    if not paths:
        paths = sorted(input_dir.glob("*_rollouts.parquet"))
    if not paths:
        raise FileNotFoundError(f"No *_rollouts.parquet files found under {input_dir}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    required = {
        "problem_index",
        "prefix_index",
        "suffix_index",
        "is_correct",
        "actual_prefix_tokens",
        "actual_suffix_tokens",
        "actual_response_tokens",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["is_correct"] = df["is_correct"].astype(bool)
    df["reward"] = df["is_correct"].astype(float)
    return df


def normalize(values: pd.Series, epsilon: float, std_normalize: bool) -> pd.Series:
    if len(values) == 1:
        return pd.Series([0.0], index=values.index)
    mean = float(values.mean())
    centered = values - mean
    if not std_normalize:
        return centered
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series([0.0] * len(values), index=values.index)
    return centered / (std + epsilon)


def add_follower_advantage(df: pd.DataFrame, epsilon: float, std_normalize: bool) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["problem_index", "prefix_index"]
    out["follower_group_size"] = out.groupby(group_cols)["reward"].transform("size")
    out["follower_group_reward_mean"] = out.groupby(group_cols)["reward"].transform("mean")
    out["follower_group_reward_std"] = out.groupby(group_cols)["reward"].transform(lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0)
    out["follower_advantage"] = out.groupby(group_cols, group_keys=False)["reward"].apply(
        lambda x: normalize(x, epsilon, std_normalize)
    )
    return out


def add_leader_advantage(df: pd.DataFrame, epsilon: float, std_normalize: bool) -> pd.DataFrame:
    prefix_rewards = (
        df.groupby(["problem_index", "prefix_index"], as_index=False)
        .agg(
            leader_prefix_reward_any_correct=("reward", "max"),
            prefix_num_suffixes=("suffix_index", "nunique"),
            prefix_num_rollouts=("reward", "size"),
            prefix_num_correct=("reward", "sum"),
            prefix_mean_suffix_reward=("reward", "mean"),
            actual_prefix_tokens=("actual_prefix_tokens", "first"),
            hpf_round=("hpf_round", "first") if "hpf_round" in df.columns else ("problem_index", "first"),
            horizon_tokens=("horizon_tokens", "first") if "horizon_tokens" in df.columns else ("actual_prefix_tokens", "first"),
        )
    )
    prefix_rewards["leader_group_size"] = prefix_rewards.groupby("problem_index")[
        "leader_prefix_reward_any_correct"
    ].transform("size")
    prefix_rewards["leader_group_reward_mean"] = prefix_rewards.groupby("problem_index")[
        "leader_prefix_reward_any_correct"
    ].transform("mean")
    prefix_rewards["leader_group_reward_std"] = prefix_rewards.groupby("problem_index")[
        "leader_prefix_reward_any_correct"
    ].transform(lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0)
    prefix_rewards["leader_advantage"] = prefix_rewards.groupby("problem_index", group_keys=False)[
        "leader_prefix_reward_any_correct"
    ].apply(lambda x: normalize(x, epsilon, std_normalize))
    return df.merge(
        prefix_rewards[
            [
                "problem_index",
                "prefix_index",
                "leader_prefix_reward_any_correct",
                "prefix_num_suffixes",
                "prefix_num_rollouts",
                "prefix_num_correct",
                "prefix_mean_suffix_reward",
                "leader_group_size",
                "leader_group_reward_mean",
                "leader_group_reward_std",
                "leader_advantage",
            ]
        ],
        on=["problem_index", "prefix_index"],
        how="left",
    ), prefix_rewards


def add_mask_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prefix_loss_tokens"] = out["actual_prefix_tokens"].astype(int).clip(lower=0)
    out["suffix_loss_tokens"] = out["actual_suffix_tokens"].astype(int).clip(lower=0)
    out["has_prefix_loss"] = out["prefix_loss_tokens"] > 0
    out["has_suffix_loss"] = out["suffix_loss_tokens"] > 0
    out["prefix_loss_token_frac"] = out["prefix_loss_tokens"] / out["actual_response_tokens"].replace(0, np.nan)
    out["suffix_loss_token_frac"] = out["suffix_loss_tokens"] / out["actual_response_tokens"].replace(0, np.nan)
    return out


def summarize(df: pd.DataFrame, prefix_rewards: pd.DataFrame, std_normalize: bool) -> dict[str, object]:
    follower_zero_std = (
        df[["problem_index", "prefix_index", "follower_group_reward_std"]]
        .drop_duplicates()
        ["follower_group_reward_std"]
        .eq(0.0)
    )
    leader_zero_std = (
        prefix_rewards[["problem_index", "leader_group_reward_std"]]
        .drop_duplicates()
        ["leader_group_reward_std"]
        .eq(0.0)
    )
    return {
        "num_rollouts": int(len(df)),
        "num_problems": int(df["problem_index"].nunique()),
        "num_prefixes": int(prefix_rewards.shape[0]),
        "mean_rollout_reward": float(df["reward"].mean()),
        "num_correct_rollouts": int(df["reward"].sum()),
        "prefix_any_correct_rate": float(prefix_rewards["leader_prefix_reward_any_correct"].mean()),
        "num_any_correct_prefixes": int(prefix_rewards["leader_prefix_reward_any_correct"].sum()),
        "std_normalize": bool(std_normalize),
        "mean_follower_advantage": float(df["follower_advantage"].mean()),
        "std_follower_advantage": float(df["follower_advantage"].std(ddof=1)),
        "min_follower_advantage": float(df["follower_advantage"].min()),
        "max_follower_advantage": float(df["follower_advantage"].max()),
        "mean_leader_advantage": float(prefix_rewards["leader_advantage"].mean()),
        "std_leader_advantage": float(prefix_rewards["leader_advantage"].std(ddof=1)),
        "min_leader_advantage": float(prefix_rewards["leader_advantage"].min()),
        "max_leader_advantage": float(prefix_rewards["leader_advantage"].max()),
        "follower_groups": int(len(follower_zero_std)),
        "follower_zero_std_groups": int(follower_zero_std.sum()),
        "follower_zero_std_group_frac": float(follower_zero_std.mean()),
        "leader_groups": int(len(leader_zero_std)),
        "leader_zero_std_groups": int(leader_zero_std.sum()),
        "leader_zero_std_group_frac": float(leader_zero_std.mean()),
        "mean_prefix_loss_tokens": float(df["prefix_loss_tokens"].mean()),
        "mean_suffix_loss_tokens": float(df["suffix_loss_tokens"].mean()),
        "suffix_empty_frac": float((~df["has_suffix_loss"]).mean()),
        "prefix_empty_frac": float((~df["has_prefix_loss"]).mean()),
        "mean_prefix_loss_token_frac": float(df["prefix_loss_token_frac"].mean()),
        "mean_suffix_loss_token_frac": float(df["suffix_loss_token_frac"].mean()),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "hpf_advantage_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    std_normalize = not args.no_std_normalize

    df = load_rollouts(input_dir)
    df = add_follower_advantage(df, args.epsilon, std_normalize)
    df, prefix_rewards = add_leader_advantage(df, args.epsilon, std_normalize)
    df = add_mask_diagnostics(df)
    summary = summarize(df, prefix_rewards, std_normalize)

    df.to_parquet(output_dir / "hpf_rollout_advantage_masks.parquet", index=False)
    prefix_rewards.to_parquet(output_dir / "hpf_prefix_rewards.parquet", index=False)
    prefix_rewards.to_csv(output_dir / "hpf_prefix_rewards.csv", index=False)
    (output_dir / "hpf_advantage_mask_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote rollout-level table to {output_dir / 'hpf_rollout_advantage_masks.parquet'}")
    print(f"Wrote prefix-level table to {output_dir / 'hpf_prefix_rewards.parquet'}")


if __name__ == "__main__":
    main()
