#!/usr/bin/env python3
"""Sample progressive prefix-follower rollouts for HPF-RLVR diagnostics.

This script only tests rollout construction. It samples high-temperature
prefixes, low-temperature suffixes conditioned on each prefix, verifies the
completed response, and writes prefix/suffix metadata for later inspection.
It does not train or update the model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.prime_math import compute_score as compute_math_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local HF model path.")
    parser.add_argument("--data", required=True, help="DAPO-MATH parquet path.")
    parser.add_argument("--dataset-name", default="dapo_math_17k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round-index", type=int, required=True, help="1-based progressive training round index.")
    parser.add_argument("--progressive-block-size", type=int, default=256)
    parser.add_argument("--max-response-length", type=int, default=3072)
    parser.add_argument("--num-prefixes", type=int, default=4, help="N high-temperature prefixes per prompt.")
    parser.add_argument("--num-suffixes", type=int, default=4, help="M low-temperature suffixes per prefix.")
    parser.add_argument("--prefix-temperature", type=float, default=1.0)
    parser.add_argument("--prefix-top-p", type=float, default=1.0)
    parser.add_argument("--suffix-temperature", type=float, default=0.25)
    parser.add_argument("--suffix-top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def normalize_messages(prompt_value: Any) -> list[dict[str, str]]:
    if hasattr(prompt_value, "tolist"):
        prompt_value = prompt_value.tolist()
    if isinstance(prompt_value, str):
        return [{"role": "user", "content": prompt_value}]
    messages = []
    for msg in prompt_value:
        if isinstance(msg, dict):
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": "user", "content": str(msg)})
    return messages


def render_prompt(tokenizer: Any, prompt_value: Any) -> str:
    messages = normalize_messages(prompt_value)
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n".join(msg["content"] for msg in messages)


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def verify_math_response(response: str, ground_truth: Any) -> dict[str, Any]:
    try:
        result = compute_math_score(response, str(ground_truth))
    except Exception as exc:
        return {
            "is_correct": False,
            "verifier_score": 0.0,
            "format_correct": False,
            "extracted_answer": None,
            "verifier_error": repr(exc),
        }

    if isinstance(result, tuple):
        is_correct = bool(result[0]) if len(result) > 0 else False
        format_correct = bool(result[1]) if len(result) > 1 else None
        extracted_answer = result[2] if len(result) > 2 else None
    else:
        is_correct = bool(result)
        format_correct = None
        extracted_answer = None

    return {
        "is_correct": is_correct,
        "verifier_score": 1.0 if is_correct else 0.0,
        "format_correct": format_correct,
        "extracted_answer": extracted_answer,
        "verifier_error": None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_llm(args: argparse.Namespace) -> LLM:
    kwargs = {
        "model": args.model,
        "tokenizer": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": True,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
    }
    try:
        return LLM(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        kwargs.pop("enforce_eager", None)
        return LLM(**kwargs)


def make_sampling_params(**kwargs: Any) -> SamplingParams:
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


def main() -> None:
    args = parse_args()
    if args.round_index < 1:
        raise ValueError(f"--round-index must be >= 1, got {args.round_index}")
    if args.progressive_block_size < 1:
        raise ValueError(f"--progressive-block-size must be >= 1, got {args.progressive_block_size}")
    if args.max_response_length < 1:
        raise ValueError(f"--max-response-length must be >= 1, got {args.max_response_length}")
    if args.num_prefixes < 1 or args.num_suffixes < 1:
        raise ValueError("--num-prefixes and --num-suffixes must be >= 1")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizon = min(args.round_index * args.progressive_block_size, args.max_response_length)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)

    df = pd.read_parquet(args.data).reset_index(names="original_problem_index")
    if args.limit is not None:
        df = df.head(args.limit)
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    df = df.iloc[args.shard_index :: args.num_shards].reset_index(drop=True)

    prompts = [render_prompt(tokenizer, row["prompt"]) for _, row in df.iterrows()]
    llm = make_llm(args)

    prefix_params = make_sampling_params(
        n=args.num_prefixes,
        temperature=args.prefix_temperature,
        top_p=args.prefix_top_p,
        max_tokens=horizon,
        seed=args.seed,
    )
    prefix_outputs = llm.generate(prompts, prefix_params)

    rows: list[dict[str, Any]] = []
    suffix_prompts: list[str] = []
    suffix_meta: list[tuple[int, int, str, int, str, str, str, dict[str, Any], dict[str, Any]]] = []
    prefix_rows: list[dict[str, Any]] = []
    for problem_idx, request_output in enumerate(prefix_outputs):
        source_row = df.iloc[problem_idx]
        prompt_text = prompts[problem_idx]
        extra_info = to_builtin(source_row.get("extra_info", {}))
        reward_model = to_builtin(source_row.get("reward_model", {}))
        for prefix_idx, prefix_output in enumerate(request_output.outputs):
            prefix_text = prefix_output.text
            prefix_len = len(token_ids(tokenizer, prefix_text))
            suffix_budget = 0 if horizon >= args.max_response_length else max(args.max_response_length - prefix_len, 0)
            prefix_finish_reason = str(getattr(prefix_output, "finish_reason", ""))
            prefix_stop_reason = str(getattr(prefix_output, "stop_reason", ""))
            prefix_rows.append(
                {
                    "problem_index": int(source_row.get("original_problem_index", problem_idx)),
                    "shard_local_problem_index": int(problem_idx),
                    "prefix_index": int(prefix_idx),
                    "data_source": str(source_row.get("data_source", "")),
                    "ability": str(source_row.get("ability", "")),
                    "raw_problem": extra_info.get("raw_problem"),
                    "split": extra_info.get("split"),
                    "ground_truth": reward_model.get("ground_truth"),
                    "reward_style": reward_model.get("style"),
                    "prompt": prompt_text,
                    "prefix": prefix_text,
                    "hpf_round": args.round_index,
                    "horizon_tokens": horizon,
                    "actual_prefix_tokens": prefix_len,
                    "suffix_budget_tokens": suffix_budget,
                    "prefix_temperature": args.prefix_temperature,
                    "prefix_top_p": args.prefix_top_p,
                    "prefix_finish_reason": prefix_finish_reason,
                    "prefix_stop_reason": prefix_stop_reason,
                }
            )
            if suffix_budget > 0:
                suffix_prompts.append(prompt_text + prefix_text)
                suffix_meta.append(
                    (
                        problem_idx,
                        prefix_idx,
                        prompt_text,
                        prefix_len,
                        prefix_text,
                        prefix_finish_reason,
                        prefix_stop_reason,
                        extra_info,
                        reward_model,
                    )
                )
            else:
                response_len = len(token_ids(tokenizer, prefix_text))
                verification = verify_math_response(prefix_text, reward_model.get("ground_truth"))
                rows.append(
                    {
                        "problem_index": int(source_row.get("original_problem_index", problem_idx)),
                        "shard_local_problem_index": int(problem_idx),
                        "prefix_index": int(prefix_idx),
                        "suffix_index": 0,
                        "trajectory_index": int(prefix_idx * args.num_suffixes),
                        "data_source": str(source_row.get("data_source", "")),
                        "ability": str(source_row.get("ability", "")),
                        "raw_problem": extra_info.get("raw_problem"),
                        "split": extra_info.get("split"),
                        "ground_truth": reward_model.get("ground_truth"),
                        "reward_style": reward_model.get("style"),
                        "prompt": prompt_text,
                        "prefix": prefix_text,
                        "suffix": "",
                        "response": prefix_text,
                        "hpf_round": args.round_index,
                        "horizon_tokens": horizon,
                        "actual_prefix_tokens": prefix_len,
                        "actual_suffix_tokens": 0,
                        "actual_response_tokens": response_len,
                        "prefix_mask_start": 0,
                        "prefix_mask_end": prefix_len,
                        "suffix_mask_start": prefix_len,
                        "suffix_mask_end": prefix_len,
                        "is_correct": verification["is_correct"],
                        "verifier_score": verification["verifier_score"],
                        "format_correct": verification["format_correct"],
                        "extracted_answer": verification["extracted_answer"],
                        "verifier_name": "verl.utils.reward_score.prime_math.compute_score",
                        "verifier_error": verification["verifier_error"],
                        "prefix_temperature": args.prefix_temperature,
                        "prefix_top_p": args.prefix_top_p,
                        "suffix_temperature": args.suffix_temperature,
                        "suffix_top_p": args.suffix_top_p,
                        "prefix_finish_reason": prefix_finish_reason,
                        "prefix_stop_reason": prefix_stop_reason,
                        "suffix_finish_reason": "empty",
                        "suffix_stop_reason": "empty",
                    }
                )

    suffix_params = make_sampling_params(
        n=args.num_suffixes,
        temperature=args.suffix_temperature,
        top_p=args.suffix_top_p,
        max_tokens=max(args.max_response_length - horizon, 1),
        seed=args.seed + 1009,
    )

    if suffix_prompts:
        suffix_outputs = llm.generate(suffix_prompts, suffix_params)
        for suffix_request_idx, request_output in enumerate(suffix_outputs):
            (
                problem_idx,
                prefix_idx,
                prompt_text,
                prefix_len,
                prefix_text,
                prefix_finish_reason,
                prefix_stop_reason,
                extra_info,
                reward_model,
            ) = suffix_meta[suffix_request_idx]
            source_row = df.iloc[problem_idx]
            for suffix_idx, suffix_output in enumerate(request_output.outputs):
                suffix_text = suffix_output.text
                full_response = prefix_text + suffix_text
                suffix_len = len(token_ids(tokenizer, suffix_text))
                response_len = len(token_ids(tokenizer, full_response))
                verification = verify_math_response(full_response, reward_model.get("ground_truth"))
                rows.append(
                    {
                        "problem_index": int(source_row.get("original_problem_index", problem_idx)),
                        "shard_local_problem_index": int(problem_idx),
                        "prefix_index": int(prefix_idx),
                        "suffix_index": int(suffix_idx),
                        "trajectory_index": int(prefix_idx * args.num_suffixes + suffix_idx),
                        "data_source": str(source_row.get("data_source", "")),
                        "ability": str(source_row.get("ability", "")),
                        "raw_problem": extra_info.get("raw_problem"),
                        "split": extra_info.get("split"),
                        "ground_truth": reward_model.get("ground_truth"),
                        "reward_style": reward_model.get("style"),
                        "prompt": prompt_text,
                        "prefix": prefix_text,
                        "suffix": suffix_text,
                        "response": full_response,
                        "hpf_round": args.round_index,
                        "horizon_tokens": horizon,
                        "actual_prefix_tokens": prefix_len,
                        "actual_suffix_tokens": suffix_len,
                        "actual_response_tokens": response_len,
                        "prefix_mask_start": 0,
                        "prefix_mask_end": prefix_len,
                        "suffix_mask_start": prefix_len,
                        "suffix_mask_end": response_len,
                        "is_correct": verification["is_correct"],
                        "verifier_score": verification["verifier_score"],
                        "format_correct": verification["format_correct"],
                        "extracted_answer": verification["extracted_answer"],
                        "verifier_name": "verl.utils.reward_score.prime_math.compute_score",
                        "verifier_error": verification["verifier_error"],
                        "prefix_temperature": args.prefix_temperature,
                        "prefix_top_p": args.prefix_top_p,
                        "suffix_temperature": args.suffix_temperature,
                        "suffix_top_p": args.suffix_top_p,
                        "prefix_finish_reason": prefix_finish_reason,
                        "prefix_stop_reason": prefix_stop_reason,
                        "suffix_finish_reason": str(getattr(suffix_output, "finish_reason", "")),
                        "suffix_stop_reason": str(getattr(suffix_output, "stop_reason", "")),
                    }
                )

    prefix = (
        f"{args.dataset_name}_hpf_round{args.round_index}_"
        f"b{args.progressive_block_size}_n{args.num_prefixes}_m{args.num_suffixes}"
    )
    rollout_parquet = output_dir / f"{prefix}_rollouts.parquet"
    rollout_jsonl = output_dir / f"{prefix}_rollouts.jsonl"
    prefix_parquet = output_dir / f"{prefix}_prefixes.parquet"
    summary_path = output_dir / f"{prefix}_summary.json"

    rollout_df = pd.DataFrame(rows)
    prefix_df = pd.DataFrame(prefix_rows)
    rollout_df.to_parquet(rollout_parquet, index=False)
    prefix_df.to_parquet(prefix_parquet, index=False)
    write_jsonl(rollout_jsonl, rows)

    summary = {
        "model": args.model,
        "data": args.data,
        "dataset_name": args.dataset_name,
        "round_index": args.round_index,
        "progressive_block_size": args.progressive_block_size,
        "horizon_tokens": horizon,
        "max_response_length": args.max_response_length,
        "num_input_rows": int(len(df)),
        "num_prefixes_per_problem": args.num_prefixes,
        "num_suffixes_per_prefix": args.num_suffixes,
        "num_prefixes": int(len(prefix_df)),
        "num_rollouts": int(len(rollout_df)),
        "prefix_temperature": args.prefix_temperature,
        "prefix_top_p": args.prefix_top_p,
        "suffix_temperature": args.suffix_temperature,
        "suffix_top_p": args.suffix_top_p,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "mean_actual_prefix_tokens": None if prefix_df.empty else float(prefix_df["actual_prefix_tokens"].mean()),
        "mean_suffix_budget_tokens": None if prefix_df.empty else float(prefix_df["suffix_budget_tokens"].mean()),
        "suffix_empty_prefix_frac": None if prefix_df.empty else float((prefix_df["suffix_budget_tokens"] <= 0).mean()),
        "mean_actual_suffix_tokens": None if rollout_df.empty else float(rollout_df["actual_suffix_tokens"].mean()),
        "mean_actual_response_tokens": None if rollout_df.empty else float(rollout_df["actual_response_tokens"].mean()),
        "accuracy": None if rollout_df.empty else float(rollout_df["is_correct"].mean()),
        "num_correct": 0 if rollout_df.empty else int(rollout_df["is_correct"].sum()),
        "rollout_parquet": str(rollout_parquet),
        "prefix_parquet": str(prefix_parquet),
        "rollout_jsonl": str(rollout_jsonl),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
