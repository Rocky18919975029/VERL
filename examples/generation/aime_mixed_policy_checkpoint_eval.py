#!/usr/bin/env python3
"""Evaluate one checkpoint on AIME with a cut-conditioned mixed policy.

The first ``horizon`` response tokens are sampled from the high-temperature
policy. If generation reaches the cut, the remaining response is sampled from
the low-temperature policy. Each input parquet row produces exactly one
trajectory; the standard AIME parquet contains 32 copies of each of 30
problems, so aggregation yields 32 samples per problem without changing the
online-validation sampling budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, default=3072)
    parser.add_argument("--prefix-temperature", type=float, default=1.0)
    parser.add_argument("--prefix-top-p", type=float, default=1.0)
    parser.add_argument("--suffix-temperature", type=float, default=0.25)
    parser.add_argument("--suffix-top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def normalize_messages(prompt_value: Any) -> list[dict[str, str]]:
    prompt_value = to_builtin(prompt_value)
    if isinstance(prompt_value, str):
        return [{"role": "user", "content": prompt_value}]
    messages = []
    for message in prompt_value:
        if isinstance(message, dict):
            messages.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
            )
        else:
            messages.append({"role": "user", "content": str(message)})
    return messages


def render_prompt(tokenizer: Any, prompt_value: Any) -> str:
    messages = normalize_messages(prompt_value)
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n".join(message["content"] for message in messages)


def make_token_prompt(prompt_token_ids: list[int]) -> Any:
    try:
        from vllm.inputs import TokensPrompt

        return TokensPrompt(prompt_token_ids=prompt_token_ids)
    except (ImportError, TypeError):
        return {"prompt_token_ids": prompt_token_ids}


def make_sampling_params(**kwargs: Any) -> Any:
    from vllm import SamplingParams

    return SamplingParams(**kwargs)


def make_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    kwargs = {
        "model": args.model,
        "tokenizer": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": True,
        # Match VERL's vLLM server rule: base rollout seed + replica rank.
        "seed": args.seed + args.shard_index,
        "enforce_eager": args.enforce_eager,
    }
    return LLM(**kwargs)


def stable_problem_key(prompt: str, ground_truth: str) -> str:
    payload = json.dumps([prompt, ground_truth], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 < args.horizon <= args.max_response_length:
        raise ValueError(
            f"horizon must be in [1, {args.max_response_length}], got {args.horizon}"
        )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"invalid shard {args.shard_index} for num_shards={args.num_shards}"
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    from verl.utils.reward_score import default_compute_score

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )

    data = pd.read_parquet(args.data).reset_index(names="original_row_index")
    if args.limit is not None:
        data = data.head(args.limit)
    data = data.iloc[args.shard_index :: args.num_shards].reset_index(drop=True)
    prompts = [render_prompt(tokenizer, row["prompt"]) for _, row in data.iterrows()]

    llm = make_llm(args)
    prefix_params = make_sampling_params(
        n=1,
        temperature=args.prefix_temperature,
        top_p=args.prefix_top_p,
        max_tokens=args.horizon,
    )
    prefix_requests = llm.generate(prompts, prefix_params)

    prefix_ids_by_row: list[list[int]] = []
    prompt_ids_by_row: list[list[int]] = []
    suffix_source_rows: list[int] = []
    suffix_prompts: list[Any] = []
    for row_index, request in enumerate(prefix_requests):
        completion = request.outputs[0]
        prefix_ids = list(completion.token_ids)
        prompt_ids = list(request.prompt_token_ids)
        prefix_ids_by_row.append(prefix_ids)
        prompt_ids_by_row.append(prompt_ids)
        if len(prefix_ids) >= args.horizon and len(prefix_ids) < args.max_response_length:
            suffix_source_rows.append(row_index)
            suffix_prompts.append(make_token_prompt(prompt_ids + prefix_ids))

    suffix_ids_by_row: dict[int, list[int]] = {}
    suffix_finish_by_row: dict[int, str | None] = {}
    if suffix_prompts:
        suffix_params = make_sampling_params(
            n=1,
            temperature=args.suffix_temperature,
            top_p=args.suffix_top_p,
            max_tokens=args.max_response_length - args.horizon,
        )
        suffix_requests = llm.generate(suffix_prompts, suffix_params)
        for source_row, request in zip(suffix_source_rows, suffix_requests, strict=True):
            completion = request.outputs[0]
            suffix_ids_by_row[source_row] = list(completion.token_ids)
            suffix_finish_by_row[source_row] = getattr(completion, "finish_reason", None)

    rows: list[dict[str, Any]] = []
    for row_index, request in enumerate(prefix_requests):
        source = data.iloc[row_index]
        reward_model = to_builtin(source.get("reward_model", {}))
        extra_info = to_builtin(source.get("extra_info", {}))
        ground_truth = str(reward_model.get("ground_truth", ""))
        prefix_ids = prefix_ids_by_row[row_index]
        suffix_ids = suffix_ids_by_row.get(row_index, [])
        response_ids = prefix_ids + suffix_ids
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        score = default_compute_score(
            "math_dapo",
            response,
            ground_truth,
            extra_info=extra_info,
            strict_box_verify=True,
        )
        if not isinstance(score, dict):
            raise TypeError(f"math_dapo scorer returned {type(score).__name__}, expected dict")
        prompt = prompts[row_index]
        prefix_completion = request.outputs[0]
        rows.append(
            {
                "checkpoint_step": args.checkpoint_step,
                "horizon": args.horizon,
                "original_row_index": int(source["original_row_index"]),
                "problem_key": stable_problem_key(prompt, ground_truth),
                "prompt": prompt,
                "raw_problem": extra_info.get("raw_problem") if isinstance(extra_info, dict) else None,
                "ground_truth": ground_truth,
                "response": response,
                "is_correct": bool(score.get("acc", False)),
                "acc": bool(score.get("acc", False)),
                "score": float(score.get("score", -1.0)),
                "pred": to_builtin(score.get("pred")),
                "prefix_token_count": len(prefix_ids),
                "suffix_token_count": len(suffix_ids),
                "response_token_count": len(response_ids),
                "used_low_temperature_suffix": row_index in suffix_ids_by_row,
                "prefix_finish_reason": getattr(prefix_completion, "finish_reason", None),
                "suffix_finish_reason": suffix_finish_by_row.get(row_index),
                "prefix_temperature": args.prefix_temperature,
                "prefix_top_p": args.prefix_top_p,
                "suffix_temperature": args.suffix_temperature,
                "suffix_top_p": args.suffix_top_p,
                "max_response_length": args.max_response_length,
                "seed": args.seed,
                "engine_seed": args.seed + args.shard_index,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
        )

    stem = f"aime24_mixed_step{args.checkpoint_step}_shard{args.shard_index}"
    parquet_path = output_dir / f"{stem}.parquet"
    jsonl_path = output_dir / f"{stem}.jsonl"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    write_jsonl(jsonl_path, rows)
    accuracy = sum(row["is_correct"] for row in rows) / len(rows) if rows else 0.0
    suffix_fraction = (
        sum(row["used_low_temperature_suffix"] for row in rows) / len(rows) if rows else 0.0
    )
    print(
        f"[mixed-eval] step={args.checkpoint_step} horizon={args.horizon} "
        f"shard={args.shard_index}/{args.num_shards} rows={len(rows)} "
        f"acc={accuracy:.6f} suffix_frac={suffix_fraction:.6f}"
    )
    print(f"Wrote {parquet_path}")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
