#!/usr/bin/env python3
"""Sample Math-500 responses and score response log likelihoods with vLLM.

For each Math-500 problem, this script samples N responses from a Qwen2.5-7B
base model, then recomputes the model log likelihood of each sampled response
conditioned on the prompt. The reported normalized score is:

    response_log_likelihood / number_of_response_tokens

The scorer uses vLLM prompt logprobs on the concatenated prompt+response so the
score is computed under the model, independently of the sampling temperature.
"""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--data", required=True, help="Math-500 parquet path.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=16, help="Responses per problem.")
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split dataset into this many interleaved shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index in [0, num_shards).")
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


def selected_logprob(logprob_entry: Any, token_id: int) -> float | None:
    if logprob_entry is None:
        return None
    candidates = logprob_entry
    if not isinstance(candidates, dict):
        return None
    item = candidates.get(token_id)
    if item is None:
        item = candidates.get(str(token_id))
    if item is None and len(candidates) == 1:
        item = next(iter(candidates.values()))
    if item is None:
        return None
    value = getattr(item, "logprob", item)
    try:
        return float(value)
    except Exception:
        return None


def generation_cumulative_logprob(output: Any) -> float | None:
    value = getattr(output, "cumulative_logprob", None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


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
    for optional_key in ("seed", "enforce_eager"):
        try:
            return LLM(**kwargs)
        except TypeError:
            kwargs.pop(optional_key, None)
    return LLM(**kwargs)


def make_sampling_params(**kwargs: Any) -> SamplingParams:
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    df = pd.read_parquet(args.data).reset_index(names="original_problem_index")
    if args.limit is not None:
        df = df.head(args.limit)
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    df = df.iloc[args.shard_index :: args.num_shards].reset_index(drop=True)

    prompts: list[str] = [render_prompt(tokenizer, row["prompt"]) for _, row in df.iterrows()]

    llm = make_llm(args)

    sampling_params = make_sampling_params(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=1,
        seed=args.seed,
    )

    request_outputs = llm.generate(prompts, sampling_params)

    rows: list[dict[str, Any]] = []
    score_prompts: list[str] = []
    score_meta: list[tuple[int, list[int], list[int]]] = []

    for problem_idx, request_output in enumerate(request_outputs):
        source_row = df.iloc[problem_idx]
        prompt_text = prompts[problem_idx]
        prompt_ids = token_ids(tokenizer, prompt_text)
        extra_info = to_builtin(source_row.get("extra_info", {}))
        reward_model = to_builtin(source_row.get("reward_model", {}))
        for sample_idx, completion in enumerate(request_output.outputs):
            response = completion.text
            verification = verify_math_response(response, reward_model.get("ground_truth"))
            full_text = prompt_text + response
            full_ids = token_ids(tokenizer, full_text)
            response_ids = full_ids[len(prompt_ids) :]
            row = {
                "problem_index": int(source_row.get("original_problem_index", problem_idx)),
                "shard_local_problem_index": int(problem_idx),
                "sample_index": int(sample_idx),
                "data_source": str(source_row.get("data_source", "")),
                "ability": str(source_row.get("ability", "")),
                "raw_problem": extra_info.get("raw_problem"),
                "split": extra_info.get("split"),
                "ground_truth": reward_model.get("ground_truth"),
                "reward_style": reward_model.get("style"),
                "prompt": prompt_text,
                "response": response,
                "response_char_len": len(response),
                "response_token_len": len(response_ids),
                "is_correct": verification["is_correct"],
                "verifier_score": verification["verifier_score"],
                "format_correct": verification["format_correct"],
                "extracted_answer": verification["extracted_answer"],
                "verifier_name": "verl.utils.reward_score.prime_math.compute_score",
                "verifier_error": verification["verifier_error"],
                "sampling_temperature": args.temperature,
                "sampling_top_p": args.top_p,
                "generation_cumulative_logprob": generation_cumulative_logprob(completion),
                "model_log_likelihood": None,
                "model_avg_log_likelihood": None,
            }
            rows.append(row)
            score_prompts.append(full_text)
            score_meta.append((len(rows) - 1, prompt_ids, full_ids))

    score_params = make_sampling_params(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
    )

    for start in range(0, len(score_prompts), args.score_batch_size):
        end = min(start + args.score_batch_size, len(score_prompts))
        scored_outputs = llm.generate(score_prompts[start:end], score_params)
        for local_idx, scored in enumerate(scored_outputs):
            row_idx, prompt_ids, full_ids = score_meta[start + local_idx]
            response_start = len(prompt_ids)
            response_ids = full_ids[response_start:]
            prompt_logprobs = getattr(scored, "prompt_logprobs", None)
            token_logprobs: list[float] = []
            if prompt_logprobs is not None:
                for pos in range(response_start, min(len(full_ids), len(prompt_logprobs))):
                    lp = selected_logprob(prompt_logprobs[pos], full_ids[pos])
                    if lp is not None and math.isfinite(lp):
                        token_logprobs.append(lp)
            total = sum(token_logprobs) if token_logprobs else None
            rows[row_idx]["scored_token_count"] = len(token_logprobs)
            rows[row_idx]["model_log_likelihood"] = total
            rows[row_idx]["model_avg_log_likelihood"] = (
                total / len(token_logprobs) if total is not None and token_logprobs else None
            )

    parquet_path = output_dir / "math500_qwen25_7b_temp025_n16_loglik.parquet"
    jsonl_path = output_dir / "math500_qwen25_7b_temp025_n16_loglik.jsonl"
    summary_path = output_dir / "math500_qwen25_7b_temp025_n16_summary.json"

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(parquet_path, index=False)
    write_jsonl(jsonl_path, rows)

    summary = {
        "model": args.model,
        "data": args.data,
        "num_problems": len(df),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "responses_per_problem": args.n,
        "num_responses": len(rows),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "parquet": str(parquet_path),
        "jsonl": str(jsonl_path),
        "mean_response_token_len": float(out_df["response_token_len"].mean()),
        "mean_model_avg_log_likelihood": float(out_df["model_avg_log_likelihood"].mean()),
        "accuracy": float(out_df["is_correct"].mean()),
        "num_correct": int(out_df["is_correct"].sum()),
        "num_unscored": int(out_df["model_avg_log_likelihood"].isna().sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
