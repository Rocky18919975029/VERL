#!/usr/bin/env python3
"""Offline CoT verifier using DeepSeek-R1-0528-Qwen3-8B-style judging.

This script consumes trajectory parquet files produced by the generation scripts
in this directory. It asks a local vLLM judge model to verify the reasoning of
each solution multiple times, then reports Pass@K and CoT-Pass@K.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from math import comb
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams


JUDGE_PROMPT_TEMPLATE = """You are an expert in mathematics and logical reasoning. Your task is to evaluate the correctness of a solution to a given math problem, with a **strong emphasis on the reasoning process**, not just the final answer.

Below is the **Problem** and the **Solution (Provided by another AI model)**:

---
**Problem**:
{question}

**Solution (Provided by another AI model)**:
{solution}
---

Please perform the following tasks:
1. **Analyze the solution step-by-step**, paying close attention to:
   - Computational accuracy
   - Logical consistency
   - Conceptual understanding
   - Whether the reasoning is valid and complete
2. **Identify any issues or errors in the reasoning**, even if the final answer is correct. Classify them into the following categories (if applicable):
   - **Calculation Error**: Mistakes in arithmetic, algebraic manipulation, or numerical computation.
   - **Logical Error**: Invalid reasoning, flawed logic, or incorrect inference.
   - **Conceptual Error**: Misunderstanding or misuse of mathematical concepts or definitions.
   - **Omission / Incompleteness**: Missing steps, incomplete justification, or not addressing all parts of the question.
   - **Other**: Any other type of error that does not fit into the above categories.
3. **Provide a final judgment** on whether the solution is logically sound and free of errors in reasoning.

Please format your response as follows:

---
**Issues Identified:**
- [Issue 1]: [Classification] - [Brief explanation]
- [Issue 2]: [Classification] - [Brief explanation]
- ...

Let's think step by step and output your final judgment within \\boxed{{}}
\\boxed{{yes}} or \\boxed{{no}}
"""


BOXED_RE = re.compile(r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?(yes|no)\s*\}?\s*\}", re.IGNORECASE)
ISSUE_TYPES = {
    "calculation_error": re.compile(r"calculation error", re.IGNORECASE),
    "logical_error": re.compile(r"logical error", re.IGNORECASE),
    "conceptual_error": re.compile(r"conceptual error", re.IGNORECASE),
    "omission_incompleteness": re.compile(r"omission|incompleteness|incomplete", re.IGNORECASE),
    "other_error": re.compile(r"\bother\b", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing shard_* parquet outputs.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>/cot_judge.")
    parser.add_argument("--judge-model", required=True, help="Local path to DeepSeek-R1-0528-Qwen3-8B.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-attempts", type=int, default=3)
    parser.add_argument("--judge-temperature", type=float, default=0.6)
    parser.add_argument("--judge-top-p", type=float, default=0.95)
    parser.add_argument("--max-judge-tokens", type=int, default=16384)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument(
        "--cot-label-aggregation",
        choices=["any", "all", "majority"],
        default="majority",
        help="Aggregation used for the row-level cot_correctness label.",
    )
    parser.add_argument("--top-k", default="1,2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument(
        "--judge-scope",
        choices=["answer-correct", "all"],
        default="answer-correct",
        help="Judge only answer-correct responses by default, matching CoT-Pass@K's D count efficiently.",
    )
    parser.add_argument("--limit-problems", type=int, default=None, help="Keep only the first N problems before judging.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Debug limit after filtering.")
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


def load_trajectories(input_dir: Path) -> pd.DataFrame:
    patterns = ["shard_*/*_rollouts.parquet", "*_rollouts.parquet", "shard_*/*_loglik.parquet", "*_loglik.parquet"]
    paths: list[Path] = []
    for pattern in patterns:
        paths = sorted(input_dir.glob(pattern))
        if paths:
            break
    if not paths:
        raise FileNotFoundError(f"No *_rollouts.parquet or *_loglik.parquet files found under {input_dir}")

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    required = {"problem_index", "response", "is_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "sample_index" not in df.columns:
        if "trajectory_index" in df.columns:
            df["sample_index"] = df["trajectory_index"]
        elif {"prefix_index", "suffix_index"} <= set(df.columns):
            df["sample_index"] = df["prefix_index"].astype(int) * 100000 + df["suffix_index"].astype(int)
        else:
            df["sample_index"] = df.groupby("problem_index").cumcount()

    if "raw_problem" in df.columns and df["raw_problem"].notna().any():
        df["judge_question"] = df["raw_problem"].astype(str)
        df["problem_key"] = df["raw_problem"].astype(str)
    elif "prompt" in df.columns:
        df["judge_question"] = df["prompt"].astype(str)
        df["problem_key"] = df["problem_index"].astype(str)
    else:
        raise ValueError("Need either raw_problem or prompt column to build judge prompts.")

    df["is_correct"] = df["is_correct"].astype(bool)
    df["row_id"] = range(len(df))
    return df


def make_llm(args: argparse.Namespace) -> LLM:
    kwargs = {
        "model": args.judge_model,
        "tokenizer": args.judge_model,
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


def render_judge_prompt(tokenizer: Any, question: str, solution: str) -> str:
    content = JUDGE_PROMPT_TEMPLATE.format(question=question, solution=solution)
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return content


def parse_judgment(text: str) -> dict[str, Any]:
    matches = BOXED_RE.findall(text)
    final = matches[-1].lower() if matches else None
    row = {
        "cot_judge_pass": final == "yes",
        "cot_judge_label": final,
        "cot_judge_parse_ok": final is not None,
    }
    for key, pattern in ISSUE_TYPES.items():
        row[key] = bool(pattern.search(text))
    return row


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples <= 0 or num_correct <= 0:
        return 0.0
    if k >= num_samples:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def aggregate_attempts(attempts: list[bool]) -> dict[str, Any]:
    n = len(attempts)
    yes = int(sum(attempts))
    return {
        "cot_judge_yes_count": yes,
        "cot_judge_attempts": n,
        "cot_any_correct": yes >= 1,
        "cot_all_correct": yes == n if n else False,
        "cot_majority_correct": yes > n / 2 if n else False,
    }


def summarize(df: pd.DataFrame, top_ks: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_problem_rows = []
    for key, group in df.groupby("problem_key", sort=False):
        group = group.sort_values("sample_index", kind="mergesort")
        g = int(len(group))
        answer_correct = group["is_correct"].astype(bool)
        c = int(answer_correct.sum())
        any_cot = group["cot_any_correct"].astype(bool) & answer_correct
        all_cot = group["cot_all_correct"].astype(bool) & answer_correct
        majority_cot = group["cot_majority_correct"].astype(bool) & answer_correct
        row: dict[str, Any] = {
            "problem_key": key,
            "problem_index": group["problem_index"].iloc[0],
            "num_samples": g,
            "num_answer_correct": c,
            "num_cot_any_correct": int(any_cot.sum()),
            "num_cot_all_correct": int(all_cot.sum()),
            "num_cot_majority_correct": int(majority_cot.sum()),
            "p_ca": c / g if g else 0.0,
            "p_cc_given_ca_any": int(any_cot.sum()) / c if c else None,
            "p_cc_given_ca_all": int(all_cot.sum()) / c if c else None,
            "p_cc_given_ca_majority": int(majority_cot.sum()) / c if c else None,
        }
        if "raw_problem" in group.columns:
            row["raw_problem"] = group["raw_problem"].iloc[0]
        if "ground_truth" in group.columns:
            row["ground_truth"] = group["ground_truth"].iloc[0]
        for k in top_ks:
            row[f"pass_at_{k}"] = pass_at_k(g, c, k)
            row[f"cot_pass_any_at_{k}"] = pass_at_k(g, int(any_cot.sum()), k)
            row[f"cot_pass_all_at_{k}"] = pass_at_k(g, int(all_cot.sum()), k)
            row[f"cot_pass_majority_at_{k}"] = pass_at_k(g, int(majority_cot.sum()), k)
        per_problem_rows.append(row)

    per_problem = pd.DataFrame(per_problem_rows)
    judged_mask = df["cot_judge_attempts"] > 0
    summary: dict[str, Any] = {
        "num_problems": int(len(per_problem)),
        "num_responses": int(len(df)),
        "num_answer_correct": int(df["is_correct"].sum()),
        "answer_accuracy": float(df["is_correct"].mean()) if len(df) else 0.0,
        "judge_attempts_per_judged_response": int(df["cot_judge_attempts"].max()) if len(df) else 0,
        "num_judged_responses": int(judged_mask.sum()),
        "num_parse_failures": int((~df.loc[judged_mask, "cot_judge_parse_all_ok"]).sum()),
        "cot_any_correct_responses": int((df["is_correct"] & df["cot_any_correct"]).sum()),
        "cot_all_correct_responses": int((df["is_correct"] & df["cot_all_correct"]).sum()),
        "cot_majority_correct_responses": int((df["is_correct"] & df["cot_majority_correct"]).sum()),
        "mean_p_ca": float(per_problem["p_ca"].mean()) if len(per_problem) else 0.0,
        "mean_p_cc_given_ca_any": none_if_nan(per_problem["p_cc_given_ca_any"].mean()),
        "mean_p_cc_given_ca_all": none_if_nan(per_problem["p_cc_given_ca_all"].mean()),
        "mean_p_cc_given_ca_majority": none_if_nan(per_problem["p_cc_given_ca_majority"].mean()),
    }
    for k in top_ks:
        for col in ["pass", "cot_pass_any", "cot_pass_all", "cot_pass_majority"]:
            metric = f"{col}_at_{k}"
            if metric in per_problem.columns:
                summary[metric] = float(per_problem[metric].mean())
    return per_problem, summary


def none_if_nan(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return float(value) if isinstance(value, float) else value


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "cot_judge"
    output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = sorted({int(k) for k in args.top_k.split(",") if k.strip()})

    df = load_trajectories(input_dir)
    if args.limit_problems is not None:
        problem_keys = list(dict.fromkeys(df["problem_key"].tolist()))[: args.limit_problems]
        df = df[df["problem_key"].isin(problem_keys)].copy()
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    df = df.iloc[args.shard_index :: args.num_shards].reset_index(drop=True)
    df["row_id"] = range(len(df))
    if args.judge_scope == "answer-correct":
        judge_df = df[df["is_correct"]].copy()
    else:
        judge_df = df.copy()
    if args.limit_rows is not None:
        judge_df = judge_df.head(args.limit_rows).copy()

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, local_files_only=True, trust_remote_code=True)
    llm = make_llm(args)
    sampling_params = make_sampling_params(
        n=args.judge_attempts,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        max_tokens=args.max_judge_tokens,
        seed=args.seed,
    )

    judge_results: dict[int, dict[str, Any]] = {}
    prompts = [
        render_judge_prompt(tokenizer, row["judge_question"], row["response"])
        for _, row in judge_df.iterrows()
    ]
    row_ids = judge_df["row_id"].tolist()

    total_batches = (len(prompts) + args.judge_batch_size - 1) // args.judge_batch_size if prompts else 0
    progress_start = time.perf_counter()
    print(
        f"[cot-judge] shard={args.shard_index}/{args.num_shards} "
        f"judge_rows={len(prompts)} attempts={args.judge_attempts} batches={total_batches}",
        flush=True,
    )
    batch_starts = list(range(0, len(prompts), args.judge_batch_size))
    progress_bar = tqdm(
        enumerate(batch_starts, start=1),
        total=total_batches,
        desc=f"CoT judge shard {args.shard_index}/{args.num_shards}",
        dynamic_ncols=True,
    )
    for batch_idx, start in progress_bar:
        end = min(start + args.judge_batch_size, len(prompts))
        batch_start = time.perf_counter()
        print(
            f"[cot-judge] shard={args.shard_index}/{args.num_shards} "
            f"batch={batch_idx}/{total_batches} rows={start}-{end}/{len(prompts)} start",
            flush=True,
        )
        outputs = llm.generate(prompts[start:end], sampling_params)
        for local_idx, request_output in enumerate(outputs):
            row_id = int(row_ids[start + local_idx])
            attempt_texts = [out.text for out in request_output.outputs]
            parsed = [parse_judgment(text) for text in attempt_texts]
            passes = [bool(item["cot_judge_pass"]) for item in parsed]
            aggregate = aggregate_attempts(passes)
            result: dict[str, Any] = {
                **aggregate,
                "cot_judge_parse_all_ok": all(bool(item["cot_judge_parse_ok"]) for item in parsed),
                "cot_judge_raw_outputs": attempt_texts,
            }
            for i, item in enumerate(parsed):
                attempt = i + 1
                result[f"cot_judge_{attempt}_pass"] = item["cot_judge_pass"]
                result[f"cot_judge_{attempt}_label"] = item["cot_judge_label"]
                result[f"cot_judge_{attempt}_parse_ok"] = item["cot_judge_parse_ok"]
                for key in ISSUE_TYPES:
                    result[f"cot_judge_{attempt}_{key}"] = item[key]
            judge_results[row_id] = result
        elapsed = time.perf_counter() - progress_start
        batch_elapsed = time.perf_counter() - batch_start
        avg_batch = elapsed / batch_idx
        remaining = avg_batch * (total_batches - batch_idx)
        progress_bar.set_postfix(
            rows=f"{end}/{len(prompts)}",
            batch_s=f"{batch_elapsed:.1f}",
            eta_s=f"{remaining:.1f}",
        )
        print(
            f"[cot-judge] shard={args.shard_index}/{args.num_shards} "
            f"batch={batch_idx}/{total_batches} rows_done={end}/{len(prompts)} "
            f"batch_elapsed_s={batch_elapsed:.1f} elapsed_s={elapsed:.1f} "
            f"eta_s={remaining:.1f}",
            flush=True,
        )

    default_result = {
        "cot_judge_yes_count": 0,
        "cot_judge_attempts": 0,
        "cot_any_correct": False,
        "cot_all_correct": False,
        "cot_majority_correct": False,
        "cot_judge_parse_all_ok": True,
        "cot_judge_raw_outputs": [],
    }
    result_rows = []
    for _, row in df.iterrows():
        row_id = int(row["row_id"])
        merged = row.to_dict()
        merged.update(judge_results.get(row_id, default_result))
        result_rows.append(merged)
    result_df = pd.DataFrame(result_rows)
    result_df["answer_correctness"] = result_df["is_correct"].astype(bool)
    cot_label_col = f"cot_{args.cot_label_aggregation}_correct"
    result_df["cot_correctness"] = result_df["answer_correctness"] & result_df[cot_label_col].astype(bool)
    result_df["cot_correctness_aggregation"] = args.cot_label_aggregation

    per_problem, summary = summarize(result_df, top_ks)
    summary.update(
        {
            "input_dir": str(input_dir),
            "judge_model": args.judge_model,
            "judge_scope": args.judge_scope,
            "limit_problems": args.limit_problems,
            "limit_rows": args.limit_rows,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "judge_attempts": args.judge_attempts,
            "judge_temperature": args.judge_temperature,
            "judge_top_p": args.judge_top_p,
            "max_judge_tokens": args.max_judge_tokens,
            "cot_label_aggregation": args.cot_label_aggregation,
        }
    )

    result_path = output_dir / "cot_judge_rows.parquet"
    per_problem_path = output_dir / "cot_judge_per_problem.csv"
    summary_path = output_dir / "cot_judge_summary.json"
    result_df.to_parquet(result_path, index=False)
    per_problem.to_csv(per_problem_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote row-level judgments to {result_path}")
    print(f"Wrote per-problem metrics to {per_problem_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
