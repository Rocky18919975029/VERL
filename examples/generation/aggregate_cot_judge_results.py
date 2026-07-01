#!/usr/bin/env python3
"""Aggregate sharded CoT judge row outputs into global CoT-Pass@K metrics."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing shard_*/cot_judge_rows.parquet.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>/aggregate.")
    parser.add_argument("--top-k", default="1,2,4,8,16,32,64,128,256,512,1024")
    return parser.parse_args()


def none_if_nan(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return float(value) if isinstance(value, float) else value


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples <= 0 or num_correct <= 0:
        return 0.0
    if k >= num_samples:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def load_rows(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("shard_*/cot_judge_rows.parquet"))
    if not paths:
        paths = sorted(input_dir.glob("*/cot_judge_rows.parquet"))
    if not paths:
        raise FileNotFoundError(f"No shard cot_judge_rows.parquet files found under {input_dir}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["judge_source_file"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if "problem_key" not in df.columns:
        if "raw_problem" in df.columns and df["raw_problem"].notna().any():
            df["problem_key"] = df["raw_problem"].astype(str)
        else:
            df["problem_key"] = df["problem_index"].astype(str)
    return df


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
        "num_shard_files": int(df["judge_source_file"].nunique()),
    }
    for k in top_ks:
        for col in ["pass", "cot_pass_any", "cot_pass_all", "cot_pass_majority"]:
            metric = f"{col}_at_{k}"
            if metric in per_problem.columns:
                summary[metric] = float(per_problem[metric].mean())
    return per_problem, summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = sorted({int(k) for k in args.top_k.split(",") if k.strip()})

    df = load_rows(input_dir)
    per_problem, summary = summarize(df, top_ks)
    summary["input_dir"] = str(input_dir)

    rows_path = output_dir / "cot_judge_rows.parquet"
    per_problem_path = output_dir / "cot_judge_per_problem.csv"
    summary_path = output_dir / "cot_judge_summary.json"
    df.to_parquet(rows_path, index=False)
    per_problem.to_csv(per_problem_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote aggregate row-level judgments to {rows_path}")
    print(f"Wrote aggregate per-problem metrics to {per_problem_path}")
    print(f"Wrote aggregate summary to {summary_path}")


if __name__ == "__main__":
    main()

