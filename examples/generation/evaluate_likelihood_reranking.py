#!/usr/bin/env python3
"""Evaluate per-problem likelihood reranking quality.

This script reads existing *_loglik.parquet files produced by
math500_qwen25_7b_sample_and_score.py. It does not generate new responses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Experiment output dir containing *_loglik.parquet files.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>/rerank_eval.")
    parser.add_argument(
        "--top-k",
        default="1,2,4,8,16,32,64",
        help="Comma-separated k values for likelihood-reranked pass@k.",
    )
    return parser.parse_args()


def load_rows(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("shard_*/*_loglik.parquet"))
    if not paths:
        paths = sorted(input_dir.glob("*_loglik.parquet"))
    if not paths:
        raise FileNotFoundError(f"No *_loglik.parquet files found under {input_dir}")

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    required = {"model_avg_log_likelihood", "is_correct", "problem_index", "sample_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["model_avg_log_likelihood", "is_correct"]).copy()
    df["is_correct"] = df["is_correct"].astype(bool)
    df["model_avg_log_likelihood"] = df["model_avg_log_likelihood"].astype(float)

    if "raw_problem" in df.columns and df["raw_problem"].notna().any():
        df["problem_key"] = df["raw_problem"].astype(str)
    else:
        df["problem_key"] = df["problem_index"].astype(str)
    return df


def auroc_binary(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def average_precision_binary(scores: np.ndarray, labels: np.ndarray) -> float | None:
    num_pos = int(labels.sum())
    if num_pos == 0 or num_pos == len(labels):
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    hit_count = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precisions = hit_count / ranks
    return float(precisions[sorted_labels].sum() / num_pos)


def pairwise_preference(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def evaluate_problem(problem_key: str, group: pd.DataFrame, top_ks: list[int]) -> dict[str, object]:
    ranked = group.sort_values(
        ["model_avg_log_likelihood", "sample_index"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    labels = ranked["is_correct"].to_numpy(dtype=bool)
    scores = ranked["model_avg_log_likelihood"].to_numpy(dtype=float)
    correct_ranks = np.flatnonzero(labels) + 1

    row: dict[str, object] = {
        "problem_key": problem_key,
        "problem_index": ranked["problem_index"].iloc[0],
        "num_responses": int(len(ranked)),
        "num_correct": int(labels.sum()),
        "response_accuracy": float(labels.mean()),
        "oracle_correct": bool(labels.any()),
        "top1_correct": bool(labels[0]) if len(labels) else False,
        "top1_sample_index": int(ranked["sample_index"].iloc[0]) if len(ranked) else None,
        "top1_likelihood": float(scores[0]) if len(scores) else None,
        "best_correct_rank": int(correct_ranks[0]) if len(correct_ranks) else None,
        "mrr": float(1.0 / correct_ranks[0]) if len(correct_ranks) else 0.0,
        "pairwise_pref_acc": pairwise_preference(scores, labels),
        "auroc": auroc_binary(scores, labels),
        "average_precision": average_precision_binary(scores, labels),
    }
    if "raw_problem" in ranked.columns:
        row["raw_problem"] = ranked["raw_problem"].iloc[0]
    if "ground_truth" in ranked.columns:
        row["ground_truth"] = ranked["ground_truth"].iloc[0]

    for k in top_ks:
        capped_k = min(k, len(labels))
        row[f"rerank_pass_at_{k}"] = bool(labels[:capped_k].any()) if capped_k > 0 else False
    return row


def summarize(per_problem: pd.DataFrame, top_ks: list[int]) -> dict[str, object]:
    valid_pairwise = per_problem["pairwise_pref_acc"].dropna()
    valid_auroc = per_problem["auroc"].dropna()
    valid_ap = per_problem["average_precision"].dropna()
    num_with_correct = int(per_problem["oracle_correct"].sum())
    best_correct_rank = per_problem["best_correct_rank"].dropna()

    summary: dict[str, object] = {
        "num_problems": int(len(per_problem)),
        "num_responses": int(per_problem["num_responses"].sum()),
        "num_correct": int(per_problem["num_correct"].sum()),
        "response_accuracy": float(per_problem["num_correct"].sum() / per_problem["num_responses"].sum()),
        "mean_per_problem_response_accuracy": float(per_problem["response_accuracy"].mean()),
        "oracle_accuracy_at_n": float(per_problem["oracle_correct"].mean()),
        "top1_accuracy": float(per_problem["top1_correct"].mean()),
        "mrr": float(per_problem["mrr"].mean()),
        "num_problems_with_any_correct": num_with_correct,
        "mean_best_correct_rank_conditional": None if best_correct_rank.empty else float(best_correct_rank.mean()),
        "median_best_correct_rank_conditional": None if best_correct_rank.empty else float(best_correct_rank.median()),
        "mean_pairwise_pref_acc": None if valid_pairwise.empty else float(valid_pairwise.mean()),
        "num_pairwise_evaluable_problems": int(len(valid_pairwise)),
        "mean_per_problem_auroc": None if valid_auroc.empty else float(valid_auroc.mean()),
        "num_auroc_evaluable_problems": int(len(valid_auroc)),
        "mean_per_problem_average_precision": None if valid_ap.empty else float(valid_ap.mean()),
        "num_average_precision_evaluable_problems": int(len(valid_ap)),
    }

    for k in top_ks:
        col = f"rerank_pass_at_{k}"
        if col in per_problem.columns:
            summary[col] = float(per_problem[col].mean())
    return summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "rerank_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = sorted({int(k) for k in args.top_k.split(",") if k.strip()})

    df = load_rows(input_dir)
    rows = [evaluate_problem(key, group, top_ks) for key, group in df.groupby("problem_key", sort=False)]
    per_problem = pd.DataFrame(rows)
    summary = summarize(per_problem, top_ks)

    per_problem.to_csv(output_dir / "likelihood_rerank_per_problem.csv", index=False)
    (output_dir / "likelihood_rerank_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote per-problem metrics to {output_dir / 'likelihood_rerank_per_problem.csv'}")
    print(f"Wrote summary to {output_dir / 'likelihood_rerank_summary.json'}")


if __name__ == "__main__":
    main()
