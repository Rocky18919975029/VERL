#!/usr/bin/env python3
"""Visualize correlation between normalized log likelihood and correctness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Experiment output dir containing shard_*/math500_*_loglik.parquet.",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>/plots.")
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


def load_rows(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("shard_*/math500_qwen25_7b_temp025_n16_loglik.parquet"))
    if not paths:
        direct = input_dir / "math500_qwen25_7b_temp025_n16_loglik.parquet"
        if direct.exists():
            paths = [direct]
    if not paths:
        raise FileNotFoundError(f"No result parquet found under {input_dir}")
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
    return df


def write_summary(df: pd.DataFrame, output_dir: Path, bins: int) -> None:
    corr = df["model_avg_log_likelihood"].corr(df["is_correct"].astype(float), method="pearson")
    spearman = df["model_avg_log_likelihood"].corr(df["is_correct"].astype(float), method="spearman")
    correct = df[df["is_correct"]]
    incorrect = df[~df["is_correct"]]
    summary = {
        "num_responses": int(len(df)),
        "num_problems": int(df["problem_index"].nunique()),
        "num_correct": int(df["is_correct"].sum()),
        "accuracy": float(df["is_correct"].mean()),
        "pearson_corr_likelihood_correct": None if pd.isna(corr) else float(corr),
        "spearman_corr_likelihood_correct": None if pd.isna(spearman) else float(spearman),
        "mean_likelihood_correct": None if correct.empty else float(correct["model_avg_log_likelihood"].mean()),
        "mean_likelihood_incorrect": None if incorrect.empty else float(incorrect["model_avg_log_likelihood"].mean()),
        "median_likelihood_correct": None if correct.empty else float(correct["model_avg_log_likelihood"].median()),
        "median_likelihood_incorrect": None if incorrect.empty else float(incorrect["model_avg_log_likelihood"].median()),
        "bins": bins,
    }
    (output_dir / "likelihood_correctness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def plot_box(df: pd.DataFrame, output_dir: Path) -> None:
    data = [
        df.loc[~df["is_correct"], "model_avg_log_likelihood"].to_numpy(),
        df.loc[df["is_correct"], "model_avg_log_likelihood"].to_numpy(),
    ]
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=["Incorrect", "Correct"], showmeans=True)
    plt.ylabel("Response-length-normalized model log likelihood")
    plt.title("Math-500: normalized log likelihood by correctness")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "likelihood_by_correctness_box.png", dpi=200)
    plt.close()


def plot_hist(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(
        df.loc[~df["is_correct"], "model_avg_log_likelihood"],
        bins=40,
        alpha=0.65,
        label="Incorrect",
        density=True,
    )
    plt.hist(
        df.loc[df["is_correct"], "model_avg_log_likelihood"],
        bins=40,
        alpha=0.65,
        label="Correct",
        density=True,
    )
    plt.xlabel("Response-length-normalized model log likelihood")
    plt.ylabel("Density")
    plt.title("Math-500 likelihood distributions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "likelihood_correctness_hist.png", dpi=200)
    plt.close()


def plot_binned_accuracy(df: pd.DataFrame, output_dir: Path, bins: int) -> None:
    ranked = df.sort_values("model_avg_log_likelihood").copy()
    ranked["likelihood_bin"] = pd.qcut(
        ranked["model_avg_log_likelihood"], q=min(bins, len(ranked)), duplicates="drop"
    )
    grouped = (
        ranked.groupby("likelihood_bin", observed=True)
        .agg(
            mean_likelihood=("model_avg_log_likelihood", "mean"),
            accuracy=("is_correct", "mean"),
            count=("is_correct", "size"),
        )
        .reset_index(drop=True)
    )
    grouped.to_csv(output_dir / "likelihood_binned_accuracy.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["mean_likelihood"], grouped["accuracy"], marker="o")
    for _, row in grouped.iterrows():
        plt.annotate(str(int(row["count"])), (row["mean_likelihood"], row["accuracy"]), fontsize=7, alpha=0.7)
    plt.xlabel("Mean normalized log likelihood in bin")
    plt.ylabel("Accuracy")
    plt.title("Math-500 correctness rate by likelihood bin")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "likelihood_binned_accuracy.png", dpi=200)
    plt.close()


def plot_problem_level(df: pd.DataFrame, output_dir: Path) -> None:
    problem = (
        df.groupby("problem_index")
        .agg(
            mean_likelihood=("model_avg_log_likelihood", "mean"),
            best_likelihood=("model_avg_log_likelihood", "max"),
            any_correct=("is_correct", "max"),
            accuracy=("is_correct", "mean"),
            n=("is_correct", "size"),
        )
        .reset_index()
    )
    problem.to_csv(output_dir / "problem_level_likelihood_correctness.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(problem["mean_likelihood"], problem["accuracy"], alpha=0.65, s=22)
    plt.xlabel("Mean normalized log likelihood over 16 responses")
    plt.ylabel("Problem-level accuracy over 16 responses")
    plt.title("Math-500 problem-level likelihood vs correctness")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "problem_level_likelihood_accuracy.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(input_dir)
    df.to_parquet(output_dir / "combined_math500_loglik_correctness.parquet", index=False)
    write_summary(df, output_dir, args.bins)
    plot_box(df, output_dir)
    plot_hist(df, output_dir)
    plot_binned_accuracy(df, output_dir, args.bins)
    plot_problem_level(df, output_dir)
    print(f"Wrote plots and summaries to {output_dir}")


if __name__ == "__main__":
    main()
