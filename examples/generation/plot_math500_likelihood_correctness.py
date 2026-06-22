#!/usr/bin/env python3
"""Visualize correlation between normalized log likelihood and correctness."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


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
    paths = sorted(input_dir.glob("shard_*/*_loglik.parquet"))
    if not paths:
        paths = sorted(input_dir.glob("*_loglik.parquet"))
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
    if "raw_problem" in df.columns and df["raw_problem"].notna().any():
        df["problem_key"] = df["raw_problem"].astype(str)
    else:
        df["problem_key"] = df["problem_index"].astype(str)
    return df


def write_summary(df: pd.DataFrame, output_dir: Path, bins: int) -> None:
    corr = df["model_avg_log_likelihood"].corr(df["is_correct"].astype(float), method="pearson")
    spearman = df["model_avg_log_likelihood"].corr(df["is_correct"].astype(float), method="spearman")
    correct = df[df["is_correct"]]
    incorrect = df[~df["is_correct"]]
    top_by_likelihood = (
        df.sort_values(["problem_key", "model_avg_log_likelihood"], ascending=[True, False])
        .groupby("problem_key", as_index=False)
        .first()
    )
    per_problem = (
        df.groupby("problem_key")
        .agg(
            problem_index=("problem_index", "first"),
            raw_problem=("raw_problem", "first") if "raw_problem" in df.columns else ("problem_key", "first"),
            ground_truth=("ground_truth", "first") if "ground_truth" in df.columns else ("problem_key", "first"),
            num_responses=("is_correct", "size"),
            num_correct=("is_correct", "sum"),
            oracle_correct=("is_correct", "max"),
            response_accuracy=("is_correct", "mean"),
            mean_likelihood=("model_avg_log_likelihood", "mean"),
            max_likelihood=("model_avg_log_likelihood", "max"),
        )
        .reset_index()
    )
    top_cols = [
        "problem_key",
        "sample_index",
        "response",
        "model_avg_log_likelihood",
        "model_log_likelihood",
        "response_token_len",
        "is_correct",
        "extracted_answer",
    ]
    available_top_cols = [col for col in top_cols if col in top_by_likelihood.columns]
    top_export = top_by_likelihood[available_top_cols].rename(
        columns={
            "sample_index": "top_likelihood_sample_index",
            "response": "top_likelihood_response",
            "model_avg_log_likelihood": "top_likelihood_avg_log_likelihood",
            "model_log_likelihood": "top_likelihood_log_likelihood",
            "response_token_len": "top_likelihood_response_token_len",
            "is_correct": "top_likelihood_is_correct",
            "extracted_answer": "top_likelihood_extracted_answer",
        }
    )
    per_problem = per_problem.merge(top_export, on="problem_key", how="left")
    per_problem.to_csv(output_dir / "top_likelihood_per_problem.csv", index=False)

    top1_acc = float(top_by_likelihood["is_correct"].mean()) if not top_by_likelihood.empty else None
    oracle_acc = float(per_problem["oracle_correct"].mean()) if not per_problem.empty else None
    mean_response_acc_per_problem = float(per_problem["response_accuracy"].mean()) if not per_problem.empty else None

    summary = {
        "num_responses": int(len(df)),
        "num_problems": int(df["problem_key"].nunique()),
        "num_correct": int(df["is_correct"].sum()),
        "accuracy": float(df["is_correct"].mean()),
        "top_likelihood_accuracy": top1_acc,
        "oracle_accuracy_at_16": oracle_acc,
        "mean_per_problem_response_accuracy": mean_response_acc_per_problem,
        "top_likelihood_num_correct": int(top_by_likelihood["is_correct"].sum()) if not top_by_likelihood.empty else 0,
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


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
    ]


def svg_axes(parts: list[str], width: int, height: int, xlabel: str, ylabel: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = 70, 50, width - 25, height - 55
    parts.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
            f'<text x="{(left + right) / 2}" y="{height - 15}" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(xlabel)}</text>',
            f'<text x="18" y="{(top + bottom) / 2}" text-anchor="middle" transform="rotate(-90 18 {(top + bottom) / 2})" font-family="Arial" font-size="12">{html.escape(ylabel)}</text>',
        ]
    )
    return left, top, right, bottom


def write_line_svg(path: Path, x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str) -> None:
    width, height = 850, 520
    parts = svg_header(width, height, title)
    left, top, right, bottom = svg_axes(parts, width, height, xlabel, ylabel)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = min(0.0, float(y.min())), max(1.0, float(y.max()))
    points = []
    for xv, yv in zip(x, y, strict=True):
        px = scale(float(xv), x_min, x_max, left, right)
        py = scale(float(yv), y_min, y_max, bottom, top)
        points.append((px, py))
    point_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    parts.append(f'<polyline points="{point_str}" fill="none" stroke="#2563eb" stroke-width="2"/>')
    for px, py in points:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#2563eb"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_scatter_svg(path: Path, x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str) -> None:
    width, height = 850, 520
    parts = svg_header(width, height, title)
    left, top, right, bottom = svg_axes(parts, width, height, xlabel, ylabel)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = min(0.0, float(y.min())), max(1.0, float(y.max()))
    for xv, yv in zip(x, y, strict=True):
        px = scale(float(xv), x_min, x_max, left, right)
        py = scale(float(yv), y_min, y_max, bottom, top)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#16a34a" opacity="0.55"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_hist_svg(path: Path, incorrect: pd.Series, correct: pd.Series) -> None:
    width, height = 850, 520
    parts = svg_header(width, height, "Math-500 likelihood distributions")
    left, top, right, bottom = svg_axes(
        parts, width, height, "Response-length-normalized model log likelihood", "Count"
    )
    all_values = pd.concat([incorrect, correct])
    counts_i, edges = np.histogram(incorrect, bins=40, range=(float(all_values.min()), float(all_values.max())))
    counts_c, _ = np.histogram(correct, bins=edges)
    max_count = max(int(counts_i.max()), int(counts_c.max()), 1)
    bar_w = (right - left) / len(counts_i)
    for idx, count in enumerate(counts_i):
        h = scale(float(count), 0, max_count, 0, bottom - top)
        x = left + idx * bar_w
        parts.append(f'<rect x="{x:.1f}" y="{bottom - h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#ef4444" opacity="0.45"/>')
    for idx, count in enumerate(counts_c):
        h = scale(float(count), 0, max_count, 0, bottom - top)
        x = left + idx * bar_w
        parts.append(f'<rect x="{x:.1f}" y="{bottom - h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2563eb" opacity="0.45"/>')
    parts.append('<text x="650" y="72" font-family="Arial" font-size="12" fill="#ef4444">Incorrect</text>')
    parts.append('<text x="650" y="92" font-family="Arial" font-size="12" fill="#2563eb">Correct</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_box_svg(path: Path, df: pd.DataFrame) -> None:
    width, height = 760, 480
    parts = svg_header(width, height, "Math-500: normalized log likelihood by correctness")
    left, top, right, bottom = svg_axes(
        parts, width, height, "Group", "Response-length-normalized model log likelihood"
    )
    groups = [("Incorrect", df.loc[~df["is_correct"], "model_avg_log_likelihood"]), ("Correct", df.loc[df["is_correct"], "model_avg_log_likelihood"])]
    all_values = df["model_avg_log_likelihood"]
    y_min, y_max = float(all_values.min()), float(all_values.max())
    for idx, (label, values) in enumerate(groups):
        if values.empty:
            continue
        q1, med, q3 = values.quantile([0.25, 0.5, 0.75]).tolist()
        vmin, vmax = float(values.min()), float(values.max())
        cx = left + (idx + 1) * (right - left) / 3
        box_w = 95
        yq1 = scale(float(q1), y_min, y_max, bottom, top)
        yq3 = scale(float(q3), y_min, y_max, bottom, top)
        ymed = scale(float(med), y_min, y_max, bottom, top)
        ymin = scale(vmin, y_min, y_max, bottom, top)
        ymax = scale(vmax, y_min, y_max, bottom, top)
        parts.append(f'<line x1="{cx:.1f}" y1="{ymax:.1f}" x2="{cx:.1f}" y2="{ymin:.1f}" stroke="#333"/>')
        parts.append(f'<rect x="{cx - box_w / 2:.1f}" y="{yq3:.1f}" width="{box_w}" height="{max(1, yq1 - yq3):.1f}" fill="#bfdbfe" stroke="#2563eb"/>')
        parts.append(f'<line x1="{cx - box_w / 2:.1f}" y1="{ymed:.1f}" x2="{cx + box_w / 2:.1f}" y2="{ymed:.1f}" stroke="#1d4ed8" stroke-width="2"/>')
        parts.append(f'<text x="{cx:.1f}" y="{bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_box(df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        write_box_svg(output_dir / "likelihood_by_correctness_box.svg", df)
        return
    data = [
        df.loc[~df["is_correct"], "model_avg_log_likelihood"].to_numpy(),
        df.loc[df["is_correct"], "model_avg_log_likelihood"].to_numpy(),
    ]
    plt.figure(figsize=(7, 5))
    try:
        plt.boxplot(data, tick_labels=["Incorrect", "Correct"], showmeans=True)
    except TypeError:
        plt.boxplot(data, labels=["Incorrect", "Correct"], showmeans=True)
    plt.ylabel("Response-length-normalized model log likelihood")
    plt.title("Math-500: normalized log likelihood by correctness")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "likelihood_by_correctness_box.png", dpi=200)
    plt.close()


def plot_hist(df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        write_hist_svg(
            output_dir / "likelihood_correctness_hist.svg",
            df.loc[~df["is_correct"], "model_avg_log_likelihood"],
            df.loc[df["is_correct"], "model_avg_log_likelihood"],
        )
        return
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

    if plt is None:
        write_line_svg(
            output_dir / "likelihood_binned_accuracy.svg",
            grouped["mean_likelihood"],
            grouped["accuracy"],
            "Math-500 correctness rate by likelihood bin",
            "Mean normalized log likelihood in bin",
            "Accuracy",
        )
        return

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
        df.groupby("problem_key")
        .agg(
            problem_index=("problem_index", "first"),
            raw_problem=("raw_problem", "first") if "raw_problem" in df.columns else ("problem_key", "first"),
            mean_likelihood=("model_avg_log_likelihood", "mean"),
            best_likelihood=("model_avg_log_likelihood", "max"),
            any_correct=("is_correct", "max"),
            accuracy=("is_correct", "mean"),
            n=("is_correct", "size"),
        )
        .reset_index()
    )
    problem.to_csv(output_dir / "problem_level_likelihood_correctness.csv", index=False)

    if plt is None:
        write_scatter_svg(
            output_dir / "problem_level_likelihood_accuracy.svg",
            problem["mean_likelihood"],
            problem["accuracy"],
            "Math-500 problem-level likelihood vs correctness",
            "Mean normalized log likelihood over 16 responses",
            "Problem-level accuracy over 16 responses",
        )
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(problem["mean_likelihood"], problem["accuracy"], alpha=0.65, s=22)
    plt.xlabel("Mean normalized log likelihood over 16 responses")
    plt.ylabel("Problem-level accuracy over 16 responses")
    plt.title("Math-500 problem-level likelihood vs correctness")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "problem_level_likelihood_accuracy.png", dpi=200)
    plt.close()


def plot_top_likelihood_selection(output_dir: Path) -> None:
    path = output_dir / "top_likelihood_per_problem.csv"
    if not path.exists():
        return
    top = pd.read_csv(path)
    if "top_likelihood_avg_log_likelihood" not in top.columns or "top_likelihood_is_correct" not in top.columns:
        return
    top["top_likelihood_is_correct"] = top["top_likelihood_is_correct"].astype(bool)

    if plt is None:
        write_scatter_svg(
            output_dir / "top_likelihood_selection.svg",
            top["top_likelihood_avg_log_likelihood"],
            top["top_likelihood_is_correct"].astype(float),
            "Top-likelihood response correctness per problem",
            "Top response normalized log likelihood",
            "Correct",
        )
        return

    plt.figure(figsize=(8, 4.8))
    y = top["top_likelihood_is_correct"].astype(float)
    plt.scatter(top["top_likelihood_avg_log_likelihood"], y, alpha=0.6, s=20)
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.xlabel("Top response normalized log likelihood")
    plt.ylabel("Top-likelihood response correctness")
    plt.title("Math-500: whether top-likelihood response is correct")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "top_likelihood_selection.png", dpi=200)
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
    plot_top_likelihood_selection(output_dir)
    print(f"Wrote plots and summaries to {output_dir}")


if __name__ == "__main__":
    main()
