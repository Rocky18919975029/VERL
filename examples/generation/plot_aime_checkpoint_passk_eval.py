#!/usr/bin/env python3
"""Aggregate AIME checkpoint eval rollouts and plot pass@k curves."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PASS_KS = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Eval root produced by submit_aime_checkpoint_passk_eval.sh.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/analysis.")
    parser.add_argument("--title", default="AIME24 pass@k by checkpoint step")
    parser.add_argument("--expected-problems", type=int, default=30)
    parser.add_argument("--expected-responses-per-problem", type=int, default=16)
    return parser.parse_args()


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples <= 0 or num_correct <= 0:
        return 0.0
    if k >= num_samples:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def load_setting(setting_dir: Path) -> pd.DataFrame:
    paths = sorted(setting_dir.glob("shard_*/*_loglik.parquet"))
    if not paths:
        paths = sorted(setting_dir.glob("*_loglik.parquet"))
    if not paths:
        raise FileNotFoundError(f"No *_loglik.parquet files found under {setting_dir}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def discover_settings(root: Path) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir() or method_dir.name in {"analysis", "cache"}:
            continue
        for step_dir in sorted(method_dir.glob("step_*")):
            try:
                step = int(step_dir.name.split("_", 1)[1])
            except Exception:
                continue
            settings.append({"method": method_dir.name, "step": step, "path": step_dir})
    return settings


def compute_setting_metrics(
    method: str,
    step: int,
    df: pd.DataFrame,
    expected_problems: int,
    expected_responses_per_problem: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if "problem_index" not in df.columns:
        raise ValueError("Expected column problem_index in rollout parquet.")
    if "is_correct" not in df.columns:
        raise ValueError("Expected column is_correct in rollout parquet.")

    grouped = df.groupby("problem_index")["is_correct"].agg(["sum", "count"]).reset_index()
    grouped["method"] = method
    grouped["step"] = step
    grouped = grouped.rename(columns={"sum": "num_correct", "count": "num_samples"})
    grouped["num_correct"] = grouped["num_correct"].astype(int)
    grouped["num_samples"] = grouped["num_samples"].astype(int)
    for k in PASS_KS:
        grouped[f"pass_at_{k}"] = grouped.apply(
            lambda row: pass_at_k(int(row["num_samples"]), int(row["num_correct"]), k),
            axis=1,
        )

    metrics: dict[str, Any] = {
        "method": method,
        "step": step,
        "files": int(df["source_file"].nunique()) if "source_file" in df.columns else 1,
        "problems": int(grouped["problem_index"].nunique()),
        "responses": int(len(df)),
        "group_size_min": int(grouped["num_samples"].min()) if len(grouped) else 0,
        "group_size_max": int(grouped["num_samples"].max()) if len(grouped) else 0,
        "trajectory_accuracy": float(df["is_correct"].mean()) if len(df) else 0.0,
        "num_correct": int(df["is_correct"].sum()),
        "complete": bool(
            int(grouped["problem_index"].nunique()) == expected_problems
            and int(grouped["num_samples"].min()) == expected_responses_per_problem
            and int(grouped["num_samples"].max()) == expected_responses_per_problem
        )
        if len(grouped)
        else False,
    }
    for k in PASS_KS:
        metrics[f"pass_at_{k}"] = float(grouped[f"pass_at_{k}"].mean()) if len(grouped) else 0.0
    for optional in ("sampling_temperature", "sampling_top_p"):
        if optional in df.columns:
            values = sorted({float(x) for x in df[optional].dropna().unique()})
            metrics[optional] = values
    return metrics, grouped


def prettify_method(method: str) -> str:
    mapping = {
        "grpo": "GRPO n16",
        "hpf_alg3": "HPF Alg3 4x4",
        "hpf": "HPF Alg3 4x4",
    }
    return mapping.get(method, method)


def plot_metrics(summary: pd.DataFrame, out_dir: Path, title: str) -> None:
    colors = {"grpo": "#1f77b4", "hpf_alg3": "#d62728", "hpf": "#d62728"}
    fig, axes = plt.subplots(1, len(PASS_KS), figsize=(18, 3.6), sharex=True)
    if len(PASS_KS) == 1:
        axes = [axes]

    for ax, k in zip(axes, PASS_KS):
        metric = f"pass_at_{k}"
        for method, frame in summary.groupby("method"):
            frame = frame.sort_values("step")
            ax.plot(
                frame["step"],
                frame[metric],
                marker="o",
                linewidth=2.2,
                markersize=5,
                label=prettify_method(str(method)),
                color=colors.get(str(method)),
            )
        ax.set_title(f"pass@{k}")
        ax.set_xlabel("Checkpoint step")
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("AIME24 pass@k")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(title, y=1.2, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "aime_passk_by_checkpoint.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "aime_passk_by_checkpoint.pdf", bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    for method, frame in summary.groupby("method"):
        frame = frame.sort_values("step")
        ax2.plot(
            frame["step"],
            frame["trajectory_accuracy"],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=prettify_method(str(method)),
            color=colors.get(str(method)),
        )
    ax2.set_xlabel("Checkpoint step")
    ax2.set_ylabel("Trajectory accuracy")
    ax2.grid(True, alpha=0.22)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(out_dir / "aime_trajectory_accuracy_by_checkpoint.png", dpi=240, bbox_inches="tight")
    fig2.savefig(out_dir / "aime_trajectory_accuracy_by_checkpoint.pdf", bbox_inches="tight")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    problem_frames: list[pd.DataFrame] = []
    for setting in discover_settings(root):
        df = load_setting(setting["path"])
        metrics, per_problem = compute_setting_metrics(
            method=setting["method"],
            step=setting["step"],
            df=df,
            expected_problems=args.expected_problems,
            expected_responses_per_problem=args.expected_responses_per_problem,
        )
        metrics["path"] = str(setting["path"])
        summary_rows.append(metrics)
        problem_frames.append(per_problem)

    if not summary_rows:
        raise FileNotFoundError(f"No setting directories found under {root}")

    summary = pd.DataFrame(summary_rows).sort_values(["method", "step"])
    per_problem_df = pd.concat(problem_frames, ignore_index=True).sort_values(["method", "step", "problem_index"])
    summary_csv = out_dir / "aime_passk_summary.csv"
    per_problem_csv = out_dir / "aime_passk_per_problem.csv"
    summary.to_csv(summary_csv, index=False)
    per_problem_df.to_csv(per_problem_csv, index=False)
    (out_dir / "aime_passk_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_metrics(summary, out_dir, args.title)

    print(summary.to_string(index=False))
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {per_problem_csv}")
    print(f"Wrote {out_dir / 'aime_passk_by_checkpoint.png'}")
    print(f"Wrote {out_dir / 'aime_trajectory_accuracy_by_checkpoint.png'}")


if __name__ == "__main__":
    main()
