#!/usr/bin/env python3
"""Compare HPF and GRPO train/eval metrics over matching training steps."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


EVAL_ACC_KEY = "val-core/aime_2024_dapo_boxed/acc/mean@1"
EVAL_REWARD_KEY = "val-aux/aime_2024_dapo_boxed/reward/mean@1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpf-rollout-dir", required=True, type=Path)
    parser.add_argument("--grpo-rollout-dir", required=True, type=Path)
    parser.add_argument("--hpf-wandb-run", required=True, help="Full W&B path, e.g. entity/project/run_id.")
    parser.add_argument("--grpo-wandb-run", required=True, help="Full W&B path, e.g. entity/project/run_id.")
    parser.add_argument("--hpf-label", default="HPF 4x4")
    parser.add_argument("--grpo-label", default="GRPO n16")
    parser.add_argument("--max-step", type=int, default=33)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hpf_grpo_epoch_comparison"))
    parser.add_argument("--title", default="HPF vs GRPO baseline over the first epoch")
    return parser.parse_args()


def response_text(row: dict[str, Any]) -> str:
    for key in ("output", "response", "full_response", "text"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rollout_step_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def summarize_rollout_dir(rollout_dir: Path, label: str, max_step: int) -> pd.DataFrame:
    records = []
    for path in sorted(rollout_dir.glob("*.jsonl"), key=lambda p: rollout_step_from_path(p) or math.inf):
        step = rollout_step_from_path(path)
        if step is None or step > max_step:
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        acc_values = [float(row.get("acc", row.get("is_correct", 0.0)) or 0.0) for row in rows]
        score_values = [float(row.get("score", row.get("reward", 0.0)) or 0.0) for row in rows]
        texts = [response_text(row) for row in rows]
        word_lengths = [len(text.split()) for text in texts]
        records.append(
            {
                "run": label,
                "step": step,
                "train_rows": len(rows),
                "train_acc": sum(acc_values) / len(acc_values),
                "train_score": sum(score_values) / len(score_values),
                "boxed_frac": sum("\\boxed" in text for text in texts) / len(texts),
                "mean_words": sum(word_lengths) / len(word_lengths),
            }
        )
    return pd.DataFrame.from_records(records)


def fetch_wandb_eval(run_path: str, label: str, max_step: int) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    history = run.history(keys=["_step", EVAL_ACC_KEY, EVAL_REWARD_KEY], pandas=True)
    if history.empty:
        return pd.DataFrame(columns=["run", "step", "eval_acc", "eval_reward"])
    history = history.rename(columns={"_step": "step", EVAL_ACC_KEY: "eval_acc", EVAL_REWARD_KEY: "eval_reward"})
    history = history[["step", "eval_acc", "eval_reward"]].copy()
    history["step"] = history["step"].astype(int)
    history = history[(history["step"] >= 1) & (history["step"] <= max_step)]
    history = history.drop_duplicates(subset=["step"], keep="last")
    history["run"] = label
    return history[["run", "step", "eval_acc", "eval_reward"]]


def plot_comparison(df: pd.DataFrame, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"HPF 4x4": "#1f77b4", "GRPO n16": "#d62728"}
    metrics = [
        ("train_acc", "Train rollout accuracy"),
        ("eval_acc", "AIME24 eval accuracy"),
        ("train_score", "Train rollout score"),
        ("eval_reward", "AIME24 eval reward"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (metric, ylabel) in zip(axes.flat, metrics, strict=True):
        for run, sub in df.groupby("run", sort=False):
            sub = sub.sort_values("step")
            color = colors.get(run)
            ax.plot(sub["step"], sub[metric], marker="o", markersize=3.5, linewidth=2.0, label=run, color=color)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.28)
    axes[1, 0].set_xlabel("Global step")
    axes[1, 1].set_xlabel("Global step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.concat(
        [
            summarize_rollout_dir(args.hpf_rollout_dir, args.hpf_label, args.max_step),
            summarize_rollout_dir(args.grpo_rollout_dir, args.grpo_label, args.max_step),
        ],
        ignore_index=True,
    )
    eval_df = pd.concat(
        [
            fetch_wandb_eval(args.hpf_wandb_run, args.hpf_label, args.max_step),
            fetch_wandb_eval(args.grpo_wandb_run, args.grpo_label, args.max_step),
        ],
        ignore_index=True,
    )
    merged = train.merge(eval_df, on=["run", "step"], how="outer").sort_values(["step", "run"])
    summary = (
        merged.groupby("run", sort=False)[
            ["train_acc", "train_score", "eval_acc", "eval_reward", "boxed_frac", "mean_words"]
        ]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )

    per_step_path = args.output_dir / "hpf_grpo_per_step_metrics.csv"
    summary_path = args.output_dir / "hpf_grpo_summary_metrics.csv"
    plot_path = args.output_dir / "hpf_grpo_first_epoch_comparison.png"
    merged.to_csv(per_step_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_comparison(merged, plot_path, args.title)

    print("Per-step metrics:")
    print(merged.to_string(index=False))
    print(f"\nWrote per-step CSV to {per_step_path}")
    print(f"Wrote summary CSV to {summary_path}")
    print(f"Wrote figure to {plot_path}")


if __name__ == "__main__":
    main()
