#!/usr/bin/env python3
"""Validate and plot mixed-policy AIME checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PASS_KS = (1, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-problems", type=int, default=30)
    parser.add_argument("--expected-samples-per-problem", type=int, default=32)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--title", default="Lambda=0.5 AIME24 mixed-policy checkpoint evaluation"
    )
    return parser.parse_args()


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_correct <= 0:
        return 0.0
    if k >= num_samples:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def discover_steps(root: Path) -> list[tuple[int, Path]]:
    candidates = list((root / "lambda0p5").glob("step_*"))
    if not candidates:
        candidates = list(root.glob("step_*"))
    steps = []
    for path in candidates:
        try:
            steps.append((int(path.name.split("_", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def load_step(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("shard_*/*_mixed_step*_shard*.parquet"))
    if not files:
        files = sorted(path.glob("shard_*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No shard parquet outputs found under {path}")
    frames = []
    for file in files:
        frame = pd.read_parquet(file)
        frame["source_file"] = str(file)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def infer_problem_groups(
    frame: pd.DataFrame,
    expected_problems: int,
    expected_samples: int,
) -> tuple[pd.Series, str]:
    """Recover dataset-level problem identity without merging duplicate items."""

    def valid(candidate: pd.Series) -> bool:
        grouped = frame.assign(_candidate=candidate).groupby("_candidate", sort=False)
        sizes = grouped.size()
        content_counts = grouped["problem_key"].nunique()
        return bool(
            len(sizes) == expected_problems
            and sizes.eq(expected_samples).all()
            and content_counts.eq(1).all()
        )

    content_groups = frame["problem_key"].astype(str)
    if valid(content_groups):
        return content_groups, "prompt_and_ground_truth"

    if "original_row_index" not in frame.columns:
        raise ValueError(
            "Content grouping merged duplicate AIME items, but original_row_index is unavailable."
        )
    row_index = frame["original_row_index"].astype(int)
    if row_index.nunique() != expected_problems * expected_samples:
        raise ValueError(
            "original_row_index is not unique and complete: "
            f"unique={row_index.nunique()} expected={expected_problems * expected_samples}"
        )

    # Common construction 1: each problem is repeated N times before the next
    # problem. Common construction 2: the complete problem set is repeated N
    # times. Requiring one content key per inferred group prevents a silent,
    # incorrect choice between the two layouts.
    contiguous = row_index // expected_samples
    if valid(contiguous):
        return "row_block_" + contiguous.astype(str), "contiguous_repetitions"

    strided = row_index % expected_problems
    if valid(strided):
        return "row_stride_" + strided.astype(str), "repeated_problem_set"

    raise ValueError(
        "Could not infer the AIME repetition layout. Neither content, contiguous-repeat, "
        "nor repeated-problem-set grouping produced the expected problem structure."
    )


def summarize_step(
    step: int,
    frame: pd.DataFrame,
    expected_problems: int,
    expected_samples: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"problem_key", "is_correct", "horizon"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Step {step} is missing columns: {sorted(missing)}")
    problem_groups, grouping_strategy = infer_problem_groups(
        frame, expected_problems, expected_samples
    )
    grouped_frame = frame.assign(aime_problem_id=problem_groups)
    per_problem = (
        grouped_frame.groupby("aime_problem_id", sort=True)
        .agg(
            problem_key=("problem_key", "first"),
            num_correct=("is_correct", "sum"),
            num_samples=("is_correct", "count"),
        )
        .reset_index()
    )
    per_problem["num_correct"] = per_problem["num_correct"].astype(int)
    per_problem["num_samples"] = per_problem["num_samples"].astype(int)
    per_problem["step"] = step
    for k in PASS_KS:
        per_problem[f"pass_at_{k}"] = per_problem.apply(
            lambda row: pass_at_k(
                int(row["num_samples"]), int(row["num_correct"]), k
            ),
            axis=1,
        )

    group_min = int(per_problem["num_samples"].min()) if len(per_problem) else 0
    group_max = int(per_problem["num_samples"].max()) if len(per_problem) else 0
    complete = (
        len(per_problem) == expected_problems
        and group_min == expected_samples
        and group_max == expected_samples
    )
    summary: dict[str, Any] = {
        "step": step,
        "horizon": int(frame["horizon"].iloc[0]),
        "rows": len(frame),
        "problems": len(per_problem),
        "group_size_min": group_min,
        "group_size_max": group_max,
        "grouping_strategy": grouping_strategy,
        "complete": complete,
        "trajectory_accuracy": float(frame["is_correct"].mean()),
        "num_correct": int(frame["is_correct"].sum()),
        "suffix_used_frac": float(frame["used_low_temperature_suffix"].mean()),
        "mean_prefix_tokens": float(frame["prefix_token_count"].mean()),
        "mean_suffix_tokens": float(frame["suffix_token_count"].mean()),
        "prefix_temperature": float(frame["prefix_temperature"].iloc[0]),
        "prefix_top_p": float(frame["prefix_top_p"].iloc[0]),
        "suffix_temperature": float(frame["suffix_temperature"].iloc[0]),
        "suffix_top_p": float(frame["suffix_top_p"].iloc[0]),
        "seed": int(frame["seed"].iloc[0]),
        "engine_seed_min": int(frame["engine_seed"].min()),
        "engine_seed_max": int(frame["engine_seed"].max()),
        "shard_files": int(frame["source_file"].nunique()),
    }
    for k in PASS_KS:
        summary[f"pass_at_{k}"] = float(per_problem[f"pass_at_{k}"].mean())
    return summary, per_problem


def plot_summary(summary: pd.DataFrame, output_dir: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    axes[0].plot(
        summary["step"], summary["trajectory_accuracy"], marker="o", linewidth=2.3
    )
    axes[0].set_title("Trajectory accuracy")
    axes[0].set_ylabel("Accuracy")
    for k in PASS_KS[:-1]:
        axes[1].plot(
            summary["step"],
            summary[f"pass_at_{k}"],
            marker="o",
            linewidth=2.0,
            label=f"pass@{k}",
        )
    axes[1].set_title("Unbiased pass@k")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.set_xlabel("Checkpoint step")
        axis.grid(True, alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylim(bottom=0)
    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "aime_mixed_policy_eval.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "aime_mixed_policy_eval.pdf", bbox_inches="tight")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    problem_frames = []
    incomplete = []
    for step, path in discover_steps(root):
        summary, per_problem = summarize_step(
            step,
            load_step(path),
            args.expected_problems,
            args.expected_samples_per_problem,
        )
        summaries.append(summary)
        problem_frames.append(per_problem)
        if not summary["complete"]:
            incomplete.append(summary)
    if not summaries:
        raise FileNotFoundError(f"No step_* mixed-policy eval outputs found under {root}")
    if incomplete and not args.allow_incomplete:
        details = ", ".join(
            f"step {row['step']}: problems={row['problems']} groups={row['group_size_min']}-{row['group_size_max']}"
            for row in incomplete
        )
        raise RuntimeError(f"Incomplete AIME evaluation; expected 30x32. {details}")

    summary_frame = pd.DataFrame(summaries).sort_values("step")
    per_problem_frame = pd.concat(problem_frames, ignore_index=True).sort_values(
        ["step", "aime_problem_id"]
    )
    summary_frame.to_csv(output_dir / "aime_mixed_policy_summary.csv", index=False)
    per_problem_frame.to_csv(output_dir / "aime_mixed_policy_per_problem.csv", index=False)
    (output_dir / "aime_mixed_policy_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_summary(summary_frame, output_dir, args.title)
    print(summary_frame.to_string(index=False))
    print(f"\nWrote {output_dir / 'aime_mixed_policy_summary.csv'}")
    print(f"Wrote {output_dir / 'aime_mixed_policy_per_problem.csv'}")
    print(f"Wrote {output_dir / 'aime_mixed_policy_eval.png'}")


if __name__ == "__main__":
    main()
