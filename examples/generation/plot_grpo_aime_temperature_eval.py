#!/usr/bin/env python3
"""Aggregate GRPO AIME fixed-temperature checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from plot_aime_mixed_policy_checkpoint_eval import PASS_KS, load_step, summarize_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--data",
        default="/data/user/zhongal/data/reschedule/aime24.parquet",
        help="Repeated AIME parquet used by the evaluation.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-problems", type=int, default=30)
    parser.add_argument("--expected-samples-per-problem", type=int, default=32)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--title",
        default="GRPO n=32 AIME24 evaluation at fixed sampling temperatures",
    )
    return parser.parse_args()


def discover_settings(root: Path) -> list[tuple[str, int, Path]]:
    settings = []
    for temperature_dir in sorted(root.glob("temperature_*")):
        if not temperature_dir.is_dir():
            continue
        for step_dir in temperature_dir.glob("step_*"):
            try:
                step = int(step_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            settings.append((temperature_dir.name, step, step_dir))
    return sorted(settings, key=lambda item: (item[0], item[1]))


def temperature_label(value: float) -> str:
    return f"T={value:g}"


def plot_summary(summary: pd.DataFrame, output_dir: Path, title: str) -> tuple[Path, Path]:
    metrics = [
        ("trajectory_accuracy", "Accuracy (= pass@1)"),
        ("pass_at_4", "pass@4"),
        ("pass_at_8", "pass@8"),
        ("pass_at_16", "pass@16"),
        ("pass_at_32", "pass@32"),
    ]
    colors = {0.25: "#b51f2e", 1.0: "#2563a6"}
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), sharex=True)
    flat_axes = axes.ravel()
    for axis, (metric, label) in zip(flat_axes, metrics):
        for temperature, frame in summary.groupby("temperature"):
            frame = frame.sort_values("step")
            axis.plot(
                frame["step"],
                frame[metric],
                marker="o",
                linewidth=2.3,
                markersize=5,
                color=colors.get(float(temperature)),
                label=temperature_label(float(temperature)),
            )
        axis.set_title(label)
        axis.set_ylabel("AIME24 score")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(True, alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    max_step = int(summary["step"].max())
    for axis in flat_axes[: len(metrics)]:
        axis.set_xlabel("Checkpoint step")
        axis.set_xticks(range(1, max_step + 1))
    handles, labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[-1].axis("off")
    flat_axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=12)
    fig.suptitle(title, fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = output_dir / "grpo_aime_temperature_acc_passk.png"
    pdf = output_dir / "grpo_aime_temperature_acc_passk.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "analysis"
    data_path = Path(args.data)
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation root does not exist: {root}")
    if not data_path.is_file():
        raise FileNotFoundError(f"AIME parquet does not exist: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_data = pd.read_parquet(data_path).reset_index(names="original_row_index")
    summary_rows: list[dict[str, Any]] = []
    problem_frames = []
    incomplete = []
    for directory_label, step, path in discover_settings(root):
        summary, per_problem = summarize_step(
            step,
            load_step(path),
            args.expected_problems,
            args.expected_samples_per_problem,
            source_data,
        )
        temperature = float(summary["prefix_temperature"])
        if abs(temperature - float(summary["suffix_temperature"])) > 1e-12:
            raise ValueError(f"Setting is not fixed-temperature: {path}")
        if abs(float(summary["trajectory_accuracy"]) - float(summary["pass_at_1"])) > 1e-12:
            raise ValueError(f"Accuracy and pass@1 differ for {path}")
        summary.update(
            {
                "temperature": temperature,
                "temperature_label": temperature_label(temperature),
                "directory_label": directory_label,
                "path": str(path),
            }
        )
        per_problem["temperature"] = temperature
        per_problem["temperature_label"] = temperature_label(temperature)
        summary_rows.append(summary)
        problem_frames.append(per_problem)
        if not summary["complete"]:
            incomplete.append(summary)

    if not summary_rows:
        raise FileNotFoundError(f"No temperature_*/step_* outputs found under {root}")
    if incomplete and not args.allow_incomplete:
        details = ", ".join(
            f"T={row['temperature']:g} step={row['step']} "
            f"problems={row['problems']} groups={row['group_size_min']}-{row['group_size_max']}"
            for row in incomplete
        )
        raise RuntimeError(f"Incomplete AIME evaluation; expected 30x32. {details}")

    summary_frame = pd.DataFrame(summary_rows).sort_values(["temperature", "step"])
    per_problem_frame = pd.concat(problem_frames, ignore_index=True).sort_values(
        ["temperature", "step", "aime_problem_id"]
    )
    summary_csv = output_dir / "grpo_aime_temperature_summary.csv"
    per_problem_csv = output_dir / "grpo_aime_temperature_per_problem.csv"
    summary_frame.to_csv(summary_csv, index=False)
    per_problem_frame.to_csv(per_problem_csv, index=False)
    (output_dir / "grpo_aime_temperature_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    png, pdf = plot_summary(summary_frame, output_dir, args.title)

    display_columns = [
        "temperature",
        "step",
        "trajectory_accuracy",
        "pass_at_1",
        "pass_at_4",
        "pass_at_8",
        "pass_at_16",
        "pass_at_32",
        "complete",
    ]
    print(summary_frame[display_columns].to_string(index=False))
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {per_problem_csv}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
