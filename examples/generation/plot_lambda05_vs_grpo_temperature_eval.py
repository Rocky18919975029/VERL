#!/usr/bin/env python3
"""Compare lambda=0.5 mixed-policy AIME eval with fixed-temperature GRPO evals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from plot_aime_mixed_policy_checkpoint_eval import load_step, summarize_step
from plot_grpo_aime_temperature_eval import discover_settings, temperature_label


SERIES_ORDER = ["lambda=0.5 mixed policy", "GRPO T=1", "GRPO T=0.25"]
SERIES_COLORS = {
    "lambda=0.5 mixed policy": "#b51f2e",
    "GRPO T=1": "#2563a6",
    "GRPO T=0.25": "#238b45",
}
PLOT_METRICS = [
    ("trajectory_accuracy", "Accuracy (= pass@1)"),
    ("pass_at_4", "pass@4"),
    ("pass_at_8", "pass@8"),
    ("pass_at_16", "pass@16"),
    ("pass_at_32", "pass@32"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo-root", type=Path)
    parser.add_argument("--lambda05-summary", type=Path)
    parser.add_argument("--search-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/data/user/zhongal/data/reschedule/aime24.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lambda0p5_vs_grpo_temperature_eval"),
    )
    parser.add_argument("--max-step", type=int)
    parser.add_argument(
        "--title",
        default="AIME24: transition-aware lambda=0.5 vs fixed-temperature GRPO",
    )
    return parser.parse_args()


def discover_latest_grpo_root(search_dir: Path) -> Path | None:
    manifests = list(
        search_dir.glob(
            "grpo_n32_aime_temp*/submitted_grpo_aime_temperature_eval_jobs.tsv"
        )
    )
    return max(manifests, key=lambda path: path.stat().st_mtime).parent if manifests else None


def summary_rank(path: Path) -> tuple[int, int, float]:
    try:
        frame = pd.read_csv(path, usecols=["step", "pass_at_1"])
        steps = pd.to_numeric(frame["step"], errors="coerce").dropna().astype(int)
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return (0, 0, path.stat().st_mtime)
    return (len(set(steps)), int(steps.max()) if len(steps) else 0, path.stat().st_mtime)


def discover_lambda05_summary(search_dir: Path) -> Path | None:
    candidates = list(
        search_dir.glob(
            "lambda0p5_aime_mixed_policy_eval*/analysis/aime_mixed_policy_summary.csv"
        )
    )
    return max(candidates, key=summary_rank) if candidates else None


def resolve_path(explicit: Path | None, discovered: Path | None, label: str) -> Path:
    path = explicit if explicit is not None else discovered
    if path is None or not path.exists():
        raise FileNotFoundError(f"Could not locate {label}: {path}")
    return path


def as_complete_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_lambda05(summary_path: Path, max_step: int | None) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    required = {
        "step",
        "trajectory_accuracy",
        "pass_at_1",
        "pass_at_4",
        "pass_at_8",
        "pass_at_16",
        "pass_at_32",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"lambda=0.5 summary is missing columns: {sorted(missing)}")
    if "complete" in frame.columns:
        frame = frame[as_complete_mask(frame["complete"])].copy()
    if max_step is not None:
        frame = frame[frame["step"] <= max_step].copy()
    if frame.empty:
        raise ValueError(f"No complete lambda=0.5 steps found in {summary_path}")
    if (frame["trajectory_accuracy"] - frame["pass_at_1"]).abs().max() > 1e-12:
        raise ValueError("lambda=0.5 trajectory accuracy and pass@1 do not match.")
    frame["series"] = "lambda=0.5 mixed policy"
    frame["evaluation_policy"] = "cut-conditioned mixed policy"
    frame["source_path"] = str(summary_path)
    return frame


def load_grpo(
    root: Path,
    data_path: Path,
    max_step: int | None,
) -> pd.DataFrame:
    source_data = pd.read_parquet(data_path).reset_index(names="original_row_index")
    rows: list[dict[str, Any]] = []
    for directory_label, step, path in discover_settings(root):
        if max_step is not None and step > max_step:
            continue
        try:
            summary, _ = summarize_step(
                step,
                load_step(path),
                expected_problems=30,
                expected_samples=32,
                source_data=source_data,
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"WARNING: skipping unfinished GRPO setting {path}: {error}")
            continue
        if not summary["complete"]:
            print(
                f"WARNING: skipping incomplete GRPO setting {path}: "
                f"problems={summary['problems']} "
                f"groups={summary['group_size_min']}-{summary['group_size_max']}"
            )
            continue
        temperature = float(summary["prefix_temperature"])
        if abs(temperature - float(summary["suffix_temperature"])) > 1e-12:
            print(f"WARNING: skipping non-fixed-temperature setting {path}")
            continue
        if abs(float(summary["trajectory_accuracy"]) - float(summary["pass_at_1"])) > 1e-12:
            raise ValueError(f"GRPO accuracy and pass@1 differ for {path}")
        label = f"GRPO {temperature_label(temperature)}"
        summary.update(
            {
                "series": label,
                "evaluation_policy": f"fixed temperature {temperature:g}",
                "temperature": temperature,
                "directory_label": directory_label,
                "source_path": str(path),
            }
        )
        rows.append(summary)
    if not rows:
        raise ValueError(f"No complete GRPO temperature eval settings found under {root}")
    return pd.DataFrame(rows)


def plot_comparison(frame: pd.DataFrame, output_dir: Path, title: str) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), sharex=True)
    flat_axes = axes.ravel()
    for axis, (metric, metric_title) in zip(flat_axes, PLOT_METRICS):
        for series in SERIES_ORDER:
            series_frame = frame[frame["series"] == series].sort_values("step")
            if series_frame.empty:
                continue
            axis.plot(
                series_frame["step"],
                series_frame[metric],
                marker="o",
                linewidth=2.3,
                markersize=5,
                color=SERIES_COLORS[series],
                label=series,
            )
        axis.set_title(metric_title)
        axis.set_ylabel("AIME24 score")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(True, alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    max_step = int(frame["step"].max())
    for axis in flat_axes[: len(PLOT_METRICS)]:
        axis.set_xlabel("Checkpoint step")
        axis.set_xticks(range(1, max_step + 1))
    handles, labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[-1].axis("off")
    flat_axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=11)
    fig.suptitle(title, fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = output_dir / "lambda0p5_vs_grpo_temperature_acc_passk.png"
    pdf = output_dir / "lambda0p5_vs_grpo_temperature_acc_passk.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    grpo_root = resolve_path(
        args.grpo_root,
        discover_latest_grpo_root(args.search_dir),
        "GRPO temperature-eval root",
    )
    lambda05_summary = resolve_path(
        args.lambda05_summary,
        discover_lambda05_summary(args.search_dir),
        "lambda=0.5 mixed-policy summary",
    )
    if not args.data.is_file():
        raise FileNotFoundError(f"AIME parquet does not exist: {args.data}")

    lambda_frame = load_lambda05(lambda05_summary, args.max_step)
    grpo_frame = load_grpo(grpo_root, args.data, args.max_step)
    combined = pd.concat([lambda_frame, grpo_frame], ignore_index=True, sort=False)
    order = {series: index for index, series in enumerate(SERIES_ORDER)}
    combined["series_order"] = combined["series"].map(order)
    combined = combined.sort_values(["series_order", "step"]).drop(columns="series_order")

    present = set(combined["series"])
    for expected in SERIES_ORDER:
        count = int((combined["series"] == expected).sum())
        print(f"{expected}: {count} complete step(s)")
        if expected not in present:
            print(f"WARNING: no complete points available for {expected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "lambda0p5_vs_grpo_temperature_metrics.csv"
    combined.to_csv(csv_path, index=False)
    png, pdf = plot_comparison(combined, args.output_dir, args.title)
    display_columns = [
        "series",
        "step",
        "trajectory_accuracy",
        "pass_at_1",
        "pass_at_4",
        "pass_at_8",
        "pass_at_16",
        "pass_at_32",
    ]
    print("\n" + combined[display_columns].to_string(index=False))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
