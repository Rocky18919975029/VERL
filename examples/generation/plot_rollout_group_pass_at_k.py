#!/usr/bin/env python3
"""Compute exact per-step pass@k for train and AIME24 rollout groups."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_KS = (1, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Experiment specification label::rollout_run_dir. Numeric steps, transition_next, and "
            "the matching validation_data directory are discovered automatically."
        ),
    )
    parser.add_argument(
        "--csv-only-run",
        action="append",
        default=[],
        help="Auxiliary rollout specification label::rollout_dir. Compute it for CSV but do not plot it.",
    )
    parser.add_argument(
        "--aime-run",
        action="append",
        default=[],
        help="Plotted AIME24 validation specification label::validation_data_dir.",
    )
    parser.add_argument(
        "--k",
        action="append",
        type=int,
        default=None,
        help="Pass@k value. Repeat as needed; defaults to 1, 4, 8, and 16.",
    )
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rollout_group_pass_at_k"))
    parser.add_argument("--title", default="Rollout-group pass@k")
    args = parser.parse_args()
    args.ks = tuple(args.k or DEFAULT_KS)
    if any(k <= 0 for k in args.ks):
        parser.error("Every --k must be positive.")
    if len(set(args.ks)) != len(args.ks):
        parser.error("Duplicate --k values are not allowed.")
    return args


def parse_run_specs(values: list[str], option: str) -> list[tuple[str, Path]]:
    specs = []
    labels = set()
    for value in values:
        parts = value.split("::", 1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid {option} {value!r}; expected label::directory")
        label, directory = parts
        if label in labels:
            raise ValueError(f"Duplicate run label: {label!r}")
        labels.add(label)
        specs.append((label, Path(directory)))
    return specs


def infer_validation_directory(rollout_directory: Path) -> Path | None:
    parts = list(rollout_directory.parts)
    try:
        index = parts.index("rollout_data")
    except ValueError:
        return None
    parts[index] = "validation_data"
    return Path(*parts)


def step_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def correctness(row: dict[str, Any], path: Path) -> bool:
    for key in ("acc", "is_correct"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
            return value.strip().lower() in {"true", "1"}
        raise ValueError(f"Unsupported correctness value {value!r} in {path}")
    raise KeyError(f"Rows in {path} contain neither 'acc' nor 'is_correct'.")


def group_key(row: dict[str, Any], path: Path) -> tuple[str, str]:
    if "input" not in row:
        raise KeyError(f"Rows in {path} are missing 'input'; cannot reconstruct rollout groups.")
    return str(row["input"]), str(row.get("gts", ""))


def exact_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Probability that a size-k sample without replacement contains a correct item."""
    if not 0 <= num_correct <= num_samples:
        raise ValueError(f"Invalid correct/sample counts: {num_correct}/{num_samples}")
    if k > num_samples:
        raise ValueError(f"pass@{k} requires at least {k} trajectories, but group has {num_samples}.")
    num_wrong = num_samples - num_correct
    if num_wrong < k:
        return 1.0
    return 1.0 - math.comb(num_wrong, k) / math.comb(num_samples, k)


def summarize_step(
    path: Path,
    label: str,
    step: int,
    ks: tuple[int, ...],
    dataset: str,
    plotted: bool,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Rollout file is empty: {path}")

    groups: dict[tuple[str, str], list[bool]] = {}
    for row in rows:
        groups.setdefault(group_key(row, path), []).append(correctness(row, path))

    sizes = [len(values) for values in groups.values()]
    min_size = min(sizes)
    max_k = max(ks)
    if min_size < max_k:
        raise ValueError(
            f"{path} has group_size_min={min_size}, which is smaller than requested max k={max_k}."
        )
    modal_group_size = max(set(sizes), key=lambda size: (sizes.count(size), size))
    record: dict[str, Any] = {
        "run": label,
        "dataset": dataset,
        "plotted": plotted,
        "step": step,
        "rows": len(rows),
        "groups": len(groups),
        "group_size_min": min_size,
        "group_size_max": max(sizes),
        "group_size_mode": modal_group_size,
        "nonmodal_groups": sum(size != modal_group_size for size in sizes),
        "rollout_acc": sum(sum(values) for values in groups.values()) / len(rows),
        "source": str(path),
    }
    for k in ks:
        record[f"pass_at_{k}"] = sum(
            exact_pass_at_k(len(values), sum(values), k) for values in groups.values()
        ) / len(groups)
    return record


def summarize_run(
    label: str,
    directory: Path,
    max_step: int | None,
    ks: tuple[int, ...],
    dataset: str,
    plotted: bool,
) -> pd.DataFrame:
    if not directory.is_dir():
        raise FileNotFoundError(f"Rollout directory does not exist: {directory}")
    paths = []
    for path in directory.glob("*.jsonl"):
        step = step_from_path(path)
        if step is not None and (max_step is None or step <= max_step):
            paths.append((step, path))
    if not paths:
        raise FileNotFoundError(f"No numeric step JSONL files found under {directory}")
    return pd.DataFrame(
        summarize_step(path, label, step, ks, dataset, plotted) for step, path in sorted(paths)
    )


def validate_aime_groups(metrics: pd.DataFrame) -> None:
    aime = metrics[metrics["dataset"] == "aime24_validation"]
    if aime.empty:
        return
    invalid = aime[
        (aime["groups"] != 30)
        | (aime["group_size_min"] != 32)
        | (aime["group_size_max"] != 32)
        | (aime["nonmodal_groups"] != 0)
    ]
    if not invalid.empty:
        columns = [
            "run",
            "step",
            "rows",
            "groups",
            "group_size_min",
            "group_size_max",
            "nonmodal_groups",
            "source",
        ]
        raise ValueError(
            "AIME24 validation must regroup into exactly 30 problems with 32 trajectories each. "
            "Invalid steps:\n" + invalid[columns].to_string(index=False)
        )


def plot_metrics(metrics: pd.DataFrame, output_path: Path, title: str, ks: tuple[int, ...]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.style.use("seaborn-v0_8-whitegrid")
    plotted = metrics[metrics["plotted"]]
    datasets = [dataset for dataset in ("train_rollout", "aime24_validation") if dataset in set(plotted["dataset"])]
    if not datasets:
        raise ValueError("No plotted train or AIME24 runs were provided.")

    fig, axes = plt.subplots(
        len(datasets),
        len(ks),
        figsize=(4.2 * len(ks), 4.0 * len(datasets)),
        sharex="col",
        sharey=True,
        squeeze=False,
    )
    row_titles = {
        "train_rollout": "Train rollout",
        "aime24_validation": "AIME24 validation",
    }
    for row, dataset in enumerate(datasets):
        dataset_metrics = plotted[plotted["dataset"] == dataset]
        for column, k in enumerate(ks):
            ax = axes[row, column]
            metric = f"pass_at_{k}"
            for label, frame in dataset_metrics.groupby("run", sort=False):
                frame = frame.sort_values("step")
                ax.plot(
                    frame["step"],
                    frame[metric],
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=label,
                )
            ax.set_title(f"{row_titles[dataset]} pass@{k}")
            ax.set_xlabel("Global step")
            if column == 0:
                ax.set_ylabel("Expected pass rate")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles_by_label = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(zip(labels, handles, strict=True))
    labels = list(handles_by_label)
    handles = [handles_by_label[label] for label in labels]
    fig.suptitle(title, y=0.99, fontsize=15)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, max(1, len(labels))),
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    train_specs = parse_run_specs(args.run, "--run")
    csv_only_specs = parse_run_specs(args.csv_only_run, "--csv-only-run")
    aime_specs = parse_run_specs(args.aime_run, "--aime-run")
    frames = [
        summarize_run(label, directory, args.max_step, args.ks, "train_rollout", True)
        for label, directory in train_specs
    ]
    explicit_csv_only_paths = {directory.resolve() for _, directory in csv_only_specs}
    explicit_aime_paths = {directory.resolve() for _, directory in aime_specs}
    for label, directory in train_specs:
        next_directory = directory / "transition_next"
        if next_directory.is_dir() and next_directory.resolve() not in explicit_csv_only_paths:
            print(f"Auto-discovered CSV-only next cut: {label} -> {next_directory}")
            frames.append(
                summarize_run(
                    f"{label} next",
                    next_directory,
                    args.max_step,
                    args.ks,
                    "auxiliary_rollout",
                    False,
                )
            )

        validation_directory = infer_validation_directory(directory)
        if (
            validation_directory is not None
            and validation_directory.is_dir()
            and validation_directory.resolve() not in explicit_aime_paths
        ):
            print(f"Auto-discovered AIME24 validation: {label} -> {validation_directory}")
            frames.append(
                summarize_run(
                    label,
                    validation_directory,
                    args.max_step,
                    args.ks,
                    "aime24_validation",
                    True,
                )
            )
    frames.extend(
        summarize_run(label, directory, args.max_step, args.ks, "auxiliary_rollout", False)
        for label, directory in csv_only_specs
    )
    frames.extend(
        summarize_run(label, directory, args.max_step, args.ks, "aime24_validation", True)
        for label, directory in aime_specs
    )
    metrics = pd.concat(frames, ignore_index=True).sort_values(["dataset", "step", "run"])
    validate_aime_groups(metrics)

    pass_columns = [f"pass_at_{k}" for k in args.ks]
    summary = (
        metrics.groupby(["dataset", "run"], sort=False)[pass_columns]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "rollout_group_pass_at_k_per_step.csv"
    summary_path = args.output_dir / "rollout_group_pass_at_k_summary.csv"
    figure_path = args.output_dir / "rollout_group_pass_at_k.png"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_metrics(metrics, figure_path, args.title, args.ks)

    display_columns = [
        "run",
        "dataset",
        "plotted",
        "step",
        "rows",
        "groups",
        "group_size_min",
        "group_size_max",
        "nonmodal_groups",
        *pass_columns,
    ]
    print(metrics[display_columns].to_string(index=False))
    if int(metrics["nonmodal_groups"].sum()) > 0:
        print("\nWARNING: Some reconstructed groups have non-modal sizes; inspect the CSV before comparing pass@k.")
    print("\npass@k is the exact expectation for sampling k trajectories without replacement.")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
