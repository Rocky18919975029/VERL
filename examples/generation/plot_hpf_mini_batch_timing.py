#!/usr/bin/env python3
"""Plot per-mini-batch actor timing from HPF Slurm logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MINI_BATCH_DONE_RE = re.compile(
    r"\[HPF\] actor mini-batch done "
    r"label=(?P<label>\S+) "
    r"mini=(?P<mini>\d+)/(?P<total>\d+) "
    r"mini_elapsed_s=(?P<mini_elapsed_s>[0-9.]+) "
    r"elapsed_s=(?P<elapsed_s>[0-9.]+) "
    r"avg_s_per_mini=(?P<avg_s_per_mini>[0-9.]+)"
)
STEP_RE = re.compile(r"(?:^|/)step-(\d+)(?:$|/)" )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Slurm job ID. May be supplied more than once.",
    )
    parser.add_argument(
        "--log",
        action="append",
        type=Path,
        default=[],
        help="Explicit log file. May be supplied more than once.",
    )
    parser.add_argument(
        "--log-prefix",
        default="slurm-verl-hpf-mask",
        help="Slurm log prefix used with --job (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hpf_mini_batch_timing"),
    )
    parser.add_argument("--title", default="HPF actor mini-batch timing")
    return parser.parse_args()


def resolve_logs(args: argparse.Namespace) -> list[Path]:
    paths = list(args.log)
    for job in args.job:
        paths.extend(
            [
                Path(f"{args.log_prefix}-{job}.out"),
                Path(f"{args.log_prefix}-{job}.err"),
            ]
        )
    paths = list(dict.fromkeys(paths))
    existing = [path for path in paths if path.is_file()]
    if not existing:
        requested = "\n".join(str(path) for path in paths) or "(none)"
        raise FileNotFoundError(f"No requested log files exist:\n{requested}")
    return existing


def parse_logs(paths: list[Path]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    source_order = {str(path): index for index, path in enumerate(paths)}
    for path in paths:
        for line_number, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            match = MINI_BATCH_DONE_RE.search(line)
            if match is None:
                continue
            values = match.groupdict()
            step_match = STEP_RE.search(values["label"])
            records.append(
                {
                    "source": str(path),
                    "source_order": source_order[str(path)],
                    "line_number": line_number,
                    "label": values["label"],
                    "step": int(step_match.group(1)) if step_match else pd.NA,
                    "mini_batch": int(values["mini"]),
                    "total_mini_batches": int(values["total"]),
                    "mini_elapsed_s": float(values["mini_elapsed_s"]),
                    "cumulative_elapsed_s": float(values["elapsed_s"]),
                    "running_avg_s_per_mini": float(values["avg_s_per_mini"]),
                }
            )
    if not records:
        raise RuntimeError("No '[HPF] actor mini-batch done' records were found.")
    frame = pd.DataFrame.from_records(records)
    return frame.sort_values(["source_order", "line_number"], kind="stable").reset_index(drop=True)


def add_chronological_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["chronological_mini_batch"] = range(1, len(frame) + 1)
    return frame


def plot(frame: pd.DataFrame, output_dir: Path, title: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)

    groups = list(frame.groupby(["label", "source"], sort=False, dropna=False))
    colors = plt.cm.tab20.colors
    for index, ((label, _source), group) in enumerate(groups):
        step = group["step"].iloc[0]
        legend_label = f"step {int(step)}" if pd.notna(step) else str(label)
        axes[0].plot(
            group["mini_batch"],
            group["mini_elapsed_s"],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=colors[index % len(colors)],
            alpha=0.9,
            label=legend_label,
        )

    axes[0].set_title("Within-step mini-batch time")
    axes[0].set_xlabel("Mini-batch index within actor update")
    axes[0].set_ylabel("Elapsed time (seconds)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=min(4, max(1, len(groups))), frameon=False)

    axes[1].plot(
        frame["chronological_mini_batch"],
        frame["mini_elapsed_s"],
        color="#2563a6",
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="Mini-batch time",
    )
    mean_time = frame["mini_elapsed_s"].mean()
    axes[1].axhline(
        mean_time,
        color="#c52b32",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall mean: {mean_time:.1f}s",
    )

    previous_label = None
    for _, group in frame.groupby(["label", "source"], sort=False, dropna=False):
        if previous_label is not None:
            axes[1].axvline(group["chronological_mini_batch"].iloc[0] - 0.5, color="0.65", linestyle=":")
        previous_label = group["label"].iloc[0]

    axes[1].set_title("Chronological mini-batch time")
    axes[1].set_xlabel("Mini-batch index across parsed logs")
    axes[1].set_ylabel("Elapsed time (seconds)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(title, fontsize=16)
    png = output_dir / "hpf_mini_batch_timing.png"
    pdf = output_dir / "hpf_mini_batch_timing.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    paths = resolve_logs(args)
    frame = add_chronological_index(parse_logs(paths))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "hpf_mini_batch_timing.csv"
    frame.drop(columns=["source_order"]).to_csv(csv_path, index=False)

    summary = (
        frame.groupby(["label", "step"], dropna=False, sort=False)["mini_elapsed_s"]
        .agg(["count", "mean", "min", "max", "sum"])
        .reset_index()
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    png, pdf = plot(frame, args.output_dir, args.title)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
