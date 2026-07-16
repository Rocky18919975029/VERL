#!/usr/bin/env python3
"""Incrementally compare lambda=1.0 and lambda=0.5 transition-aware runs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EVAL_ACC_KEY = "val-core/aime_2024_dapo_boxed/acc/mean@1"
STEP_PATTERN = re.compile(r"\bstep:(\d+)\b")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
DEFAULT_LAMBDA1_PROJECT = Path(
    "rollout_data/hpf_transition_aware_fulltail_k16_mini1536_boxed_seed42"
)
DEFAULT_LAMBDA1_RUN = (
    "transition_aware_fulltail_k16_b512_mini1536_boxed_1epoch_seed42_20260717_001640"
)
DEFAULT_LAMBDA05_PROJECT = Path(
    "rollout_data/hpf_transition_aware_fulltail_k16_lambda0p5_mini1536_boxed_seed42"
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    lambda_value: float
    rollout_dir: Path
    color: str
    marker: str

    @property
    def run_name(self) -> str:
        return self.rollout_dir.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambda1-run-dir",
        type=Path,
        default=DEFAULT_LAMBDA1_PROJECT / DEFAULT_LAMBDA1_RUN,
    )
    parser.add_argument(
        "--lambda05-run-dir",
        type=Path,
        help="Defaults to the lambda=0.5 run with the most complete steps.",
    )
    parser.add_argument("--lambda05-project-dir", type=Path, default=DEFAULT_LAMBDA05_PROJECT)
    parser.add_argument("--slurm-log-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/transition_aware_lambda1_vs_lambda0p5"),
    )
    parser.add_argument("--expected-rows", type=int, default=8192)
    parser.add_argument("--max-step", type=int)
    return parser.parse_args()


def numeric_step_files(run_dir: Path) -> dict[int, Path]:
    files = {}
    for path in run_dir.glob("*.jsonl"):
        if path.stem.isdigit():
            files[int(path.stem)] = path
    return files


def paired_steps(run_dir: Path) -> list[int]:
    current = numeric_step_files(run_dir)
    next_cut = numeric_step_files(run_dir / "transition_next")
    return sorted(set(current) & set(next_cut))


def discover_latest_run(project_dir: Path) -> Path | None:
    if not project_dir.is_dir():
        return None
    candidates = []
    for run_dir in project_dir.iterdir():
        if not run_dir.is_dir():
            continue
        steps = paired_steps(run_dir)
        if steps:
            candidates.append((len(steps), max(steps), run_dir.stat().st_mtime, run_dir))
    if not candidates:
        return None
    return max(candidates)[-1]


def read_accuracy(path: Path) -> tuple[int, float]:
    rows = 0
    correct = 0.0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            correct += float(record.get("acc", record.get("is_correct", 0.0)) or 0.0)
    if rows == 0:
        raise ValueError(f"Rollout file is empty: {path}")
    return rows, correct / rows


def summarize_rollouts(spec: RunSpec, expected_rows: int, max_step: int | None) -> pd.DataFrame:
    records = []
    current_files = numeric_step_files(spec.rollout_dir)
    next_files = numeric_step_files(spec.rollout_dir / "transition_next")
    for step in sorted(set(current_files) & set(next_files)):
        if max_step is not None and step > max_step:
            continue
        current_rows, current_acc = read_accuracy(current_files[step])
        next_rows, next_acc = read_accuracy(next_files[step])
        if expected_rows > 0 and (current_rows != expected_rows or next_rows != expected_rows):
            print(
                f"WARNING: ignoring incomplete {spec.label} step {step}: "
                f"current={current_rows}, next={next_rows}, expected={expected_rows}"
            )
            continue
        records.append(
            {
                "run": spec.label,
                "lambda": spec.lambda_value,
                "run_name": spec.run_name,
                "step": step,
                "current_rows": current_rows,
                "current_train_acc": current_acc,
                "next_rows": next_rows,
                "next_train_acc": next_acc,
            }
        )
    if not records:
        raise RuntimeError(f"No complete paired rollout steps found for {spec.label}: {spec.rollout_dir}")
    return pd.DataFrame.from_records(records)


def log_belongs_to_run(path: Path, run_name: str, max_lines: int = 800) -> bool:
    marker = f"Run name: {run_name}"
    try:
        with path.open(errors="ignore") as handle:
            for index, line in enumerate(handle):
                if marker in line:
                    return True
                if index + 1 >= max_lines:
                    break
    except OSError:
        return False
    return False


def discover_logs(log_dir: Path, run_name: str) -> list[Path]:
    return [
        path
        for path in sorted(log_dir.glob("slurm-verl-hpf-mask-*.out"))
        if log_belongs_to_run(path, run_name)
    ]


def parse_eval_metrics(log_paths: list[Path], valid_steps: set[int]) -> dict[int, float]:
    values = {}
    for path in log_paths:
        with path.open(errors="ignore") as handle:
            for line in handle:
                if EVAL_ACC_KEY not in line:
                    continue
                step_match = STEP_PATTERN.search(line)
                if not step_match:
                    continue
                step = int(step_match.group(1))
                if step not in valid_steps:
                    continue
                value_text = line.split(EVAL_ACC_KEY, 1)[1]
                value_text = value_text.replace("np.float64", "").replace("np.float32", "")
                value_match = NUMBER_PATTERN.search(value_text)
                if value_match:
                    values[step] = float(value_match.group(0))
    return values


def attach_eval_metrics(frame: pd.DataFrame, spec: RunSpec, log_dir: Path) -> pd.DataFrame:
    logs = discover_logs(log_dir, spec.run_name)
    if not logs:
        print(f"WARNING: no Slurm logs discovered for {spec.run_name}")
    values = parse_eval_metrics(logs, set(frame["step"].astype(int)))
    result = frame.copy()
    result["eval_acc"] = result["step"].map(values)
    print(
        f"{spec.label}: run={spec.run_name}, steps={result['step'].min()}-{result['step'].max()}, "
        f"logs={len(logs)}, eval_points={result['eval_acc'].notna().sum()}"
    )
    return result


def add_epoch_guides(ax, max_step: int) -> None:
    for boundary in range(3, max_step, 3):
        ax.axvline(boundary + 0.5, color="#777777", linestyle="--", linewidth=1.0, alpha=0.35)


def make_plot(metrics: pd.DataFrame, specs: list[RunSpec], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True)
    for spec in specs:
        frame = metrics[metrics["run"] == spec.label].sort_values("step")
        axes[0].plot(
            frame["step"],
            frame["current_train_acc"],
            color=spec.color,
            marker=spec.marker,
            linewidth=2.4,
            label=f"{spec.label} current",
        )
        axes[0].plot(
            frame["step"],
            frame["next_train_acc"],
            color=spec.color,
            marker=spec.marker,
            linewidth=2.0,
            linestyle="--",
            alpha=0.78,
            label=f"{spec.label} next",
        )
        axes[1].plot(
            frame["step"],
            frame["eval_acc"],
            color=spec.color,
            marker=spec.marker,
            linewidth=2.4,
            label=spec.label,
        )

    max_step = int(metrics["step"].max())
    for ax, title in zip(axes, ["Train rollout accuracy", "AIME24 validation accuracy"]):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Global step")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(range(1, max_step + 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_epoch_guides(ax, max_step)
        ax.legend(frameon=False)

    fig.suptitle(
        "Transition-aware mixed-policy optimization\n"
        "lambda comparison, full tail, K=16, mini1536, seed42",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "lambda1_vs_lambda0p5.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "lambda1_vs_lambda0p5.pdf", bbox_inches="tight")


def main() -> None:
    args = parse_args()
    specs = [
        RunSpec("lambda=1.0", 1.0, args.lambda1_run_dir, "#2563a6", "o"),
    ]
    if args.lambda05_run_dir is not None:
        if not args.lambda05_run_dir.is_dir():
            raise FileNotFoundError(f"Requested lambda=0.5 run does not exist: {args.lambda05_run_dir}")
        lambda05_dir = args.lambda05_run_dir
    else:
        lambda05_dir = discover_latest_run(args.lambda05_project_dir)
    if lambda05_dir is None:
        print(
            "WARNING: lambda=0.5 has no complete paired rollout step yet; "
            "plotting lambda=1.0 only. Rerun this command after its first step is dumped."
        )
    else:
        specs.append(RunSpec("lambda=0.5", 0.5, lambda05_dir, "#b51f2e", "s"))

    frames = []
    for spec in specs:
        frame = summarize_rollouts(spec, args.expected_rows, args.max_step)
        frames.append(attach_eval_metrics(frame, spec, args.slurm_log_dir))
    metrics = pd.concat(frames, ignore_index=True).sort_values(["lambda", "step"], ascending=[False, True])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "lambda1_vs_lambda0p5_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    make_plot(metrics, specs, args.output_dir)

    print(metrics.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_dir / 'lambda1_vs_lambda0p5.png'}")
    print(f"Wrote {args.output_dir / 'lambda1_vs_lambda0p5.pdf'}")


if __name__ == "__main__":
    main()
