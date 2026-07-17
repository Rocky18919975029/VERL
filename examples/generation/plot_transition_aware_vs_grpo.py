#!/usr/bin/env python3
"""Incrementally compare two transition-aware lambdas with GRPO baselines."""

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

DEFAULT_LAMBDA1_PROJECT = Path("rollout_data/hpf_transition_aware_fulltail_k16_mini1536_boxed_seed42")
DEFAULT_LAMBDA05_PROJECT = Path(
    "rollout_data/hpf_transition_aware_fulltail_k16_lambda0p5_mini1536_boxed_seed42"
)
DEFAULT_GRPO_PROJECT = Path("rollout_data/grpo_dapo_math17k_mini1536_boxed_n32_seed42")
DEFAULT_GRPO_VLLM_PROJECT = Path("rollout_data/grpo_vllm_logprob_mini1536_boxed_seed42")


@dataclass(frozen=True)
class RunSpec:
    label: str
    rollout_dir: Path
    color: str
    marker: str
    paired: bool
    expected_rows: int
    log_prefix: str

    @property
    def run_name(self) -> str:
        return self.rollout_dir.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambda1-run-dir",
        type=Path,
    )
    parser.add_argument("--lambda1-project-dir", type=Path, default=DEFAULT_LAMBDA1_PROJECT)
    parser.add_argument("--lambda05-run-dir", type=Path)
    parser.add_argument("--lambda05-project-dir", type=Path, default=DEFAULT_LAMBDA05_PROJECT)
    parser.add_argument("--grpo-run-dir", type=Path)
    parser.add_argument("--grpo-project-dir", type=Path, default=DEFAULT_GRPO_PROJECT)
    parser.add_argument("--grpo-vllm-run-dir", type=Path)
    parser.add_argument("--grpo-vllm-project-dir", type=Path, default=DEFAULT_GRPO_VLLM_PROJECT)
    parser.add_argument("--slurm-log-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/transition_aware_lambda_vs_grpo"),
    )
    parser.add_argument("--max-step", type=int)
    return parser.parse_args()


def numeric_step_files(run_dir: Path) -> dict[int, Path]:
    return {
        int(path.stem): path
        for path in run_dir.glob("*.jsonl")
        if path.stem.isdigit()
    }


def available_steps(run_dir: Path, paired: bool) -> list[int]:
    current = set(numeric_step_files(run_dir))
    if not paired:
        return sorted(current)
    next_cut = set(numeric_step_files(run_dir / "transition_next"))
    return sorted(current & next_cut)


def discover_most_complete_run(project_dir: Path, paired: bool) -> Path | None:
    if not project_dir.is_dir():
        return None
    candidates: list[tuple[int, int, float, Path]] = []
    for run_dir in project_dir.iterdir():
        if not run_dir.is_dir():
            continue
        steps = available_steps(run_dir, paired)
        if steps:
            candidates.append((len(steps), max(steps), run_dir.stat().st_mtime, run_dir))
    return max(candidates)[-1] if candidates else None


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


def summarize_rollouts(spec: RunSpec, max_step: int | None) -> pd.DataFrame:
    current_files = numeric_step_files(spec.rollout_dir)
    next_files = numeric_step_files(spec.rollout_dir / "transition_next") if spec.paired else {}
    records: list[dict[str, object]] = []
    for step in available_steps(spec.rollout_dir, spec.paired):
        if max_step is not None and step > max_step:
            continue
        current_rows, current_acc = read_accuracy(current_files[step])
        if current_rows != spec.expected_rows:
            print(
                f"WARNING: ignoring incomplete {spec.label} step {step}: "
                f"rows={current_rows}, expected={spec.expected_rows}"
            )
            continue

        next_rows: int | None = None
        next_acc: float | None = None
        if spec.paired:
            next_rows, next_acc = read_accuracy(next_files[step])
            if next_rows != spec.expected_rows:
                print(
                    f"WARNING: ignoring incomplete {spec.label} next-cut step {step}: "
                    f"rows={next_rows}, expected={spec.expected_rows}"
                )
                continue

        records.append(
            {
                "run": spec.label,
                "run_name": spec.run_name,
                "step": step,
                "train_rows": current_rows,
                "train_acc": current_acc,
                "next_train_rows": next_rows,
                "next_train_acc": next_acc,
            }
        )
    if not records:
        raise RuntimeError(f"No complete rollout steps found for {spec.label}: {spec.rollout_dir}")
    return pd.DataFrame.from_records(records)


def log_contains_run(path: Path, run_name: str, max_lines: int = 1200) -> bool:
    try:
        with path.open(errors="ignore") as handle:
            for index, line in enumerate(handle):
                if run_name in line:
                    return True
                if index + 1 >= max_lines:
                    break
    except OSError:
        return False
    return False


def discover_logs(log_dir: Path, spec: RunSpec) -> list[Path]:
    pattern = f"{spec.log_prefix}-*.out"
    return [
        path
        for path in sorted(log_dir.glob(pattern))
        if log_contains_run(path, spec.run_name)
    ]


def parse_eval_metrics(log_paths: list[Path], valid_steps: set[int]) -> dict[int, float]:
    values: dict[int, float] = {}
    for path in log_paths:
        with path.open(errors="ignore") as handle:
            for line in handle:
                if EVAL_ACC_KEY not in line:
                    continue
                step_match = STEP_PATTERN.search(line)
                if step_match is None:
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
    logs = discover_logs(log_dir, spec)
    values = parse_eval_metrics(logs, set(frame["step"].astype(int)))
    result = frame.copy()
    result["eval_acc"] = result["step"].map(values)
    print(
        f"{spec.label}: run={spec.run_name}, steps={result['step'].min()}-{result['step'].max()}, "
        f"logs={len(logs)}, eval_points={result['eval_acc'].notna().sum()}"
    )
    if not logs:
        print(f"WARNING: no Slurm logs found for {spec.run_name}")
    return result


def add_epoch_guides(ax, max_step: int) -> None:
    for boundary in range(3, max_step, 3):
        ax.axvline(boundary + 0.5, color="#777777", linestyle="--", linewidth=1.0, alpha=0.35)


def make_plot(metrics: pd.DataFrame, specs: list[RunSpec], output_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), sharex=True)
    for spec in specs:
        frame = metrics[metrics["run"] == spec.label].sort_values("step")
        axes[0].plot(
            frame["step"],
            frame["train_acc"],
            color=spec.color,
            marker=spec.marker,
            linewidth=2.4,
            label=spec.label,
        )
        if frame["next_train_acc"].notna().any():
            axes[0].plot(
                frame["step"],
                frame["next_train_acc"],
                color=spec.color,
                marker=spec.marker,
                linewidth=1.8,
                linestyle="--",
                alpha=0.65,
                label=f"{spec.label} next cut",
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
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Transition-aware mixed-policy optimization vs full-trajectory GRPO\n"
        "mini1536, strict boxed verifier, seed42",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "transition_aware_lambda_vs_grpo.png"
    pdf = output_dir / "transition_aware_lambda_vs_grpo.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def resolve_run(explicit: Path | None, project: Path, paired: bool, label: str) -> Path | None:
    if explicit is not None:
        if not explicit.is_dir():
            raise FileNotFoundError(f"Requested {label} run does not exist: {explicit}")
        return explicit
    return discover_most_complete_run(project, paired)


def main() -> None:
    args = parse_args()
    lambda1_dir = resolve_run(
        args.lambda1_run_dir,
        args.lambda1_project_dir,
        paired=True,
        label="lambda=1.0",
    )
    if lambda1_dir is None:
        raise FileNotFoundError(f"No complete lambda=1.0 run found under {args.lambda1_project_dir}")
    specs = [
        RunSpec(
            "lambda=1.0",
            lambda1_dir,
            "#2563a6",
            "o",
            True,
            8192,
            "slurm-verl-hpf-mask",
        )
    ]

    lambda05_dir = resolve_run(
        args.lambda05_run_dir,
        args.lambda05_project_dir,
        paired=True,
        label="lambda=0.5",
    )
    if lambda05_dir is not None:
        specs.append(
            RunSpec("lambda=0.5", lambda05_dir, "#b51f2e", "s", True, 8192, "slurm-verl-hpf-mask")
        )
    else:
        print("WARNING: no complete lambda=0.5 run discovered")

    grpo_dir = resolve_run(args.grpo_run_dir, args.grpo_project_dir, paired=False, label="GRPO n=32")
    if grpo_dir is not None:
        specs.append(
            RunSpec("GRPO n=32", grpo_dir, "#238b45", "^", False, 16384, "slurm-verl-resched-grpo")
        )
    else:
        print("WARNING: no complete GRPO n=32 run discovered; rerun after its first rollout dump")

    grpo_vllm_dir = resolve_run(
        args.grpo_vllm_run_dir,
        args.grpo_vllm_project_dir,
        paired=False,
        label="GRPO n=16 (vLLM behavior log-prob)",
    )
    if grpo_vllm_dir is not None:
        specs.append(
            RunSpec(
                "GRPO n=16 (vLLM behavior log-prob)",
                grpo_vllm_dir,
                "#7a3db8",
                "D",
                False,
                8192,
                "slurm-verl-resched-grpo",
            )
        )
    else:
        print(
            "WARNING: no complete GRPO n=16 vLLM-logprob run discovered; "
            "rerun after its first rollout dump"
        )

    frames = []
    for spec in specs:
        frame = summarize_rollouts(spec, args.max_step)
        frames.append(attach_eval_metrics(frame, spec, args.slurm_log_dir))
    metrics = pd.concat(frames, ignore_index=True)
    order = {spec.label: index for index, spec in enumerate(specs)}
    metrics["run_order"] = metrics["run"].map(order)
    metrics = metrics.sort_values(["run_order", "step"]).drop(columns=["run_order"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "transition_aware_lambda_vs_grpo_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    png, pdf = make_plot(metrics, specs, args.output_dir)

    print(metrics.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
