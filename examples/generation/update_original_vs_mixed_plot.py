#!/usr/bin/env python3
"""Incrementally update Original Alg3 versus mixed-policy training plots."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


EVAL_ACC_KEY = "val-core/aime_2024_dapo_boxed/acc/mean@1"
DEFAULT_OUTPUT_DIR = Path("outputs/hpf_alg3_mini1536_boxed_3epoch_acc")
DEFAULT_MIXED_RUN = "hpf_mixed_grpo_k16_b512_mini1536_boxed_smoke1_seed42_20260715_112434"
DEFAULT_MIXED_PROJECT = "hpf_mixed_grpo_k16_mini1536_boxed_seed42"
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
STEP_PATTERN = re.compile(r"\bstep:(\d+)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-csv",
        type=Path,
        help="Original Alg3 step metrics. If omitted, known filenames under --output-dir are tried.",
    )
    parser.add_argument(
        "--mixed-rollout-dir",
        type=Path,
        default=Path("rollout_data") / DEFAULT_MIXED_PROJECT / DEFAULT_MIXED_RUN,
    )
    parser.add_argument(
        "--mixed-checkpoint-dir",
        type=Path,
        default=Path("checkpoints") / DEFAULT_MIXED_PROJECT / DEFAULT_MIXED_RUN,
    )
    parser.add_argument("--mixed-run-name", default=DEFAULT_MIXED_RUN)
    parser.add_argument("--slurm-log-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--mixed-job",
        action="append",
        type=int,
        default=[],
        help="Optional Slurm job ID. Repeat to bypass automatic log discovery.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-rows", type=int, default=8192)
    return parser.parse_args()


def find_original_csv(args: argparse.Namespace) -> Path:
    if args.original_csv is not None:
        if not args.original_csv.exists():
            raise FileNotFoundError(f"Original metrics CSV does not exist: {args.original_csv}")
        return args.original_csv

    candidates = [
        args.output_dir / "train_val_acc_steps1_18.csv",
        args.output_dir / "train_valacc_steps1_18.csv",
        args.output_dir / "train_val_acc.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Original Alg3 metrics CSV not found. Tried:\n{tried}")


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "global_step": "step",
        "train_rollout_acc": "train_acc",
        "aime24_val_acc": "eval_acc",
        "val_acc": "eval_acc",
    }
    df = df.rename(
        columns={old: new for old, new in aliases.items() if old in df.columns and new not in df.columns}
    ).copy()
    required = {"step", "train_acc", "eval_acc"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Metrics CSV is missing {sorted(missing)}; columns are {list(df.columns)}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values("step").drop_duplicates("step", keep="last")


def latest_completed_step(checkpoint_dir: Path) -> int:
    pointer = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if pointer.exists():
        value = pointer.read_text().strip()
        if value.isdigit():
            return int(value)

    steps = []
    for path in checkpoint_dir.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if match and (path / "actor").is_dir():
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No completed checkpoints found under {checkpoint_dir}")
    return max(steps)


def summarize_mixed_rollouts(rollout_dir: Path, latest_step: int, expected_rows: int) -> pd.DataFrame:
    records = []
    for step in range(1, latest_step + 1):
        path = rollout_dir / f"{step}.jsonl"
        if not path.exists():
            print(f"WARNING: completed checkpoint step {step} has no rollout file: {path}")
            continue

        row_count = 0
        acc_sum = 0.0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_count += 1
                acc_sum += float(row.get("acc", row.get("is_correct", 0.0)) or 0.0)

        if row_count == 0:
            print(f"WARNING: ignoring empty rollout file: {path}")
            continue
        if expected_rows > 0 and row_count != expected_rows:
            print(
                f"WARNING: ignoring step {step}: expected {expected_rows} rollout rows, "
                f"found {row_count}"
            )
            continue
        records.append({"step": step, "train_rows": row_count, "train_acc": acc_sum / row_count})

    if not records:
        raise RuntimeError(f"No complete mixed-policy rollout files found under {rollout_dir}")
    return pd.DataFrame.from_records(records)


def log_belongs_to_run(path: Path, run_name: str, max_lines: int = 700) -> bool:
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


def discover_mixed_logs(args: argparse.Namespace) -> list[Path]:
    if args.mixed_job:
        paths = [args.slurm_log_dir / f"slurm-verl-hpf-mask-{job}.out" for job in args.mixed_job]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing requested Slurm logs:\n" + "\n".join(map(str, missing)))
        return paths

    paths = []
    for path in sorted(args.slurm_log_dir.glob("slurm-verl-hpf-mask-*.out")):
        if log_belongs_to_run(path, args.mixed_run_name):
            paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f"No Slurm stdout logs for run {args.mixed_run_name!r} under {args.slurm_log_dir}"
        )
    return paths


def parse_eval_metrics(log_paths: list[Path], latest_step: int) -> pd.DataFrame:
    values: dict[int, float] = {}
    for path in log_paths:
        with path.open(errors="ignore") as handle:
            for line in handle:
                if EVAL_ACC_KEY not in line:
                    continue
                step_match = STEP_PATTERN.search(line)
                if not step_match:
                    continue
                step = int(step_match.group(1))
                if not 1 <= step <= latest_step:
                    continue

                value_text = line.split(EVAL_ACC_KEY, 1)[1]
                value_text = value_text.replace("np.float64", "").replace("np.float32", "")
                value_match = NUMBER_PATTERN.search(value_text)
                if value_match:
                    values[step] = float(value_match.group(0))

    return pd.DataFrame(
        [{"step": step, "eval_acc": value} for step, value in sorted(values.items())]
    )


def add_epoch_guides(ax, max_step: int) -> None:
    for boundary in range(3, max_step, 3):
        ax.axvline(boundary + 0.5, color="#777777", linestyle="--", linewidth=1.0, alpha=0.45)
    upper = ax.get_ylim()[1]
    for epoch_start in range(1, max_step + 1, 3):
        epoch = (epoch_start - 1) // 3 + 1
        center = min(epoch_start + 1, max_step)
        ax.text(
            center,
            upper,
            f"E{epoch}\nH={64 * epoch}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#666666",
        )


def style_axis(ax, title: str, xlabel: str) -> None:
    from matplotlib.ticker import PercentFormatter

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_global_steps(original: pd.DataFrame, mixed: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    series = [(original, "Original Alg3", "#2563a6", "o"), (mixed, "Mixed-policy GRPO", "#b51f2e", "s")]
    for frame, label, color, marker in series:
        axes[0].plot(frame["step"], frame["train_acc"], label=label, color=color, marker=marker, linewidth=2.4)
        axes[1].plot(frame["step"], frame["eval_acc"], label=label, color=color, marker=marker, linewidth=2.4)

    max_step = int(max(original["step"].max(), mixed["step"].max()))
    style_axis(axes[0], "Train rollout accuracy", "Global step")
    style_axis(axes[1], "AIME24 validation accuracy", "Global step")
    for ax in axes:
        ax.set_xlim(0.5, max_step + 0.5)
        ax.set_xticks(range(1, max_step + 1))
        add_epoch_guides(ax, max_step)
        ax.legend(frameon=False)

    fig.suptitle("Original Alg3 vs mixed-policy GRPO\nmini1536, seed42, strict boxed verifier", fontsize=17, y=1.05)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"train_val_acc_original_vs_mixed_18steps.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_update_aligned(original: pd.DataFrame, mixed: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = original.copy()
    mixed = mixed.copy()
    # Train accuracy is measured on the rollout sampled at this step, so place
    # it at the cumulative rollout count including that batch. Alg3 performs an
    # initial and a fresh rollout per global step, while mixed-policy performs
    # one rollout per global step. Only Alg3's initial-rollout metric is logged.
    original["train_budget_step"] = 2 * original["step"] - 1
    original["eval_budget_step"] = 2 * original["step"]
    mixed["train_budget_step"] = mixed["step"]
    mixed["eval_budget_step"] = mixed["step"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    series = [(original, "Original Alg3", "#2563a6", "o"), (mixed, "Mixed-policy GRPO", "#b51f2e", "s")]
    for frame, label, color, marker in series:
        axes[0].plot(
            frame["train_budget_step"], frame["train_acc"], label=label, color=color, marker=marker, linewidth=2.4
        )
        axes[1].plot(
            frame["eval_budget_step"], frame["eval_acc"], label=label, color=color, marker=marker, linewidth=2.4
        )

    style_axis(axes[0], "Train rollout accuracy", "Cumulative rollout batches including this sample")
    style_axis(axes[1], "AIME24 validation accuracy", "Completed rollout/update rounds")
    for ax in axes:
        ax.set_xlim(left=0)
        ax.legend(frameon=False)

    fig.suptitle("Original Alg3 vs mixed-policy GRPO\naligned by rollout and actor-update budget", fontsize=17, y=1.04)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"train_val_acc_original_vs_mixed_update_aligned.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_csv = find_original_csv(args)
    original = normalize_metrics(pd.read_csv(original_csv))
    latest_step = latest_completed_step(args.mixed_checkpoint_dir)
    mixed_train = summarize_mixed_rollouts(args.mixed_rollout_dir, latest_step, args.expected_rows)
    log_paths = discover_mixed_logs(args)
    mixed_eval = parse_eval_metrics(log_paths, latest_step)
    mixed = mixed_train.merge(mixed_eval, on="step", how="left").sort_values("step")

    missing_eval = mixed.loc[mixed["eval_acc"].isna(), "step"].astype(int).tolist()
    mixed_csv = args.output_dir / "mixed_policy_completed_metrics.csv"
    mixed.to_csv(mixed_csv, index=False)

    plot_global_steps(original, mixed, args.output_dir)
    plot_update_aligned(original, mixed, args.output_dir)

    print(f"Original metrics: {original_csv}")
    print(f"Mixed run: {args.mixed_run_name}")
    print(f"Latest completed checkpoint: {latest_step}")
    print("Matched Slurm logs:")
    for path in log_paths:
        print(f"  {path}")
    print("\nCompleted mixed-policy metrics:")
    print(mixed.to_string(index=False))
    if missing_eval:
        print(f"\nWARNING: completed steps missing validation metrics: {missing_eval}")
    print(f"\nWrote {mixed_csv}")
    print(f"Wrote {args.output_dir / 'train_val_acc_original_vs_mixed_18steps.png'}")
    print(f"Wrote {args.output_dir / 'train_val_acc_original_vs_mixed_update_aligned.png'}")


if __name__ == "__main__":
    main()
