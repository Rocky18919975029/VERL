#!/usr/bin/env python3
"""Summarize HPF training diagnostics from one synced Weights & Biases run.

Run this on a login node after syncing an offline run. The script reads only
the W&B history and writes a compact Markdown and JSON report locally.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "val-core/aime_2024_dapo_boxed/acc/mean@1",
    "val-aux/aime_2024_dapo_boxed/reward/mean@1",
    "hpf/tree_horizon_tokens",
    "hpf/horizon_round_index",
    "hpf/tree_prefix_tokens_mean",
    "hpf/tree_prefix_stopped_frac",
    "hpf/follower/actor/pg_loss",
    "hpf/leader/actor/pg_loss",
    "hpf/follower/actor/hpf_kl_loss",
    "hpf/leader/actor/hpf_kl_loss",
    "hpf/correction_log_ratio_mean",
    "hpf/correction_log_ratio_std",
    "hpf/correction_clip_upper_frac",
    "hpf/correction_clip_lower_frac",
    "hpf/correction_clip_frac",
    "hpf/correction_ratio_mean",
    "hpf/correction_ratio_max",
    "hpf/correction_ratio_min",
    "timing_s/hpf/tree_rollout_total_wall",
    "timing_s/hpf/prefix_rollout_wall",
    "timing_s/hpf/suffix_rollout_wall",
    "timing_s/hpf/role_old_log_prob",
    "timing_s/hpf/follower_update_actor",
    "timing_s/hpf/suffix_correction_log_prob",
    "timing_s/hpf/leader_update_actor",
    "timing_s/hpf/update_actor_total",
    "hpf/follower_optimizer_steps",
    "hpf/leader_optimizer_steps",
)

TIMING_METRICS = tuple(metric for metric in DEFAULT_METRICS if metric.startswith("timing_s/"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True, help="W&B entity, for example zhongal-hkust")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run-id", required=True, help="W&B run ID, not the display name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reports (default: outputs/wandb_diagnostics/<run-id>)",
    )
    return parser.parse_args()


def as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metric_summary(points: list[tuple[int, float]]) -> dict[str, float | int]:
    values = [value for _, value in points]
    steps = [step for step, _ in points]
    return {
        "count": len(values),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first": values[0],
        "last": values[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p90": percentile(values, 0.9),
    }


def collect_history(run: Any) -> tuple[dict[str, list[tuple[int, float]]], list[int]]:
    metrics: dict[str, list[tuple[int, float]]] = defaultdict(list)
    logged_steps: list[int] = []

    for row in run.scan_history():
        raw_step = row.get("training/global_step", row.get("_step"))
        step_value = as_finite_float(raw_step)
        if step_value is None:
            continue
        step = int(step_value)
        logged_steps.append(step)
        for metric in DEFAULT_METRICS:
            value = as_finite_float(row.get(metric))
            if value is not None:
                metrics[metric].append((step, value))

    # W&B may yield more than one row per global step. Keep the latest value.
    deduplicated: dict[str, list[tuple[int, float]]] = {}
    for metric, points in metrics.items():
        by_step = {step: value for step, value in points}
        deduplicated[metric] = sorted(by_step.items())
    return deduplicated, sorted(set(logged_steps))


def horizon_segments(points: list[tuple[int, float]]) -> list[dict[str, float | int]]:
    if not points:
        return []
    segments: list[dict[str, float | int]] = []
    start_step, horizon = points[0]
    end_step = start_step
    for step, value in points[1:]:
        if value != horizon:
            segments.append({"start_step": start_step, "end_step": end_step, "horizon_tokens": horizon})
            start_step, horizon = step, value
        end_step = step
    segments.append({"start_step": start_step, "end_step": end_step, "horizon_tokens": horizon})
    return segments


def diagnostics(summaries: dict[str, dict[str, float | int]], metrics: dict[str, list[tuple[int, float]]]) -> list[str]:
    notes: list[str] = []
    ratio_max = summaries.get("hpf/correction_ratio_max")
    if ratio_max:
        max_value = float(ratio_max["max"])
        if max_value >= math.exp(4.9):
            notes.append("Correction ratio reached the configured exp(5) clipping boundary.")
        elif max_value > 10:
            notes.append("Correction ratio is occasionally large (>10); inspect its max curve.")
        else:
            notes.append("Correction ratio remained below 10 in this run.")

    clipped_fraction = summaries.get("hpf/correction_clip_frac")
    if clipped_fraction:
        notes.append(
            "Mean correction clipping fraction is "
            f"{float(clipped_fraction['mean']):.2%} (max {float(clipped_fraction['max']):.2%})."
        )

    leader_kl = summaries.get("hpf/leader/actor/hpf_kl_loss")
    leader_steps = summaries.get("hpf/leader_optimizer_steps")
    if leader_kl and float(leader_kl["max"]) == 0.0:
        if leader_steps and float(leader_steps["max"]) <= 1:
            notes.append("Leader KL is zero because each leader phase has only one optimizer step.")
        else:
            notes.append("Leader KL is zero despite multiple leader optimizer steps; inspect the leader KL mask and reference.")

    total = summaries.get("timing_s/hpf/update_actor_total")
    rollout = summaries.get("timing_s/hpf/tree_rollout_total_wall")
    if total and rollout:
        notes.append(
            "Mean HPF actor update wall time is "
            f"{float(total['mean']):.1f}s; mean tree rollout wall time is {float(rollout['mean']):.1f}s."
        )

    validation = metrics.get("val-core/aime_2024_dapo_boxed/acc/mean@1", [])
    if validation:
        notes.append(
            "AIME-24 has only 30 problems, so short-run validation points are high-variance and should not be "
            "interpreted as a performance trend."
        )
    return notes


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HPF W&B Diagnostic Report",
        "",
        f"- Run: `{report['run_path']}`",
        f"- Retrieved: `{report['retrieved_at_utc']}`",
        f"- Logged global steps: `{report['global_steps']}`",
        "",
        "## Horizon Schedule",
        "",
    ]
    segments = report["horizon_segments"]
    if segments:
        lines.extend(["| Start | End | Horizon tokens |", "|---:|---:|---:|"])
        lines.extend(
            f"| {segment['start_step']} | {segment['end_step']} | {segment['horizon_tokens']:.0f} |"
            for segment in segments
        )
    else:
        lines.append("`hpf/tree_horizon_tokens` was not found.")

    lines.extend(["", "## Metric Summary", "", "| Metric | N | Steps | First | Last | Mean | Min | Max |", "|---|---:|---|---:|---:|---:|---:|---:|"])
    for metric, summary in report["metrics"].items():
        lines.append(
            f"| `{metric}` | {summary['count']} | {summary['first_step']}-{summary['last_step']} | "
            f"{summary['first']:.6g} | {summary['last']:.6g} | {summary['mean']:.6g} | "
            f"{summary['min']:.6g} | {summary['max']:.6g} |"
        )

    lines.extend(["", "## Diagnostics", ""])
    lines.extend(f"- {note}" for note in report["diagnostics"])
    lines.append("")
    lines.append("## Missing Expected Metrics")
    lines.append("")
    if report["missing_metrics"]:
        lines.extend(f"- `{metric}`" for metric in report["missing_metrics"])
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required. Activate the verl environment before running this script.") from exc

    run_path = f"{args.entity}/{args.project}/{args.run_id}"
    run = wandb.Api().run(run_path)
    metrics, logged_steps = collect_history(run)
    summaries = {metric: metric_summary(points) for metric, points in metrics.items()}
    output_dir = args.output_dir or Path("outputs/wandb_diagnostics") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_path": run_path,
        "run_name": run.name,
        "run_state": run.state,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "global_steps": logged_steps,
        "metrics": summaries,
        "horizon_segments": horizon_segments(metrics.get("hpf/tree_horizon_tokens", [])),
        "diagnostics": diagnostics(summaries, metrics),
        "missing_metrics": [metric for metric in DEFAULT_METRICS if metric not in summaries],
    }
    markdown = markdown_report(report)
    json_path = output_dir / "hpf_wandb_diagnostic.json"
    markdown_path = output_dir / "hpf_wandb_diagnostic.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown)

    print(markdown, end="")
    print(f"Wrote JSON report to {json_path}")
    print(f"Wrote Markdown report to {markdown_path}")


if __name__ == "__main__":
    main()
