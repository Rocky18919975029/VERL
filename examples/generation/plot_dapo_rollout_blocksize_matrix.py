#!/usr/bin/env python3
"""Summarize and plot DAPO rollout quality across tree block sizes and seeds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Output root from submit_dapo_rollout_blocksize_matrix.sh.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/analysis.")
    parser.add_argument("--metric", action="append", default=None, help="Metric to plot. Defaults to acr,acc,pass_at_8.")
    parser.add_argument("--shade", choices=["std", "sem"], default="std", help="Seed variation band.")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def find_parquet_files(root: Path) -> list[Path]:
    patterns = [
        "shard_*/*_rollouts.parquet",
        "*_rollouts.parquet",
        "shard_*/*_loglik.parquet",
        "*_loglik.parquet",
        "shard_*/*.parquet",
        "*.parquet",
    ]
    for pattern in patterns:
        paths = sorted(root.glob(pattern))
        if paths:
            return paths
    return []


def read_setting(root: Path) -> pd.DataFrame:
    paths = find_parquet_files(root)
    if not paths:
        raise FileNotFoundError(f"No rollout parquet files found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    grouped = df.groupby("problem_index")["is_correct"].agg(["sum", "count"])
    all_correct = int((grouped["sum"] == grouped["count"]).sum())
    all_wrong = int((grouped["sum"] == 0).sum())
    all_groups = int(len(grouped))
    mixed_groups = all_groups - all_correct - all_wrong
    return {
        "groups": all_groups,
        "responses": int(len(df)),
        "group_size_min": int(grouped["count"].min()) if all_groups else 0,
        "group_size_max": int(grouped["count"].max()) if all_groups else 0,
        "all_correct_groups": all_correct,
        "all_wrong_groups": all_wrong,
        "mixed_groups": mixed_groups,
        "acr": (all_correct + all_wrong) / all_groups if all_groups else 0.0,
        "acc": float(df["is_correct"].mean()) if len(df) else 0.0,
        "pass_at_8": float((grouped["sum"] > 0).mean()) if all_groups else 0.0,
    }


def discover_settings(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setting_root in sorted((root / "tree").glob("seed_*/block_*/leader_temp_*")):
        seed = first_match(r"seed_([^/]+)", str(setting_root))
        block_size = first_match(r"block_(\d+)", str(setting_root))
        temp = first_match(r"leader_temp_([^/]+)", str(setting_root))
        if seed is None or block_size is None:
            continue
        rows.append(
            {
                "kind": "tree",
                "seed": int(seed),
                "block_size": int(block_size),
                "temperature_tag": temp,
                "path": setting_root,
            }
        )
    for setting_root in sorted((root / "full").glob("seed_*/temp_*")):
        seed = first_match(r"seed_([^/]+)", str(setting_root))
        temp = first_match(r"temp_([^/]+)", str(setting_root))
        if seed is None:
            continue
        rows.append(
            {
                "kind": "full",
                "seed": int(seed),
                "block_size": None,
                "temperature_tag": temp,
                "path": setting_root,
            }
        )
    return rows


def summarize(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    setting_rows = discover_settings(root)
    if not setting_rows:
        raise FileNotFoundError(f"No tree/full rollout settings found under {root}")

    per_seed_rows = []
    for setting in setting_rows:
        df = read_setting(setting["path"])
        metrics = compute_metrics(df)
        row = {k: v for k, v in setting.items() if k != "path"}
        row["path"] = str(setting["path"])
        row.update(metrics)
        per_seed_rows.append(row)

    per_seed = pd.DataFrame(per_seed_rows).sort_values(["kind", "block_size", "seed"], na_position="last")
    grouped = per_seed.groupby(["kind", "block_size"], dropna=False)
    metric_cols = ["acr", "acc", "pass_at_8", "mixed_groups", "all_correct_groups", "all_wrong_groups"]
    summary_parts = []
    for metric in metric_cols:
        part = grouped[metric].agg(["mean", "std", "count"]).reset_index()
        part["sem"] = part["std"] / part["count"].pow(0.5)
        part["metric"] = metric
        part = part.rename(columns={"mean": "value_mean", "std": "value_std", "count": "num_seeds"})
        summary_parts.append(part)
    summary = pd.concat(summary_parts, ignore_index=True)
    return per_seed, summary


def plot(per_seed: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, metrics: list[str], shade: str, title: str | None) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; skipped plot: {exc}")
        return None

    tree = summary[summary["kind"] == "tree"].copy()
    full = summary[summary["kind"] == "full"].copy()
    tree["block_size"] = tree["block_size"].astype(int)
    blocks = sorted(tree["block_size"].unique())
    if not blocks:
        print("No tree block sizes found; skipped plot.")
        return None

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    labels = {"acr": "ACR", "acc": "Trajectory accuracy", "pass_at_8": "Pass@8"}
    x_min, x_max = min(blocks), max(blocks)
    x_pad = max(1, int((x_max - x_min) * 0.05))

    for ax, metric in zip(axes, metrics):
        tree_metric = tree[tree["metric"] == metric].sort_values("block_size")
        x = tree_metric["block_size"].astype(float)
        y = tree_metric["value_mean"].astype(float)
        band = tree_metric[f"value_{shade}"].fillna(0.0).astype(float)
        ax.plot(x, y, marker="o", label="tree rollout")
        ax.fill_between(x, y - band, y + band, alpha=0.2, label=f"tree ± {shade}")

        full_metric = full[full["metric"] == metric]
        if not full_metric.empty:
            full_mean = float(full_metric["value_mean"].iloc[0])
            full_band = float(full_metric[f"value_{shade}"].fillna(0.0).iloc[0])
            ax.axhline(full_mean, linestyle="--", color="tab:orange", label="full rollout")
            ax.fill_between(
                [x_min - x_pad, x_max + x_pad],
                [full_mean - full_band, full_mean - full_band],
                [full_mean + full_band, full_mean + full_band],
                color="tab:orange",
                alpha=0.15,
                label=f"full ± {shade}",
            )
        ax.set_ylabel(labels.get(metric, metric))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Tree leader block size")
    axes[-1].set_xlim(x_min - x_pad, x_max + x_pad)
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    plot_path = output_dir / "rollout_blocksize_metrics.png"
    fig.savefig(plot_path, dpi=200)
    pdf_path = output_dir / "rollout_blocksize_metrics.pdf"
    fig.savefig(pdf_path)
    print(f"Wrote plot to {plot_path}")
    print(f"Wrote PDF plot to {pdf_path}")
    return plot_path


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = args.metric or ["acr", "acc", "pass_at_8"]

    per_seed, summary = summarize(root)
    per_seed_path = output_dir / "rollout_metrics_per_seed.csv"
    summary_path = output_dir / "rollout_metrics_summary.csv"
    json_path = output_dir / "rollout_metrics_summary.json"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")

    print("Per-seed metrics:")
    print(per_seed.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"Wrote per-seed metrics to {per_seed_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote JSON summary to {json_path}")
    plot(per_seed, summary, output_dir, metrics, args.shade, args.title)


if __name__ == "__main__":
    main()
