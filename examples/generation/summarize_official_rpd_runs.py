#!/usr/bin/env python3
"""Summarize official RPD outputs for a submitted setting matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="submitted_rpd_jobs.tsv from submit_selected_dapo_rpd_jobs.sh")
    parser.add_argument("--output-dir", required=True, help="Directory for summary CSV/JSON outputs.")
    parser.add_argument(
        "--rpd-output-root",
        default=None,
        help="Override output root. Defaults to manifest parent.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        return np.array([], dtype=np.float32)
    i, j = np.triu_indices(matrix.shape[0], k=1)
    return matrix[i, j]


def valid_summary_count(summary_path: Path) -> tuple[int, int]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    summaries = data.get("summaries", [])
    valid = 0
    for item in summaries:
        steps = item.get("logical_steps", [])
        if isinstance(steps, list) and any(
            isinstance(step, dict) and isinstance(step.get("step_description"), str) and step["step_description"].strip()
            for step in steps
        ):
            valid += 1
    return len(summaries), valid


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(row: dict[str, str], rpd_output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_name = row["run_name"]
    run_dir = rpd_output_root / run_name
    processed_dir = run_dir / "processed"
    summary_dir = processed_dir / "summaries"
    matrix_path = processed_dir / "05_distance_matrix.npz"
    export_summary = load_json(processed_dir / "rpd_export_summary.json")
    manifest_path = processed_dir / "rpd_rollout_manifest.json"
    rollout_manifest = load_json(manifest_path) if manifest_path.exists() else []
    rpd_to_problem = {
        int(item["rpd_global_idx"]): int(item["problem_index"])
        for item in rollout_manifest
        if isinstance(item, dict) and "rpd_global_idx" in item and "problem_index" in item
    }

    problem_rows: list[dict[str, Any]] = []
    total_summaries = 0
    valid_summaries = 0

    if summary_dir.exists():
        for path in sorted(summary_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            total, valid = valid_summary_count(path)
            total_summaries += total
            valid_summaries += valid
            rpd_idx = int(path.stem) if path.stem.isdigit() else None
            problem_rows.append(
                {
                    "kind": row["kind"],
                    "setting": row["setting"],
                    "shard_index": row.get("shard_index", "0"),
                    "num_shards": row.get("num_shards", "1"),
                    "job_id": row["job_id"],
                    "run_name": run_name,
                    "rpd_global_idx": rpd_idx,
                    "problem_index": rpd_to_problem.get(rpd_idx) if rpd_idx is not None else None,
                    "total_summaries": total,
                    "valid_summaries": valid,
                    "valid_summary_frac": valid / total if total else np.nan,
                    "matrix_size": 0,
                    "mean_pairwise_rpd": np.nan,
                    "median_pairwise_rpd": np.nan,
                    "num_pairwise_distances": 0,
                }
            )

    matrix_count = 0
    matrix_problem_count = 0
    all_pairwise_values: list[np.ndarray] = []

    if matrix_path.exists():
        z = np.load(matrix_path)
        problem_by_rpd_idx = {item["rpd_global_idx"]: item for item in problem_rows}
        for key in sorted(k for k in z.files if k.endswith("_matrix")):
            rpd_idx_text = key[len("q_") : -len("_matrix")]
            if not rpd_idx_text.isdigit():
                continue
            rpd_idx = int(rpd_idx_text)
            matrix = z[key]
            values = upper_triangle_values(matrix)
            matrix_count += 1
            if values.size:
                matrix_problem_count += 1
                all_pairwise_values.append(values)
            item = problem_by_rpd_idx.get(rpd_idx)
            if item is not None:
                item["matrix_size"] = int(matrix.shape[0])
                item["mean_pairwise_rpd"] = float(values.mean()) if values.size else np.nan
                item["median_pairwise_rpd"] = float(np.median(values)) if values.size else np.nan
                item["num_pairwise_distances"] = int(values.size)

    concat_values = np.concatenate(all_pairwise_values) if all_pairwise_values else np.array([], dtype=np.float32)
    per_problem_rpd = [
        item["mean_pairwise_rpd"]
        for item in problem_rows
        if item.get("num_pairwise_distances", 0) and np.isfinite(item.get("mean_pairwise_rpd", np.nan))
    ]

    setting_summary = {
        "kind": row["kind"],
        "setting": row["setting"],
        "shard_index": row.get("shard_index", "0"),
        "num_shards": row.get("num_shards", "1"),
        "job_id": row["job_id"],
        "run_name": run_name,
        "input_dir": row["input_dir"],
        "run_dir": str(run_dir),
        "num_exported_problems": export_summary.get("num_exported_problems"),
        "num_source_rows": export_summary.get("num_source_rows"),
        "summary_json_files": len(problem_rows),
        "total_summaries": total_summaries,
        "valid_summaries": valid_summaries,
        "valid_summary_frac": valid_summaries / total_summaries if total_summaries else np.nan,
        "matrix_count": matrix_count,
        "matrix_problem_count": matrix_problem_count,
        "mean_pairwise_rpd_over_pairs": float(concat_values.mean()) if concat_values.size else np.nan,
        "median_pairwise_rpd_over_pairs": float(np.median(concat_values)) if concat_values.size else np.nan,
        "mean_problem_rpd": float(np.mean(per_problem_rpd)) if per_problem_rpd else np.nan,
        "median_problem_rpd": float(np.median(per_problem_rpd)) if per_problem_rpd else np.nan,
        "num_pairwise_distances": int(concat_values.size),
        "matrix_exists": matrix_path.exists(),
    }
    return setting_summary, problem_rows


def aggregate_setting_rows(run_df: pd.DataFrame, problem_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (kind, setting), group in run_df.groupby(["kind", "setting"], dropna=False):
        problem_group = problem_df[(problem_df["kind"] == kind) & (problem_df["setting"] == setting)]
        total_summaries = int(problem_group["total_summaries"].sum()) if not problem_group.empty else 0
        valid_summaries = int(problem_group["valid_summaries"].sum()) if not problem_group.empty else 0
        matrix_problem_count = int((problem_group["num_pairwise_distances"] > 0).sum()) if not problem_group.empty else 0
        num_pairwise_distances = int(problem_group["num_pairwise_distances"].sum()) if not problem_group.empty else 0

        weighted_pairwise = np.nan
        if num_pairwise_distances:
            valid_pairs = problem_group[problem_group["num_pairwise_distances"] > 0]
            weighted_pairwise = float(
                np.average(valid_pairs["mean_pairwise_rpd"], weights=valid_pairs["num_pairwise_distances"])
            )

        per_problem_rpd = (
            problem_group.loc[problem_group["num_pairwise_distances"] > 0, "mean_pairwise_rpd"].astype(float).to_numpy()
            if not problem_group.empty
            else np.array([])
        )

        rows.append(
            {
                "kind": kind,
                "setting": setting,
                "num_rpd_shards": int(len(group)),
                "job_ids": ",".join(str(x) for x in group["job_id"].tolist()),
                "run_names": ",".join(str(x) for x in group["run_name"].tolist()),
                "num_exported_problems": int(group["num_exported_problems"].fillna(0).sum()),
                "summary_json_files": int(group["summary_json_files"].fillna(0).sum()),
                "total_summaries": total_summaries,
                "valid_summaries": valid_summaries,
                "valid_summary_frac": valid_summaries / total_summaries if total_summaries else np.nan,
                "matrix_count": int(group["matrix_count"].fillna(0).sum()),
                "matrix_problem_count": matrix_problem_count,
                "mean_problem_rpd": float(np.mean(per_problem_rpd)) if per_problem_rpd.size else np.nan,
                "median_problem_rpd": float(np.median(per_problem_rpd)) if per_problem_rpd.size else np.nan,
                "mean_pairwise_rpd_over_pairs": weighted_pairwise,
                "num_pairwise_distances": num_pairwise_distances,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    rpd_output_root = Path(args.rpd_output_root) if args.rpd_output_root else manifest.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    for row in read_manifest(manifest):
        run_summary, per_problem = summarize_run(row, rpd_output_root)
        run_rows.append(run_summary)
        problem_rows.extend(per_problem)

    run_df = pd.DataFrame(run_rows)
    problem_df = pd.DataFrame(problem_rows)
    setting_df = aggregate_setting_rows(run_df, problem_df)

    run_csv = output_dir / "official_rpd_run_summary.csv"
    setting_csv = output_dir / "official_rpd_setting_summary.csv"
    problem_csv = output_dir / "official_rpd_per_problem.csv"
    setting_json = output_dir / "official_rpd_setting_summary.json"

    run_df.to_csv(run_csv, index=False)
    setting_df.to_csv(setting_csv, index=False)
    problem_df.to_csv(problem_csv, index=False)
    setting_json.write_text(json.dumps(setting_df.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")

    display_cols = [
        "kind",
        "setting",
        "num_rpd_shards",
        "summary_json_files",
        "valid_summary_frac",
        "matrix_count",
        "mean_problem_rpd",
        "mean_pairwise_rpd_over_pairs",
    ]
    print(setting_df[display_cols].to_string(index=False))
    print(f"Wrote per-run summary to {run_csv}")
    print(f"Wrote setting summary to {setting_csv}")
    print(f"Wrote per-problem summary to {problem_csv}")
    print(f"Wrote JSON summary to {setting_json}")


if __name__ == "__main__":
    main()
