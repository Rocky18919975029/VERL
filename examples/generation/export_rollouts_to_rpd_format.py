#!/usr/bin/env python3
"""Export VERL rollout parquet files to the official RPD repository format.

The Reasoning-Path-Divergence repository expects Step 4 inputs under
`data/processed/quality_filtered_data_*.parquet` with one row per problem:

    question, answer_0, answer_1, ...

This adapter reads our rollout/loglik parquet outputs, groups responses by
problem_index, and writes exactly that shape. It does not compute RPD itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="VERL rollout output directory.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Official RPD repo processed directory, or any directory later assigned to config.PROCESSED_DATA_DIR.",
    )
    parser.add_argument("--output-prefix", default="quality_filtered_data")
    parser.add_argument("--responses-per-problem", type=int, default=None, help="Optional cap per problem.")
    parser.add_argument("--correct-only", action="store_true", help="Export only answer-correct responses.")
    parser.add_argument("--min-responses", type=int, default=2, help="Drop problems with fewer responses after filtering.")
    parser.add_argument("--max-rows-per-file", type=int, default=10000)
    parser.add_argument("--num-shards", type=int, default=1, help="Split exported problems into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Export only this 0-based shard index.")
    return parser.parse_args()


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


def to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def question_text(row: pd.Series) -> str:
    raw_problem = row.get("raw_problem")
    if isinstance(raw_problem, str) and raw_problem.strip():
        return raw_problem

    prompt = to_builtin(row.get("prompt"))
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        text = "\n".join(part for part in parts if part)
        if text.strip():
            return text
    return str(prompt)


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ["problem_index", "prefix_index", "suffix_index", "sample_index", "trajectory_index"] if col in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols, kind="mergesort")
    return df.sort_values("problem_index", kind="mergesort")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")

    paths = find_parquet_files(input_dir)
    if not paths:
        raise FileNotFoundError(f"No rollout/loglik parquet files found under {input_dir}")

    df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    required = {"problem_index", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if args.correct_only:
        if "is_correct" not in df.columns:
            raise ValueError("--correct-only requires an is_correct column")
        df = df[df["is_correct"].astype(bool)].copy()

    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for rpd_idx, (problem_index, group) in enumerate(sort_columns(df).groupby("problem_index", sort=False)):
        group = sort_columns(group)
        if args.responses_per_problem is not None:
            group = group.head(args.responses_per_problem)
        responses = [str(text) for text in group["response"].tolist() if isinstance(text, str) and text.strip()]
        if len(responses) < args.min_responses:
            continue

        first = group.iloc[0]
        out_row: dict[str, Any] = {
            "question": question_text(first),
            "problem_index": int(problem_index),
        }
        if "ground_truth" in group.columns:
            out_row["ground_truth"] = first.get("ground_truth")
        for answer_id, response in enumerate(responses):
            out_row[f"answer_{answer_id}"] = response
        rows.append(out_row)
        manifest.append(
            {
                "rpd_global_idx": len(rows) - 1,
                "problem_index": int(problem_index),
                "num_answers": len(responses),
                "source_files": [str(path) for path in paths],
            }
        )

    if not rows:
        raise ValueError("No problems left after filtering.")

    total_exported_before_shard = len(rows)
    if args.num_shards > 1:
        selected = [
            (row, item)
            for original_idx, (row, item) in enumerate(zip(rows, manifest, strict=True))
            if original_idx % args.num_shards == args.shard_index
        ]
        rows = [row for row, _ in selected]
        manifest = [item for _, item in selected]
        for local_idx, item in enumerate(manifest):
            item["source_rpd_global_idx"] = item["rpd_global_idx"]
            item["rpd_global_idx"] = local_idx
            item["export_num_shards"] = args.num_shards
            item["export_shard_index"] = args.shard_index
    if not rows:
        raise ValueError(f"No problems left for shard {args.shard_index}/{args.num_shards}.")

    for file_idx, start in enumerate(range(0, len(rows), args.max_rows_per_file)):
        chunk = pd.DataFrame(rows[start : start + args.max_rows_per_file])
        out_path = output_dir / f"{args.output_prefix}_{file_idx:05d}.parquet"
        chunk.to_parquet(out_path, index=False)
        print(f"Wrote {len(chunk)} rows to {out_path}")

    manifest_path = output_dir / "rpd_rollout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_source_files": len(paths),
        "num_source_rows": int(len(df)),
        "num_exported_problems_before_shard": total_exported_before_shard,
        "num_exported_problems": len(rows),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "min_responses": args.min_responses,
        "responses_per_problem": args.responses_per_problem,
        "correct_only": args.correct_only,
        "manifest": str(manifest_path),
    }
    summary_path = output_dir / "rpd_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
