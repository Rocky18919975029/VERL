#!/usr/bin/env python3
"""Build a seed-fixed DAPO parquet after applying VERL prompt filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source parquet file.")
    parser.add_argument("--output", required=True, help="Output sampled parquet file.")
    parser.add_argument("--model", required=True, help="Tokenizer/model path used by training.")
    parser.add_argument("--sample-size", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--filter-workers", type=int, default=1)
    parser.add_argument("--truncation", default="left")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = hf_tokenizer(args.model, trust_remote_code=True)
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": args.filter_workers,
            "truncation": args.truncation,
            "shuffle": False,
            "seed": args.seed,
        }
    )
    dataset = RLHFDataset(
        data_files=[str(input_path)],
        tokenizer=tokenizer,
        config=config,
        processor=None,
        max_samples=-1,
    )

    filtered = dataset.dataframe
    filtered_len = len(filtered)
    if filtered_len < args.sample_size:
        raise ValueError(f"Need {args.sample_size} filtered rows, but only found {filtered_len}.")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(filtered_len, size=args.sample_size, replace=False)
    sampled = filtered.select(indices.tolist())
    sampled_df = sampled.to_pandas()
    sampled_df.to_parquet(output_path, index=False)

    verify_df = pd.read_parquet(output_path)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "source_filtered_rows": filtered_len,
        "output_rows": len(verify_df),
        "max_prompt_length": args.max_prompt_length,
        "filter_workers": args.filter_workers,
        "truncation": args.truncation,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
