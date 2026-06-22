#!/usr/bin/env python3
"""Analyze fixed token prefix/suffix splits on existing sampled responses.

This is an offline diagnostic for prefix-follower RLVR experiments. It reads
previous *_loglik.parquet outputs, splits each response at candidate prefix
token horizons, and measures whether answer-like content is in the prefix or
suffix for correct and incorrect responses.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Protocol

import pandas as pd


BOXED_RE = re.compile(r"\\boxed\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
ANSWER_RE = re.compile(
    r"(?:answer|final answer|therefore|thus|so|hence|答案|最终答案|所以|因此)\b",
    re.IGNORECASE,
)


class TokenizerLike(Protocol):
    def split(self, text: str, prefix_tokens: int) -> tuple[str, str, int, int]: ...


class WhitespaceTokenizer:
    def split(self, text: str, prefix_tokens: int) -> tuple[str, str, int, int]:
        words = str(text).split()
        prefix = " ".join(words[:prefix_tokens])
        suffix = " ".join(words[prefix_tokens:])
        return prefix, suffix, len(words[:prefix_tokens]), len(words[prefix_tokens:])


class HFTokenizer:
    def __init__(self, model_path: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)

    def split(self, text: str, prefix_tokens: int) -> tuple[str, str, int, int]:
        token_ids = self.tokenizer(str(text), add_special_tokens=False).input_ids
        prefix_ids = token_ids[:prefix_tokens]
        suffix_ids = token_ids[prefix_tokens:]
        prefix = self.tokenizer.decode(prefix_ids, skip_special_tokens=False)
        suffix = self.tokenizer.decode(suffix_ids, skip_special_tokens=False)
        return prefix, suffix, len(prefix_ids), len(suffix_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Experiment output dir containing *_loglik.parquet files.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>/prefix_suffix_analysis.")
    parser.add_argument(
        "--model",
        default=None,
        help="Local HF tokenizer/model path. Use this for real token splits.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["hf", "whitespace"],
        default="hf",
        help="Use 'whitespace' only for script smoke tests.",
    )
    parser.add_argument("--prefix-tokens", default="128,256,512,768,1024")
    parser.add_argument("--examples-per-split", type=int, default=20)
    return parser.parse_args()


def load_rows(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("shard_*/*_loglik.parquet"))
    if not paths:
        paths = sorted(input_dir.glob("*_loglik.parquet"))
    if not paths:
        raise FileNotFoundError(f"No *_loglik.parquet files found under {input_dir}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    required = {"response", "is_correct", "problem_index", "sample_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["is_correct"] = df["is_correct"].astype(bool)
    if "raw_problem" in df.columns and df["raw_problem"].notna().any():
        df["problem_key"] = df["raw_problem"].astype(str)
    else:
        df["problem_key"] = df["problem_index"].astype(str)
    return df


def make_tokenizer(args: argparse.Namespace) -> TokenizerLike:
    if args.tokenizer == "whitespace":
        return WhitespaceTokenizer()
    if not args.model:
        raise ValueError("--model is required when --tokenizer=hf")
    return HFTokenizer(args.model)


def normalize_answer(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().strip("$")
    text = re.sub(r"^\\boxed\s*\{|\}$", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def compact(text: object) -> str:
    return re.sub(r"\s+", "", str(text))


def contains_extracted_answer(text: str, extracted_answer: object) -> bool:
    answer = normalize_answer(extracted_answer)
    return bool(answer and answer in compact(text))


def answer_like(text: str, extracted_answer: object) -> bool:
    return bool(BOXED_RE.search(text) or ANSWER_RE.search(text) or contains_extracted_answer(text, extracted_answer))


def add_split_features(df: pd.DataFrame, tokenizer: TokenizerLike, prefix_tokens: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        response = str(row["response"])
        prefix_text, suffix_text, actual_prefix_tokens, actual_suffix_tokens = tokenizer.split(response, prefix_tokens)
        extracted_answer = row.get("extracted_answer")
        output = row.to_dict()
        output.update(
            {
                "split_prefix_tokens": int(prefix_tokens),
                "actual_response_tokens": int(actual_prefix_tokens + actual_suffix_tokens),
                "actual_prefix_tokens": int(actual_prefix_tokens),
                "actual_suffix_tokens": int(actual_suffix_tokens),
                "has_suffix": bool(actual_suffix_tokens > 0),
                "prefix_text": prefix_text,
                "suffix_text": suffix_text,
                "prefix_has_boxed": bool(BOXED_RE.search(prefix_text)),
                "suffix_has_boxed": bool(BOXED_RE.search(suffix_text)),
                "prefix_contains_extracted_answer": contains_extracted_answer(prefix_text, extracted_answer),
                "suffix_contains_extracted_answer": contains_extracted_answer(suffix_text, extracted_answer),
                "prefix_answer_like": answer_like(prefix_text, extracted_answer),
                "suffix_answer_like": answer_like(suffix_text, extracted_answer),
            }
        )
        rows.append(output)
    return pd.DataFrame(rows)


def mean_bool(df: pd.DataFrame, column: str) -> float | None:
    if df.empty:
        return None
    return float(df[column].mean())


def summarize_split(split_df: pd.DataFrame, prefix_tokens: int) -> dict[str, object]:
    correct = split_df[split_df["is_correct"]]
    incorrect = split_df[~split_df["is_correct"]]
    answer_cols = [
        "has_suffix",
        "prefix_has_boxed",
        "suffix_has_boxed",
        "prefix_contains_extracted_answer",
        "suffix_contains_extracted_answer",
        "prefix_answer_like",
        "suffix_answer_like",
    ]
    summary: dict[str, object] = {
        "prefix_tokens": int(prefix_tokens),
        "num_responses": int(len(split_df)),
        "num_problems": int(split_df["problem_key"].nunique()),
        "num_correct": int(split_df["is_correct"].sum()),
        "response_accuracy": float(split_df["is_correct"].mean()),
        "mean_response_tokens": float(split_df["actual_response_tokens"].mean()),
        "mean_suffix_tokens": float(split_df["actual_suffix_tokens"].mean()),
        "mean_suffix_tokens_correct": None if correct.empty else float(correct["actual_suffix_tokens"].mean()),
        "mean_suffix_tokens_incorrect": None if incorrect.empty else float(incorrect["actual_suffix_tokens"].mean()),
    }
    for column in answer_cols:
        summary[column] = mean_bool(split_df, column)
        summary[f"{column}_correct"] = mean_bool(correct, column)
        summary[f"{column}_incorrect"] = mean_bool(incorrect, column)
    return summary


def write_examples(features: pd.DataFrame, output_dir: Path, examples_per_split: int) -> None:
    columns = [
        "split_prefix_tokens",
        "problem_index",
        "sample_index",
        "is_correct",
        "actual_response_tokens",
        "actual_suffix_tokens",
        "extracted_answer",
        "prefix_contains_extracted_answer",
        "suffix_contains_extracted_answer",
        "prefix_answer_like",
        "suffix_answer_like",
        "prefix_text",
        "suffix_text",
        "response",
    ]
    available = [column for column in columns if column in features.columns]
    samples = []
    for prefix_tokens, group in features.groupby("split_prefix_tokens", sort=True):
        for correct_value in [True, False]:
            subset = group[group["is_correct"] == correct_value]
            if not subset.empty:
                samples.append(subset.head(examples_per_split)[available])
    if samples:
        pd.concat(samples, ignore_index=True).to_csv(output_dir / "prefix_suffix_examples.csv", index=False)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "prefix_suffix_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_values = sorted({int(value) for value in args.prefix_tokens.split(",") if value.strip()})
    tokenizer = make_tokenizer(args)
    df = load_rows(input_dir)

    feature_frames = [add_split_features(df, tokenizer, prefix_tokens) for prefix_tokens in prefix_values]
    features = pd.concat(feature_frames, ignore_index=True)
    summaries = [summarize_split(features[features["split_prefix_tokens"] == value], value) for value in prefix_values]

    features.to_parquet(output_dir / "prefix_suffix_features.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "prefix_suffix_summary.csv", index=False)
    write_examples(features, output_dir, args.examples_per_split)
    (output_dir / "prefix_suffix_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Wrote summary to {output_dir / 'prefix_suffix_summary.json'}")
    print(f"Wrote per-response features to {output_dir / 'prefix_suffix_features.parquet'}")
    print(f"Wrote examples to {output_dir / 'prefix_suffix_examples.csv'}")


if __name__ == "__main__":
    main()
