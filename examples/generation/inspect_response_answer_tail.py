#!/usr/bin/env python3
"""Inspect whether generated responses end with an answer-like sentence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


BOXED_RE = re.compile(r"\\boxed\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
ANSWER_RE = re.compile(
    r"(?:answer|final answer|therefore|thus|so|hence|答案|最终答案|所以|因此)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Experiment output dir containing *_loglik.parquet files.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>/tail_inspection.")
    parser.add_argument("--tail-chars", type=int, default=500)
    parser.add_argument("--examples-per-group", type=int, default=20)
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


def last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def tail_sentence(text: str, tail_chars: int) -> str:
    tail = str(text)[-tail_chars:].strip()
    parts = re.split(r"(?<=[.!?。！？])\s+", tail)
    parts = [part.strip() for part in parts if part.strip()]
    return parts[-1] if parts else tail


def normalize_answer(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.strip("$")
    text = re.sub(r"^\\boxed\s*\{|\}$", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def contains_extracted_answer(text: str, extracted_answer: object) -> bool:
    answer = normalize_answer(extracted_answer)
    if not answer:
        return False
    compact_text = re.sub(r"\s+", "", str(text))
    return answer in compact_text


def add_tail_features(df: pd.DataFrame, tail_chars: int) -> pd.DataFrame:
    out = df.copy()
    out["last_nonempty_line"] = out["response"].map(last_nonempty_line)
    out["tail_sentence"] = out["response"].map(lambda text: tail_sentence(text, tail_chars))
    out["tail_text"] = out["response"].map(lambda text: str(text)[-tail_chars:].strip())
    out["last_line_has_boxed"] = out["last_nonempty_line"].map(lambda text: bool(BOXED_RE.search(text)))
    out["tail_sentence_has_boxed"] = out["tail_sentence"].map(lambda text: bool(BOXED_RE.search(text)))
    out["tail_has_boxed"] = out["tail_text"].map(lambda text: bool(BOXED_RE.search(text)))
    out["last_line_answer_like"] = out["last_nonempty_line"].map(lambda text: bool(ANSWER_RE.search(text)))
    out["tail_sentence_answer_like"] = out["tail_sentence"].map(lambda text: bool(ANSWER_RE.search(text)))
    if "extracted_answer" in out.columns:
        out["last_line_contains_extracted_answer"] = out.apply(
            lambda row: contains_extracted_answer(row["last_nonempty_line"], row["extracted_answer"]),
            axis=1,
        )
        out["tail_sentence_contains_extracted_answer"] = out.apply(
            lambda row: contains_extracted_answer(row["tail_sentence"], row["extracted_answer"]),
            axis=1,
        )
        out["tail_contains_extracted_answer"] = out.apply(
            lambda row: contains_extracted_answer(row["tail_text"], row["extracted_answer"]),
            axis=1,
        )
    else:
        out["last_line_contains_extracted_answer"] = False
        out["tail_sentence_contains_extracted_answer"] = False
        out["tail_contains_extracted_answer"] = False
    out["last_line_is_answer_sentence"] = (
        out["last_line_has_boxed"] | out["last_line_contains_extracted_answer"] | out["last_line_answer_like"]
    )
    out["tail_sentence_is_answer_sentence"] = (
        out["tail_sentence_has_boxed"]
        | out["tail_sentence_contains_extracted_answer"]
        | out["tail_sentence_answer_like"]
    )
    return out


def mean_bool(df: pd.DataFrame, column: str) -> float | None:
    if df.empty:
        return None
    return float(df[column].mean())


def summarize(df: pd.DataFrame) -> dict[str, object]:
    correct = df[df["is_correct"]]
    incorrect = df[~df["is_correct"]]
    feature_cols = [
        "last_line_has_boxed",
        "tail_sentence_has_boxed",
        "tail_has_boxed",
        "last_line_contains_extracted_answer",
        "tail_sentence_contains_extracted_answer",
        "tail_contains_extracted_answer",
        "last_line_is_answer_sentence",
        "tail_sentence_is_answer_sentence",
    ]
    summary: dict[str, object] = {
        "num_responses": int(len(df)),
        "num_problems": int(df["problem_key"].nunique()),
        "num_correct": int(df["is_correct"].sum()),
    }
    for column in feature_cols:
        summary[column] = mean_bool(df, column)
        summary[f"{column}_correct"] = mean_bool(correct, column)
        summary[f"{column}_incorrect"] = mean_bool(incorrect, column)
    return summary


def write_examples(df: pd.DataFrame, output_dir: Path, examples_per_group: int) -> None:
    columns = [
        "problem_index",
        "sample_index",
        "is_correct",
        "model_avg_log_likelihood",
        "extracted_answer",
        "last_nonempty_line",
        "tail_sentence",
        "last_line_has_boxed",
        "last_line_contains_extracted_answer",
        "tail_sentence_contains_extracted_answer",
        "last_line_is_answer_sentence",
        "tail_sentence_is_answer_sentence",
        "response",
    ]
    available = [column for column in columns if column in df.columns]
    samples = []
    for correct_value in [True, False]:
        group = df[df["is_correct"] == correct_value]
        if not group.empty:
            samples.append(group.head(examples_per_group)[available])
    if samples:
        pd.concat(samples, ignore_index=True).to_csv(output_dir / "tail_examples.csv", index=False)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "tail_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_tail_features(load_rows(input_dir), args.tail_chars)
    summary = summarize(df)
    feature_path = output_dir / "response_tail_features.parquet"
    summary_path = output_dir / "response_tail_summary.json"
    df.to_parquet(feature_path, index=False)
    write_examples(df, output_dir, args.examples_per_group)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote row-level features to {feature_path}")
    print(f"Wrote examples to {output_dir / 'tail_examples.csv'}")


if __name__ == "__main__":
    main()
