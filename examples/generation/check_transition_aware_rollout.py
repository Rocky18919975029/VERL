#!/usr/bin/env python3
"""Validate paired current/next-cut rollout dumps without loading a model."""

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing rollout dump: {path}")
    return [json.loads(line) for line in path.open() if line.strip()]


def index_pairs(rows: list[dict], expected_cut: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        pair_uid = row.get("hpf_transition_pair_uid")
        if not pair_uid:
            raise ValueError(f"Missing hpf_transition_pair_uid in {expected_cut}-cut dump.")
        if row.get("hpf_transition_cut") != expected_cut:
            raise ValueError(
                f"Pair {pair_uid} has cut={row.get('hpf_transition_cut')!r}; expected {expected_cut!r}."
            )
        if pair_uid in indexed:
            raise ValueError(f"Duplicate pair UID in {expected_cut}-cut dump: {pair_uid}")
        indexed[pair_uid] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    current_rows = load_rows(args.run_dir / f"{args.step}.jsonl")
    next_rows = load_rows(args.run_dir / "transition_next" / f"{args.step}.jsonl")
    if not current_rows or not next_rows:
        raise ValueError(
            f"Rollout dumps must be nonempty; got current={len(current_rows)}, next={len(next_rows)}."
        )
    if args.expected_rows is not None:
        if len(current_rows) != args.expected_rows or len(next_rows) != args.expected_rows:
            raise ValueError(
                f"Expected {args.expected_rows} rows per cut; "
                f"got current={len(current_rows)}, next={len(next_rows)}."
            )

    current = index_pairs(current_rows, "current")
    next_cut = index_pairs(next_rows, "next")
    if current.keys() != next_cut.keys():
        raise ValueError(
            f"Pair UID sets differ: current_only={len(current.keys() - next_cut.keys())}, "
            f"next_only={len(next_cut.keys() - current.keys())}."
        )

    shared_prefixes = 0
    current_suffix_lengths = []
    next_suffix_lengths = []
    current_horizons = set()
    next_horizons = set()
    for pair_uid, current_row in current.items():
        next_row = next_cut[pair_uid]
        current_prefix = [int(token_id) for token_id in current_row["hpf_prefix_ids"]]
        next_prefix = [int(token_id) for token_id in next_row["hpf_prefix_ids"]]
        current_suffix_lengths.append(len(current_row["hpf_transition_suffix_ids"]))
        next_suffix_lengths.append(len(next_row["hpf_transition_suffix_ids"]))
        if current_prefix != next_prefix[: len(current_prefix)]:
            raise ValueError(f"Current prefix is not a truncation of next prefix for pair {pair_uid}.")
        shared_prefixes += 1
        current_horizons.add(int(current_row["hpf_transition_prefix_horizon"]))
        next_horizons.add(int(next_row["hpf_transition_prefix_horizon"]))

    if len(current_horizons) != 1 or len(next_horizons) != 1:
        raise ValueError(
            f"Expected one horizon per cut, got current={sorted(current_horizons)}, next={sorted(next_horizons)}."
        )
    current_horizon = next(iter(current_horizons))
    next_horizon = next(iter(next_horizons))
    if next_horizon < current_horizon:
        raise ValueError(f"Next horizon {next_horizon} is smaller than current horizon {current_horizon}.")

    current_reward = sum(float(row["score"]) for row in current_rows) / len(current_rows)
    next_reward = sum(float(row["score"]) for row in next_rows) / len(next_rows)
    print("Transition-aware rollout validation passed")
    print(f"rows per cut: {len(current_rows)}")
    print(f"paired prefixes verified: {shared_prefixes}")
    print(f"current horizon: {current_horizon}")
    print(f"next horizon: {next_horizon}")
    print(f"current suffix length mean: {sum(current_suffix_lengths) / len(current_suffix_lengths):.2f}")
    print(f"next suffix length mean: {sum(next_suffix_lengths) / len(next_suffix_lengths):.2f}")
    print(f"current reward mean: {current_reward:.6f}")
    print(f"next reward mean: {next_reward:.6f}")


if __name__ == "__main__":
    main()
