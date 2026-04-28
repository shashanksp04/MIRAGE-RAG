#!/usr/bin/env python3
"""Collect unique US states from meta_data_state in Datasets/standard/standard_benchmark.json."""
# to run it you run the command: "python list_benchmark_states.py"

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "Datasets" / "standard" / "standard_benchmark.json",
        help="Path to standard_benchmark.json (default: <repo>/Datasets/standard/standard_benchmark.json)",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Print record counts per state after the unique list",
    )
    args = parser.parse_args()

    path: Path = args.dataset
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: JSON root must be an array of records", file=sys.stderr)
        return 1

    states: list[str] = []
    missing = 0
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            print(f"Warning: skipping non-object at index {i}", file=sys.stderr)
            continue
        raw = rec.get("meta_data_state")
        if raw is None or raw == "":
            missing += 1
            continue
        states.append(str(raw).strip())

    unique = sorted(set(states))
    for s in unique:
        print(s, flush=True)

    if missing:
        print(f"\n# Records with missing or empty meta_data_state: {missing}", file=sys.stderr)

    if args.counts:
        print("\n# Counts per state:", file=sys.stderr)
        for state, n in Counter(states).most_common():
            print(f"  {state}\t{n}", file=sys.stderr)

    print(f"\n# Unique states: {len(unique)}", file=sys.stderr)
    print(f"# Total records with meta_data_state: {len(states)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
