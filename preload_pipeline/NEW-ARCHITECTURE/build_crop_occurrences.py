#!/usr/bin/env python3
"""
Build or refresh one cumulative crop_occurrences.json from
county_crops_frequency_multi_year_cleaned.csv.

The output is intentionally a single state-keyed JSON file that can later be
enriched in place by successive MetaMIRAGE state runs.

Behavior:
- Aggregates county-level crop frequencies into state-level occurrence totals.
- Creates the 50 U.S. state keys even when a state is absent from the CSV.
- Preserves any existing disease/pests/management enrichment already present.
- Preserves crops previously added by the notebook even if they are not in the CSV.
- Updates only occurrence values sourced from the CSV.
- Writes atomically so an interrupted write does not corrupt the JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def parse_crop_frequencies(value: str) -> Dict[str, int]:
    """
    Parse:
        'Corn:19; Soybeans:31; Winter_Wheat:12'
    into:
        {'corn': 19, 'soybeans': 31, 'winter_wheat': 12}
    """
    result: Dict[str, int] = {}

    for item in str(value or "").split(";"):
        item = item.strip()
        if not item:
            continue

        if ":" not in item:
            raise ValueError(f"Malformed crop-frequency item: {item!r}")

        crop_raw, count_raw = item.rsplit(":", 1)
        crop = normalize_name(crop_raw)

        if not crop:
            continue

        try:
            count = int(str(count_raw).strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid occurrence count in item {item!r}"
            ) from exc

        result[crop] = result.get(crop, 0) + count

    return result


def aggregate_state_occurrences(csv_path: Path) -> Dict[str, Dict[str, int]]:
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {"state", "crops"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            state = normalize_name(row.get("state"))

            if not state:
                continue

            try:
                parsed = parse_crop_frequencies(row.get("crops", ""))
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{row_number}: {exc}"
                ) from exc

            for crop, count in parsed.items():
                totals[state][crop] += count

    return {
        state: dict(crops)
        for state, crops in totals.items()
    }


def load_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Existing output must contain a top-level JSON object: {path}"
        )

    return raw


def blank_crop_entry(occurrence: int = 0) -> Dict[str, Any]:
    return {
        "occurrence": int(occurrence),
        "disease": {},
        "pests": {},
        "management": {},
    }


def merge_occurrences(
    existing: Dict[str, Any],
    aggregated: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    # Preserve existing state data, including enrichment from completed runs.
    for state, value in existing.items():
        state_key = normalize_name(state)
        if state_key:
            output[state_key] = value if isinstance(value, dict) else {}

    # Guarantee all 50 states exist. Alaska/Hawaii are absent from the source
    # CSV, so they begin as empty state objects.
    for state in US_STATES:
        output.setdefault(normalize_name(state), {})

    # Retain any additional jurisdiction appearing in the source, e.g. DC.
    for state in aggregated:
        output.setdefault(state, {})

    # Refresh occurrence totals while preserving enrichment.
    for state, crops in aggregated.items():
        state_entry = output.setdefault(state, {})

        for crop, occurrence in crops.items():
            current = state_entry.get(crop)

            if not isinstance(current, dict):
                current = blank_crop_entry(occurrence)
                state_entry[crop] = current
            else:
                current.setdefault("disease", {})
                current.setdefault("pests", {})
                current.setdefault("management", {})
                current["occurrence"] = int(occurrence)

    # Preserve notebook-discovered crops that are not in the occurrence CSV.
    for state_entry in output.values():
        if not isinstance(state_entry, dict):
            continue

        for crop, current in list(state_entry.items()):
            if not isinstance(current, dict):
                state_entry[crop] = blank_crop_entry(0)
                continue

            current.setdefault("occurrence", 0)
            current.setdefault("disease", {})
            current.setdefault("pests", {})
            current.setdefault("management", {})

    return output


def write_atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create/update one cumulative crop_occurrences.json from the "
            "county-level multi-year crop-frequency CSV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("county_crops_frequency_multi_year_cleaned.csv"),
        help="Input county crop-frequency CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("crop_occurrences.json"),
        help="Single cumulative JSON output.",
    )

    args = parser.parse_args()
    csv_path = args.input.resolve()
    output_path = args.output.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    aggregated = aggregate_state_occurrences(csv_path)
    existing = load_existing(output_path)
    merged = merge_occurrences(existing, aggregated)
    write_atomic_json(output_path, merged)

    crop_count = sum(
        len(crops)
        for crops in merged.values()
        if isinstance(crops, dict)
    )

    print(f"Input:  {csv_path}")
    print(f"Output: {output_path}")
    print(f"Jurisdictions in output: {len(merged)}")
    print(f"Total state/crop entries: {crop_count}")
    print("Existing enrichment fields were preserved.")


if __name__ == "__main__":
    main()
