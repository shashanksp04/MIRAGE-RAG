#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
import sys

# Default category lists for classification
IDENTIFICATION_CATEGORIES = [
    "Plant Identification",
    "Insect and Pest Identification",
    "Plant Disease Identification",
]
MANAGEMENT_CATEGORIES = [
    "Plant Care and Gardening Guidance",
    "Plant Disease Management",
    "Insect and Pest Management",
    "Weeds/Invasive Plants Management",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split inference results for a single model into identification vs. management buckets"
    )
    parser.add_argument(
        "--bench_type",
        type=str,
        default="standard",
        help="Benchmark type, e.g. 'standard' or 'contextual'",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to process (e.g. 'gpt-4o-mini')",
    )
    parser.add_argument(
        "--raw_data_path",
        type=Path,
        required=True,
        help="Path to the raw benchmark JSON, e.g. ../Datasets/sample_bench/sample_standard_benchmark.json",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing the model's inference JSON file, e.g. ../Inference/results/standard",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where split JSONs will be saved, e.g. ../Datasets/sample_inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print()
    print("#"* 80)
    print(f"Splitting inference results for model: {args.model_name}")
    # Verify raw data file exists
    if not args.raw_data_path.is_file():
        sys.exit(f"ERROR: Raw data file not found: {args.raw_data_path}")
    # Verify results directory exists
    if not args.results_dir.is_dir():
        sys.exit(f"ERROR: Results directory not found: {args.results_dir}")
    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data and initialize model field to None
    raw_data_path = json.loads(args.raw_data_path.read_text())
    all_data = {
        sample["id"]: {**sample, args.model_name: None}
        for sample in raw_data_path
    }

    # Read inference results for the specified model
    result_file = args.results_dir / f"{args.model_name}.json"
    if not result_file.exists():
        print(f"WARNING: File not found for model {args.model_name}: {result_file}")
    else:
        with result_file.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    sample_id = rec.get("id")
                    if sample_id in all_data:
                        all_data[sample_id][args.model_name] = rec.get(args.model_name)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {result_file}")

    # Split samples by category
    identification_data = []
    management_data = []
    for sample in all_data.values():
        category = sample.get("category", "")
        if category in IDENTIFICATION_CATEGORIES:
            identification_data.append(sample)
        elif category in MANAGEMENT_CATEGORIES:
            management_data.append(sample)

    # Save split results
    id_output = args.output_dir / f"{args.bench_type}_benchmark_ID_{args.model_name}.json"
    mg_output = args.output_dir / f"{args.bench_type}_benchmark_MG_{args.model_name}.json"

    if identification_data:
        json.dump(
            [s for s in identification_data if s.get(args.model_name) is not None],
            id_output.open("w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
        print(f"Saved {len(identification_data)} identification samples to {id_output}")
    if management_data:
        json.dump(
            [s for s in management_data if s.get(args.model_name) is not None],
            mg_output.open("w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
        print(f"Saved {len(management_data)} management samples to {mg_output}")
    print(f"Total samples: {len(all_data)}")
    print(f"Identification samples: {len(identification_data)}")
    print(f"Management samples:     {len(management_data)}")
    missing = sum(1 for s in all_data.values() if s.get(args.model_name) is None)
    print(f"Missing results for {args.model_name}: {missing}")

if __name__ == "__main__":
    main()