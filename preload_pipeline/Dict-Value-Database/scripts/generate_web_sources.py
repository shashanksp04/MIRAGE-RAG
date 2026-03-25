"""
Generate url_batches YAML for build_crop_dictionary.py.

Output format matches YAMLfilesForDict/url_batches_example.yaml:
  batches:
    - state: <state>
      category: <category>
      items:
        - url: "<url>"
          name: "<name>"
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate url_batches YAML for build_crop_dictionary.py from a list of names."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL, e.g. https://en.wikipedia.org/wiki/ or https://extension.illinois.edu/plant-problems/",
    )
    parser.add_argument(
        "--names-file",
        required=True,
        help="Text file with one disease/pest name per line.",
    )
    parser.add_argument(
        "--state",
        required=True,
        help="State name for the batch, e.g. Illinois.",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Category for the batch, e.g. disease or pests.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--url-style",
        choices=["wikipedia", "slug"],
        default="slug",
        help="URL slug style: wikipedia (spaces->underscores) or slug (lowercase hyphenated). Default: slug.",
    )
    return parser.parse_args()


def slugify_wikipedia(name: str) -> str:
    """Convert name to Wikipedia-style URL slug: spaces -> underscores."""
    normalized = (name or "").strip()
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized


def slugify_extension(name: str) -> str:
    """Convert name to extension-style URL slug: lowercase, hyphenated."""
    normalized = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    )
    normalized = normalized.replace("/", "")
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized


def read_names(path: Path) -> List[str]:
    names: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        cleaned = raw.strip()
        if cleaned:
            names.append(cleaned)
    if not names:
        raise ValueError(f"No names found in {path}")
    return names


def build_batch(
    *,
    state: str,
    category: str,
    names: List[str],
    base_url: str,
    url_style: str,
) -> dict:
    base = base_url.rstrip("/")
    items: List[dict] = []

    for name in names:
        if url_style == "wikipedia":
            slug = slugify_wikipedia(name)
        else:
            slug = slugify_extension(name)
        url = f"{base}/{slug}"
        items.append({"url": url, "name": name})

    return {
        "state": state,
        "category": category,
        "items": items,
    }


def dump_batches_yaml(batches: List[dict]) -> str:
    lines: List[str] = ["batches:"]
    for batch in batches:
        lines.append(f"  - state: {batch['state']}")
        lines.append(f"    category: {batch['category']}")
        lines.append("    items:")
        for item in batch["items"]:
            lines.append(f'      - url: "{item["url"]}"')
            lines.append(f'        name: "{item["name"]}"')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    names = read_names(Path(args.names_file).resolve())

    batch = build_batch(
        state=args.state,
        category=args.category,
        names=names,
        base_url=args.base_url,
        url_style=args.url_style,
    )

    output_text = dump_batches_yaml([batch])

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote url_batches YAML ({len(names)} items) to: {out_path}")
        return 0

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
