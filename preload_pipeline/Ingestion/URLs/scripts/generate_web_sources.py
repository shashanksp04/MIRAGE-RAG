from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlsplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate preload manifest web_page_list sources from names or URLs."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL, e.g. https://extension.illinois.edu/plant-problems/",
    )
    parser.add_argument(
        "--names-file",
        default=None,
        help="Text file with one name per line.",
    )
    parser.add_argument(
        "--urls-file",
        default=None,
        help="Text file with one full URL per line.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output YAML file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="Prefix for each generated source name.",
    )
    parser.add_argument(
        "--entity-type",
        default=None,
        help="entity_type value to apply to all records.",
    )
    parser.add_argument(
        "--source-org",
        default=None,
        help="source_org value to apply to all records.",
    )
    parser.add_argument(
        "--location",
        default=None,
        help='Location value for all records. Preferred format: "State, County" or "State".',
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Tag value. Repeat for multiple tags (e.g. --tag plants --tag disease).",
    )
    args = parser.parse_args()
    if bool(args.names_file) == bool(args.urls_file):
        parser.error("Provide exactly one of --names-file or --urls-file.")
    return args


def slugify(name: str) -> str:
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


def read_urls(path: Path) -> List[str]:
    urls: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        cleaned = raw.strip()
        if cleaned:
            urls.append(cleaned)
    if not urls:
        raise ValueError(f"No URLs found in {path}")
    return urls


def derive_name_from_url(url: str, base_url: str) -> str:
    base_prefix = f"{base_url.rstrip('/')}/"
    cleaned_url = url.strip()

    if cleaned_url.startswith(base_prefix):
        suffix = cleaned_url[len(base_prefix) :].strip("/")
        derived = slugify(suffix)
        if derived:
            return derived

    parsed = urlsplit(cleaned_url)
    tail = parsed.path.rstrip("/").split("/")[-1] if parsed.path else ""
    derived_tail = slugify(tail)
    if derived_tail:
        return derived_tail

    fallback = slugify(cleaned_url)
    return fallback or "source"


def build_source_record_from_name(
    *,
    prefix: Optional[str],
    disease_name: str,
    base_url: str,
    entity_type: Optional[str],
    source_org: Optional[str],
    location: Optional[str],
    tags: Optional[List[str]],
) -> dict:
    disease_slug = slugify(disease_name)
    url = f"{base_url.rstrip('/')}/{disease_slug}"
    return build_source_record(
        name_slug=disease_slug,
        url=url,
        entity_type=entity_type,
        source_org=source_org,
        location=location,
        prefix=prefix,
        tags=tags,
    )


def build_source_record(
    *,
    name_slug: str,
    url: str,
    entity_type: Optional[str],
    source_org: Optional[str],
    location: Optional[str],
    prefix: Optional[str],
    tags: Optional[List[str]],
) -> dict:
    final_name = f"{prefix}_{name_slug}" if prefix else name_slug

    record = {
        "name": final_name,
        "type": "web_page_list",
        "urls": [url],
    }
    if entity_type is not None:
        record["entity_type"] = entity_type
    if source_org is not None:
        record["source_org"] = source_org
    if location is not None:
        record["location"] = location
    if tags:
        record["tags"] = tags
    return record


def dump_sources_yaml(sources: List[dict]) -> str:
    lines: List[str] = ["sources:"]
    for src in sources:
        lines.append(f"  - name: {src['name']}")
        lines.append(f"    type: {src['type']}")
        lines.append("    urls:")
        for url in src["urls"]:
            lines.append(f'      - "{url}"')
        if "entity_type" in src:
            lines.append(f"    entity_type: {src['entity_type']}")
        if "source_org" in src:
            lines.append(f"    source_org: {src['source_org']}")
        if "location" in src:
            lines.append(f"    location: {src['location']}")
        if "tags" in src:
            quoted_tags = ", ".join(src["tags"])
            lines.append(f"    tags: [{quoted_tags}]")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    tags = args.tag

    if args.names_file:
        names = read_names(Path(args.names_file).resolve())
        sources = [
            build_source_record_from_name(
                prefix=args.name_prefix,
                disease_name=name,
                base_url=args.base_url,
                entity_type=args.entity_type,
                source_org=args.source_org,
                location=args.location,
                tags=tags,
            )
            for name in names
        ]
    else:
        urls = read_urls(Path(args.urls_file).resolve())
        sources = [
            build_source_record(
                name_slug=derive_name_from_url(url, args.base_url),
                url=url,
                entity_type=args.entity_type,
                source_org=args.source_org,
                location=args.location,
                prefix=args.name_prefix,
                tags=tags,
            )
            for url in urls
        ]

    output_text = dump_sources_yaml(sources)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote {len(sources)} source records to: {out_path}")
        return 0

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
