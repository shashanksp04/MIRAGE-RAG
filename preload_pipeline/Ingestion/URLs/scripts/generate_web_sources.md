# `generate_web_sources.py`

This script converts a plain text list of names into manifest-ready `web_page_list` source records for the preload pipeline.

## What it does

- Reads names from a file (`--names-file`), one per line.
- Converts each name into a URL-safe slug.
- Appends each slug to a base URL (`--base-url`).
- Creates one source record per name with shared metadata.
- Outputs YAML to stdout, or writes to a file if `--output` is provided.

## Inputs

- `--base-url` (required): Base path for generated URLs.
- `--names-file` (required): Text file with one entry per line.
- `--output` (optional): Output YAML file path.
- `--name-prefix` (optional): Prefix for generated source `name`. Default: `None`.
- `--entity-type` (optional): Shared `entity_type` value. Default: `None`.
- `--source-org` (optional): Shared `source_org` value. Default: `None`.
- `--location` (optional): Shared `location` value for every generated source. Preferred format: `"State, County"` or `"State"`. Default: `None`.
- `--tag` (optional, repeatable): Shared tags for every source. Default: `None`.

## How slug generation works

For each input name:

1. Normalize Unicode to ASCII and lowercase.
2. Remove `/` characters.
3. Replace non-alphanumeric runs with `-`.
4. Trim leading/trailing `-`.

Example:

- `Bacterial Blight/Canker` -> `bacterial-blightcanker`

## Record format

Each generated item always includes:

```yaml
sources:
  - name: <slug> # or <name-prefix>_<slug> when --name-prefix is provided
    type: web_page_list
    urls:
      - "<base-url>/<slug>"
```

Optional fields are only included if you provide them:

```yaml
sources:
  - name: <slug>
    type: web_page_list
    urls:
      - "<base-url>/<slug>"
    entity_type: <entity-type>   # only if --entity-type is passed
    source_org: <source-org>     # only if --source-org is passed
    location: <location>         # only if --location is passed
    tags: [tag1, tag2, ...]      # only if one or more --tag args are passed
```

## Usage example

```bash
python preload_pipeline/scripts/generate_web_sources.py \
  --base-url "https://extension.illinois.edu/plant-problems/" \
  --names-file "preload_pipeline/uiuc_names.txt" \
  --location "Illinois" \
  --output "preload_pipeline/generated_sources.yaml"
```

Location examples:

- `Illinois`
- `Illinois, Cook`
- `IL`
- `IL, Cook`

## Typical flow

1. Put names into a text file (one per line).
2. Run the script with your base URL.
3. Paste or include the generated YAML in your manifest.
