# How `Dict-Value-Database` JSON Is Created

This document explains how the final dictionary JSON (for example, `output/crop_dictionary_output.json`) is generated in this project.

## Files Involved

- `scripts/generate_web_sources.py`  
  Builds a YAML config file containing URL batches grouped by `state` and `category`.
- `scripts/build_crop_dictionary.py`  
  Reads the YAML + crop-frequency CSV, extracts crop mentions from each URL, and writes the final JSON.
- `YAMLfilesForDict/*.yaml`  
  Input config used by the dictionary builder.
- `output/*.json`  
  Final generated dictionary JSON.

## High-Level Flow

1. Create a YAML batch file from a names list (or write it manually).
2. Run `build_crop_dictionary.py` with:
   - the YAML config,
   - the crop-frequency CSV,
   - output JSON path.
3. The script:
   - normalizes states,
   - loads crop occurrence counts from CSV,
   - fetches and extracts page text from each URL,
   - asks the LLM which crops are discussed in each page,
   - fills category lists per crop,
   - writes the final per-state dictionary JSON.

## Step 1: Generate YAML URL Batches

Use the helper script:

```bash
python preload_pipeline/Dict-Value-Database/scripts/generate_web_sources.py \
  --base-url "https://extension.illinois.edu/plant-problems/" \
  --names-file "preload_pipeline/Ingestion/URLs/names/uiuc.txt" \
  --state "Illinois" \
  --category "disease" \
  --output "preload_pipeline/Dict-Value-Database/YAMLfilesForDict/uiuc.yaml"
```

What this creates:

- A YAML with top-level `batches`.
- Each batch has:
  - `state`
  - `category`
  - `items` (each item has `url` and optional `name`)

Minimal shape:

```yaml
batches:
  - state: Illinois
    category: disease
    items:
      - url: "https://example.com/page"
        name: "Example page"
```

## Step 2: Build the Dictionary JSON

Run:

```bash
python preload_pipeline/Dict-Value-Database/scripts/build_crop_dictionary.py \
  --config preload_pipeline/Dict-Value-Database/YAMLfilesForDict/uiuc.yaml \
  --csv preload_pipeline/Datasets/county_crops_frequency_multi_year_cleaned.csv \
  --output preload_pipeline/Dict-Value-Database/output/crop_dictionary_output.json
```

Optional important flags:

- `--rag-agent-dir` path to sibling `rag_agent` directory.
- `--model` LLM model name used for crop extraction.
- `--openai-api-base` local OpenAI-compatible endpoint (default points to local server).
- `--quiet` to reduce logs.

## What `build_crop_dictionary.py` Does Internally

### 1) Load and validate inputs

- Loads YAML (`batches` list is required).
- Validates each batch has `state`, `category`, and valid URL `items`.
- Loads canonical state resolver from `rag_agent`.

### 2) Build crop occurrence map from CSV

- Reads `state` and `crops` columns from the CSV.
- Parses `crops` values like: `Cotton:18; Soybeans:31; Sweet_Corn:6`.
- Aggregates crop counts per canonical state.

These counts become the `occurrence` value in output JSON.

### 3) Extract page text for each URL

For each URL in each batch:

- Tries `rag_agent` `WebAddition` extraction when available.
- Falls back to `requests` + HTML parsing when needed.
- Skips URL if no text can be extracted.

### 4) Ask LLM which crops are present

- Sends page text + candidate crop list for that state to the LLM.
- Expects a JSON array of crop names.
- Keeps only crops that match known crop names from the CSV.

### 5) Populate category values

- For each matched crop, adds the page `name` (or fallback title/slug) into that crop's category list.
- Category lists are unique string lists.

### 6) Write final JSON

- Output keys are normalized state names (lowercase).
- Each state maps to a list of single-key crop objects.
- Each crop object contains:
  - category arrays (`disease`, `pests`, etc. depending on YAML categories),
  - `occurrence` integer.

## Output JSON Shape

```json
{
  "illinois": [
    {
      "Soybeans": {
        "disease": ["Soybean cyst nematode", "Sclerotinia stem rot"],
        "occurrence": 31
      }
    }
  ]
}
```

## Notes and Troubleshooting

- If YAML has missing `state`, `category`, or bad `items`, the script fails validation.
- If CSV has no rows for a configured state, the script raises an error.
- If text extraction fails for a URL, that URL is skipped.
- Better extraction quality generally improves LLM crop matching quality.
- Keep batch categories consistent (`disease`, `pest`, etc.) because they become JSON keys.

