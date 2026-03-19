# Dict-Value-Database `generate_web_sources.py`

Generates **url_batches** YAML for `build_crop_disease_dictionary.py`, in the format expected by `YAMLfilesForDict/url_batches_example.yaml`.

## Output format

```yaml
batches:
  - state: Illinois
    category: disease
    items:
      - url: "https://en.wikipedia.org/wiki/Soybean_cyst_nematode"
        name: "Soybean cyst nematode"
      - url: "https://en.wikipedia.org/wiki/Cotton_bollworm"
        name: "Cotton bollworm"
```

## Usage

```bash
python preload_pipeline/Dict-Value-Database/generate_web_sources.py \
  --base-url "https://en.wikipedia.org/wiki/" \
  --names-file "preload_pipeline/names/uiuc.txt" \
  --state "Illinois" \
  --category "disease" \
  --output "preload_pipeline/Dict-Value-Database/YAMLfilesForDict/my_batches.yaml"
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--base-url` | Yes | Base URL (e.g. `https://en.wikipedia.org/wiki/` or `https://extension.illinois.edu/plant-problems/`) |
| `--names-file` | Yes | Text file with one disease/pest name per line |
| `--state` | Yes | State name for the batch |
| `--category` | Yes | Category (e.g. `disease` or `pests`) |
| `--output` | No | Output YAML path. If omitted, prints to stdout |
| `--url-style` | No | `wikipedia` (spaces→underscores) or `slug` (lowercase hyphenated). Default: `wikipedia` |

## Typical flow

1. Put disease/pest names into a text file (one per line).
2. Run the script with your base URL, state, and category.
3. Use the generated YAML as `--config` for `build_crop_disease_dictionary.py`.
