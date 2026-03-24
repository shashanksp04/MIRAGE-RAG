from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import yaml as _yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _yaml = None

# Ensure `preload` and `rag_agent` imports work when running from repo root.
_THIS_FILE = Path(__file__).resolve()
_PRELOAD_DIR = _THIS_FILE.parents[2]  # preload_pipeline/
sys.path.insert(0, str(_PRELOAD_DIR))

from preload.utils.paths import add_project_root_to_syspath  # noqa: E402

# Text truncation for LLM context (chars).
_LLM_TEXT_MAX_CHARS = 8000

DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_OPENAI_API_BASE = "http://127.0.0.1:11434/v1"

_LOG_PREFIX = "[build_crop_disease]"


def _log(msg: str) -> None:
    print(f"{_LOG_PREFIX} {msg}", flush=True)


def _default_rag_agent_dir() -> Path:
    # preload_pipeline/ is sibling to rag_agent/
    return _PRELOAD_DIR.parent / "rag_agent"


def _normalize_state_key(state: str) -> str:
    return (state or "").strip().lower()


def _extract_first_json_blob(text: str) -> Optional[str]:
    """
    Extract the first JSON array or object from possibly messy LLM output.

    Handles SGLang/Llama output with code fences and surrounding prose.
    """
    t = (text or "").strip()
    if not t:
        return None
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE).replace("```", "").strip()
    m = re.search(r"\[[\s\S]*?\]", t)
    if m:
        return m.group(0)
    m = re.search(r"\{[\s\S]*?\}", t)
    if m:
        return m.group(0)
    return None


def _parse_crops_any(raw_text: str) -> List[str]:
    """
    Parse crop names from LLM response.

    Accepts: ["a","b"], {"crops":["a","b"]}, single quotes, trailing commas.
    Returns list of non-empty strings.
    """
    blob = _extract_first_json_blob(raw_text) or (raw_text or "").strip()
    if not blob:
        return []

    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(blob)
        except Exception:
            return []

    if isinstance(obj, dict):
        obj = obj.get("crops") or obj.get("keywords") or obj.get("keyphrases") or obj.get("terms")

    if isinstance(obj, str):
        obj = obj.strip()
        try:
            obj = json.loads(obj)
        except Exception:
            try:
                obj = ast.literal_eval(obj)
            except Exception:
                return []

    if not isinstance(obj, list):
        return []

    out: List[str] = []
    for item in obj:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def _extract_crops_via_llm(
    *,
    text: str,
    title: Optional[str],
    crop_names: List[str],
    llm_client: Any,
    item_url: str = "",
    verbose: bool = True,
) -> Set[str]:
    """
    Call LLM to extract crop names from document text.

    Returns canonical crop names from crop_names that the LLM identifies as present.
    """
    if not crop_names or not (text or "").strip():
        return set()

    truncated = (text or "")[: _LLM_TEXT_MAX_CHARS]
    crop_list_str = ", ".join(sorted(crop_names))

    parts = ["Document title: " + (title or "(unknown)"), "", "Document text:", truncated]
    doc_block = "\n".join(parts)

    prompt = f"""This document describes agricultural diseases, pests, or similar topics.

CANDIDATE CROPS (only use names from this exact list):
{crop_list_str}

TASK: Identify which crops from the list above are explicitly mentioned or clearly discussed in the document. Return ONLY those crop names.

Return a JSON array of crop names. No prose. No markdown code fences unless required for valid JSON.
Example: ["Corn", "Soybeans"]

{doc_block}"""

    if verbose:
        _log(f"Sending to LLM | url={item_url!r} | title={title!r} | text_len={len(text)} chars (truncated to {len(truncated)})")
        _log(f"Prompt preview (first 400 chars): {prompt[:400]}...")
        _log(f"Candidate crops ({len(crop_names)}): {crop_list_str[:200]}{'...' if len(crop_list_str) > 200 else ''}")

    try:
        raw = llm_client.chat(prompt=prompt)
    except Exception as e:
        _log(f"LLM crop extraction failed: {e}")
        return set()

    if not raw or not isinstance(raw, str):
        _log(f"LLM returned empty or non-string: {type(raw)}")
        return set()

    if verbose:
        _log(f"LLM raw response ({len(raw)} chars): {raw[:800]}{'...' if len(raw) > 800 else ''}")

    parsed = _parse_crops_any(raw)
    if not parsed:
        _log(f"Could not parse crop list from LLM response")
        return set()

    if verbose:
        _log(f"Parsed crops from response: {parsed}")

    # Build normalization map: normalized_key -> canonical crop_name
    norm_to_canon: Dict[str, str] = {}
    for c in crop_names:
        key = c.replace("_", " ").strip().lower()
        if key:
            norm_to_canon[key] = c

    result: Set[str] = set()
    for p in parsed:
        key = (p or "").replace("_", " ").strip().lower()
        if key and key in norm_to_canon:
            result.add(norm_to_canon[key])

    if verbose:
        _log(f"Matched canonical crops: {sorted(result)}")

    return result


def _parse_crops_cell(cell: str) -> Iterable[Tuple[str, int]]:
    """
    Parse crops cell like: "Cotton:18; Soybeans:31; Sweet_Corn:6"
    """
    raw = (cell or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    out: List[Tuple[str, int]] = []
    for p in parts:
        if ":" not in p:
            continue
        name, count_str = p.rsplit(":", 1)
        name = name.strip()
        count_str = count_str.strip()
        if not name or not count_str:
            continue
        try:
            count = int(float(count_str))
        except ValueError:
            continue
        out.append((name, count))
    return out


@dataclass(frozen=True)
class UrlItem:
    url: str
    name: Optional[str]


def _coerce_url_items(items_cfg: Any) -> List[UrlItem]:
    if not isinstance(items_cfg, list):
        raise ValueError("Batch 'items' must be a list")
    out: List[UrlItem] = []
    for it in items_cfg:
        if not isinstance(it, dict):
            raise ValueError("Each item must be an object with url and optional name")
        url = (it.get("url") or "").strip()
        if not url:
            raise ValueError("Each item must include non-empty 'url'")
        name = it.get("name")
        if name is not None:
            name = str(name).strip() or None
        out.append(UrlItem(url=url, name=name))
    return out


def _slug_fallback_for_name(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return "unknown"
    last = u.rstrip("/").split("/")[-1]
    last = re.sub(r"[^A-Za-z0-9]+", " ", last).strip()
    return last[:80] if last else "unknown"


def _strip_optional_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1].strip()
    return s


def _load_yaml_subset(config_path: Path) -> Dict[str, Any]:
    """
    Minimal YAML loader for the exact schema used by this repo's example config.

    Supports:
      - top-level "batches:"
      - batches as a list of objects with keys: state, category, items
      - items as a list of objects with keys: url, name (optional)

    This is a fallback when PyYAML isn't installed.
    """
    lines = config_path.read_text(encoding="utf-8").splitlines()

    cfg: Dict[str, Any] = {"batches": []}
    batches = cfg["batches"]

    current_batch: Optional[Dict[str, Any]] = None
    current_item: Optional[Dict[str, Any]] = None
    in_items = False

    def _emit_item() -> None:
        nonlocal current_item
        if current_batch is None or current_item is None:
            return
        current_batch.setdefault("items", [])
        current_batch["items"].append(current_item)
        current_item = None

    def _emit_batch() -> None:
        nonlocal current_batch, current_item, in_items
        if current_batch is None:
            return
        # Ensure items are emitted if file ends while in items.
        _emit_item()
        current_batch.setdefault("items", [])
        batches.append(current_batch)
        current_batch = None
        current_item = None
        in_items = False

    # Regexes tuned for the limited schema.
    re_state = re.compile(r"^\s*-\s*state:\s*(.+?)\s*$")
    re_category = re.compile(r"^\s*category:\s*(.+?)\s*$")
    re_items = re.compile(r"^\s*items:\s*$")
    re_url = re.compile(r"^\s*-\s*url:\s*(.+?)\s*$")
    re_name = re.compile(r"^\s*name:\s*(.+?)\s*$")

    for raw_line in lines:
        # Ignore empty lines.
        if not raw_line.strip():
            continue

        line = raw_line.rstrip()
        # Remove simple comments.
        if "#" in line:
            # Only strip if it looks like a comment delimiter (start of token).
            line = line.split("#", 1)[0].rstrip()
            if not line:
                continue

        if line.strip() == "batches:":
            continue

        m_state = re_state.match(line)
        if m_state:
            _emit_batch()
            current_batch = {"state": _strip_optional_quotes(m_state.group(1))}
            in_items = False
            continue

        if current_batch is None:
            continue

        m_category = re_category.match(line)
        if m_category:
            current_batch["category"] = _strip_optional_quotes(m_category.group(1))
            continue

        if re_items.match(line):
            in_items = True
            continue

        if not in_items:
            continue

        m_url = re_url.match(line)
        if m_url:
            _emit_item()
            current_item = {"url": _strip_optional_quotes(m_url.group(1))}
            continue

        m_name = re_name.match(line)
        if m_name and current_item is not None:
            current_item["name"] = _strip_optional_quotes(m_name.group(1))
            continue

    _emit_batch()
    return cfg


def _load_rag_modules(rag_agent_dir: Path) -> Tuple[Any, Optional[Any]]:
    """
    Load rag_agent canonical-state and optional WebAddition.

    WebAddition may fail to import if its optional deps (e.g., trafilatura) aren't installed.
    """
    # Adds the *parent* of rag_agent to sys.path, matching bootstrap.py.
    add_project_root_to_syspath(rag_agent_dir)
    from rag_agent.utils.metadata import _canonical_state_name  # noqa: WPS433

    web_addition_cls: Optional[Any] = None
    try:
        from rag_agent.tools.web_addition import WebAddition as _WebAddition  # noqa: WPS433

        web_addition_cls = _WebAddition
    except ModuleNotFoundError:
        web_addition_cls = None

    return _canonical_state_name, web_addition_cls


def _extract_title_and_text_via_requests(url: str, *, timeout_s: int = 20) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort HTML fetch + cleaning (fallback when trafilatura-based extraction isn't available).
    """
    try:
        import requests  # type: ignore
    except ModuleNotFoundError:
        return None, None

    headers = {"User-Agent": "Mozilla/5.0 (compatible; MetaMirageBot/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s)
    except Exception:
        return None, None

    if not getattr(resp, "text", None):
        return None, None

    html = resp.text

    # Prefer BeautifulSoup if available.
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = " ".join(soup.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            return title or None, text[:30000]
    except ModuleNotFoundError:
        pass
    except Exception:
        # Keep going to regex fallback.
        pass

    # Regex fallback: strip tags aggressively.
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = None
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() or None

    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return title, None
    return title, text[:30000]


def main() -> int:
    p = argparse.ArgumentParser("Build crop->disease/pest dictionary from URL batches + crop frequency CSV.")
    p.add_argument("--config", required=True, help="YAML config containing batches")
    p.add_argument("--csv", required=True, help="Path to county_crops_frequency_multi_year_cleaned.csv")
    p.add_argument("--output", default="build_crop_disease_dictionary_output.json", help="Output JSON path")
    p.add_argument("--rag-agent-dir", default=str(_default_rag_agent_dir()), help="Path to rag_agent sibling dir")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--openai-api-base",
        default=DEFAULT_OPENAI_API_BASE,
        help="OpenAI-compatible API base URL (e.g. SGLang). Empty for cloud OpenAI.",
    )
    p.add_argument("--verbose", action="store_true", default=True, help="Log processing details (default: True)")
    p.add_argument("--quiet", action="store_true", help="Disable verbose logging")
    args = p.parse_args()

    verbose = args.verbose and not args.quiet

    config_path = Path(args.config).resolve()
    csv_path = Path(args.csv).resolve()
    out_path = Path(args.output).resolve()
    rag_agent_dir = Path(args.rag_agent_dir).resolve()

    _log(f"Config: {config_path}")
    _log(f"CSV: {csv_path}")
    _log(f"Output: {out_path}")
    _log(f"Model: {args.model}")
    _log(f"API base: {args.openai_api_base or '(cloud OpenAI)'}")
    _log(f"Verbose: {verbose}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if _yaml is not None:
        cfg = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        cfg = _load_yaml_subset(config_path)

    batches_cfg = cfg.get("batches") if isinstance(cfg, dict) else None
    if not isinstance(batches_cfg, list) or not batches_cfg:
        raise ValueError("YAML must contain non-empty top-level 'batches' list")

    _log(f"Loaded {len(batches_cfg)} batches from config")

    canonical_state_name_fn, web_addition_cls = _load_rag_modules(rag_agent_dir)

    # LLM client for crop extraction (fresh instance per call; created in loop).
    api_base = (args.openai_api_base or "").strip()
    model_name = (args.model or DEFAULT_MODEL).strip()
    use_local = bool(api_base)

    def _make_llm_client() -> Any:
        if use_local:
            from chat_models.Client import Client as _Client
            return _Client(
                model_name=model_name,
                openai_api_key="EMPTY",
                openai_api_base=api_base,
                messages=[],
            )
        from chat_models.OpenAI_Chat import OpenAI_Chat as _OpenAIChat
        return _OpenAIChat(model_name=model_name, messages=[])

    # Optional extractor (only used if WebAddition import + deps are available).
    web_adder: Optional[Any] = None
    if web_addition_cls is not None:
        class _ContentUtilsStub:
            pass

        try:
            web_adder = web_addition_cls(collection=None, content_utils=_ContentUtilsStub())  # type: ignore[arg-type]
            _log("WebAddition available for HTML extraction")
        except Exception:
            web_adder = None
            _log("WebAddition not available, will use requests fallback for HTML")
    else:
        _log("WebAddition not loaded, will use requests fallback for HTML")

    # Validate batches + gather states/categories.
    wanted_states_canon: Set[str] = set()
    state_output_key_by_canon: Dict[str, str] = {}
    categories_by_state: Dict[str, Set[str]] = defaultdict(set)
    items_by_state_and_category: Dict[Tuple[str, str], List[UrlItem]] = defaultdict(list)
    batches: List[Dict[str, Any]] = []

    for b in batches_cfg:
        if not isinstance(b, dict):
            raise ValueError("Each batch must be a mapping/object")
        raw_state = str(b.get("state") or "").strip()
        if not raw_state:
            raise ValueError("Each batch must include 'state'")
        category = str(b.get("category") or "").strip()
        if not category:
            raise ValueError("Each batch must include 'category'")
        items = _coerce_url_items(b.get("items"))

        canon_state = canonical_state_name_fn(raw_state)
        if not canon_state:
            canon_state = raw_state.strip().upper()

        wanted_states_canon.add(canon_state)
        state_output_key_by_canon.setdefault(canon_state, _normalize_state_key(raw_state))
        categories_by_state[canon_state].add(category)
        items_by_state_and_category[(canon_state, category)].extend(items)
        batches.append({"canon_state": canon_state, "category": category, "items": items})

    _log(f"Batches: states={sorted(wanted_states_canon)}, categories_by_state={dict(categories_by_state)}")

    # Parse CSV once and accumulate occurrence for wanted states.
    state_crop_occurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"state", "crops"}
        if not required_cols.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV must contain columns: {sorted(required_cols)} (found {reader.fieldnames})")

        for row in reader:
            row_state_raw = str(row.get("state") or "").strip()
            if not row_state_raw:
                continue
            row_canon = canonical_state_name_fn(row_state_raw) or row_state_raw.strip().upper()
            if row_canon not in wanted_states_canon:
                continue

            crops_cell = row.get("crops") or ""
            for crop_name, count in _parse_crops_cell(crops_cell):
                state_crop_occurrence[row_canon][crop_name] += count

    for st in sorted(wanted_states_canon):
        n_crops = len(state_crop_occurrence.get(st) or {})
        _log(f"CSV crops for state {st}: {n_crops} crops")

    # Initialize output sets for each wanted state and crop.
    output_sets: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    output_occurrence: Dict[str, Dict[str, int]] = defaultdict(dict)

    for canon_state in wanted_states_canon:
        crop_occ = state_crop_occurrence.get(canon_state) or {}
        if not crop_occ:
            raise ValueError(f"No crop occurrence rows found for state={canon_state} in CSV: {csv_path}")

        for crop_name, occ in crop_occ.items():
            output_occurrence[canon_state][crop_name] = int(occ)
            for cat in categories_by_state.get(canon_state, set()):
                output_sets[canon_state][crop_name][cat] = set()

    # Process URLs.
    for b in batches:
        canon_state = b["canon_state"]
        category = b["category"]
        items: List[UrlItem] = b["items"]

        _log(f"Processing batch: state={canon_state}, category={category}, {len(items)} URLs")

        crop_names = sorted(output_occurrence[canon_state].keys(), key=len, reverse=True)

        for idx, item in enumerate(items, 1):
            _log(f"  [{idx}/{len(items)}] URL: {item.url}")

            title: Optional[str] = None
            text: Optional[str] = None

            if web_adder is not None:
                try:
                    html = web_adder.extract_data(item.url)
                except Exception:
                    html = None

                if html:
                    try:
                        extraction = web_adder.extract_web_page(html=html)
                    except Exception as ex:
                        extraction = {"status": "failed"}
                        if verbose:
                            _log(f"    WebAddition extract_web_page failed: {ex}")

                    if extraction.get("status") == "success":
                        title = (extraction.get("title") or "").strip() or None
                        text = (extraction.get("text") or "").strip() or None
                        if verbose:
                            _log(f"    Extracted via WebAddition: title={title!r}, text_len={len(text) if text else 0}")
                    elif verbose:
                        _log(f"    WebAddition extraction status: {extraction.get('status', 'unknown')}")

            if not text:
                # Fallback extraction path when trafilatura/bs4 aren't available.
                title_fb, text_fb = _extract_title_and_text_via_requests(item.url)
                title = title or title_fb
                text = text_fb
                if verbose and text:
                    _log(f"    Extracted via requests fallback: title={title!r}, text_len={len(text)}")

            if not text:
                _log(f"    SKIP: No text extracted from URL")
                continue

            item_name = item.name or title or _slug_fallback_for_name(item.url)
            if not item_name:
                item_name = _slug_fallback_for_name(item.url)

            llm_client = _make_llm_client()
            matched_crops = _extract_crops_via_llm(
                text=text,
                title=title,
                crop_names=crop_names,
                llm_client=llm_client,
                item_url=item.url,
                verbose=verbose,
            )

            _log(f"    Result: {len(matched_crops)} matched crops -> {sorted(matched_crops) if matched_crops else '[]'}")

            for crop_name in matched_crops:
                output_sets[canon_state][crop_name][category].add(item_name)

    # Convert to required JSON structure.
    result: Dict[str, Any] = {}
    for canon_state in sorted(wanted_states_canon):
        state_key = state_output_key_by_canon.get(canon_state) or _normalize_state_key(canon_state)
        crop_entries: List[Dict[str, Any]] = []

        for crop_name in sorted(output_occurrence[canon_state].keys()):
            cats_dict: Dict[str, Any] = {}
            for cat in sorted(categories_by_state.get(canon_state, set())):
                vals = sorted(output_sets[canon_state][crop_name].get(cat, set()))
                cats_dict[cat] = vals
            cats_dict["occurrence"] = int(output_occurrence[canon_state][crop_name])
            crop_entries.append({crop_name: cats_dict})

        result[state_key] = crop_entries

    total_entries = sum(len(v) for v in result.values())
    _log(f"Output: {len(result)} states, {total_entries} crop entries total")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Wrote output JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

