"""
Crop-dictionary query enrichment: split location prefix, LLM may insert crop names only, recombine.
Used by Inference/generate.py RAG workers only; no ADK/Chroma dependencies.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Set

from openai import OpenAI

# Match get_prompt: "[User location: …]\n\n" then question body
_LOCATION_PREFIX_RE = re.compile(
    r"^(\[User location:[^\]]*\]\s*\n\s*\n)(.*)$",
    re.DOTALL,
)
# First line "[User location: …]" — capture text inside brackets for state parsing
_LOCATION_LINE_RE = re.compile(r"^\[User location:\s*([^\]]*)\]", re.DOTALL)

# Avoid blowing context if the built dictionary grows very large
_MAX_DICT_JSON_CHARS = 200_000
# Cap allowlist text in the prompt (names are still fully extracted for validation sampling)
_MAX_ALLOWLIST_CHARS = 48_000


_SYSTEM_PROMPT = """You help a crop-specific agricultural RAG system by optionally inserting CROP NAMES into the user's question.

The crop dictionary JSON and ALLOWED_CROP_NAMES refer only to the user's state/region when provided (not all states).

Hard rules (violating any rule means you MUST return enriched_body EXACTLY equal to the input body, character-for-character):

1) Use ONLY crop names from the ALLOWED_CROP_NAMES list provided in the user message. Do not use any plant, tree, grass, or crop name that is not in that list. Do not use your general knowledge, common names, or botanical names unless they appear verbatim in ALLOWED_CROP_NAMES.

2) Do NOT paraphrase, summarize, shorten, expand, fix grammar, add adjectives (e.g. "invasive"), or change any wording of the user's text.

3) Do NOT delete, replace, or reorder any character or word from the input body. The only permitted change is INSERTING one or more allowed crop names (exact spelling from the list) at positions where the question clearly refers to a specific crop from the dictionary (e.g. pests/disease/fields) but does not name it.

4) If the question already names a crop, or you are not certain which single allowed crop to insert, or no allowed crop clearly fits, return enriched_body identical to the input body.

5) Do NOT answer a different question, strip instructions, or replace the body with a shorter or unrelated question.

Output: a single JSON object only: {"enriched_body": "<string>"}
enriched_body must be the full question body (no location line, no commentary)."""


def split_location_prefix(full_query: str) -> tuple[str, str]:
    """Split full user message into location prefix (possibly empty) and question body."""
    m = _LOCATION_PREFIX_RE.match(full_query)
    if not m:
        return "", full_query
    return m.group(1), m.group(2)


def _parse_state_from_location_prefix(prefix: str) -> Optional[str]:
    """
    Parse state from [User location: …] (same format as Generate.get_prompt).
    Uses the first segment before a comma when present, e.g. "Illinois, Cook" -> illinois.
    """
    if not (prefix or "").strip():
        return None
    m = _LOCATION_LINE_RE.match(prefix.strip())
    if not m:
        return None
    loc = (m.group(1) or "").strip()
    if not loc:
        return None
    state_part = loc.split(",")[0].strip()
    return state_part.lower() if state_part else None


def _dictionary_slice_for_state(full: Dict[str, Any], state_normalized: str) -> Optional[Dict[str, Any]]:
    """Return a one-key dict {StateKey: [...]} for the matching state, or None if not found."""
    if not state_normalized or not isinstance(full, dict):
        return None
    want = state_normalized.strip().lower()
    for k, v in full.items():
        if isinstance(k, str) and k.strip().lower() == want:
            return {k: v}
    return None


def _state_slice_has_crop_data(sliced: Dict[str, Any]) -> bool:
    """True if the one-state slice has a non-empty list of crop entries."""
    if not sliced:
        return False
    for _k, state_val in sliced.items():
        if not isinstance(state_val, list):
            return False
        return len(state_val) > 0
    return False


def _extract_crop_names_from_dictionary(d: Any) -> Set[str]:
    """Collect crop name keys from the built dictionary (state -> list of {crop_name: {...}})."""
    names: Set[str] = set()
    if not isinstance(d, dict):
        return names
    for _state_key, state_val in d.items():
        if not isinstance(state_val, list):
            continue
        for item in state_val:
            if isinstance(item, dict):
                for crop_name in item.keys():
                    if isinstance(crop_name, str):
                        s = crop_name.strip()
                        if s:
                            names.add(s)
    return names


def _body_is_subsequence_of_enriched(body: str, enriched: str) -> bool:
    """
    True if `body` can be obtained by deleting characters from `enriched` only
    (only insertions were added to get from body to enriched). Rejects replacements,
    deletions from the original, and full rewrites.
    """
    if body == enriched:
        return True
    i = 0
    for c in enriched:
        if i < len(body) and c == body[i]:
            i += 1
    return i == len(body)


def _format_allowlist_for_prompt(names: Set[str]) -> str:
    """Comma-separated sorted names; truncate by char budget (full dict remains in JSON below)."""
    if not names:
        return "(none)"
    sorted_names = sorted(names, key=str.lower)
    text = ", ".join(sorted_names)
    if len(text) <= _MAX_ALLOWLIST_CHARS:
        return text
    return (
        text[: _MAX_ALLOWLIST_CHARS - 80]
        + f"\n… ({len(sorted_names)} total names; use only names from this list or the crop keys in the JSON above.)"
    )


def _parse_enriched_body(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "enriched_body" in obj:
            return str(obj["enriched_body"])
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict) and "enriched_body" in obj:
                return str(obj["enriched_body"])
        except json.JSONDecodeError:
            pass
    return None


class CropQueryEnricher:
    def __init__(
        self,
        api_base: str,
        model: str,
        dictionary: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        api_key: str = "EMPTY",
        timeout_seconds: float = 120.0,
    ):
        self._model = model
        self._dictionary = dictionary
        self._enabled = bool(enabled) and dictionary is not None
        self._api_base = api_base
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        if not self._enabled:
            print(
                "[CropQueryEnricher] Enrichment disabled (no dictionary or feature off); "
                "worker will use original user messages.",
                flush=True,
            )

    def enrich(self, query: str) -> str:
        """Return full user string (prefix + enriched body), or original query on skip/failure."""
        if not query:
            print("[CropQueryEnricher] Empty query; skipping enrichment.", flush=True)
            return query
        if not self._enabled:
            return query

        prefix, body = split_location_prefix(query)
        if not body.strip():
            print("[CropQueryEnricher] No question body after location prefix; using original query.", flush=True)
            return query

        state_norm = _parse_state_from_location_prefix(prefix)
        dict_for_prompt: Dict[str, Any] = self._dictionary  # type: ignore[assignment]
        if state_norm:
            sliced = _dictionary_slice_for_state(self._dictionary, state_norm)
            if sliced is None:
                print(
                    f"[CropQueryEnricher] No crop dictionary entry for state {state_norm!r}; "
                    "skipping enrichment (no LLM call).",
                    flush=True,
                )
                return query
            if not _state_slice_has_crop_data(sliced):
                only_key = next(iter(sliced.keys()))
                print(
                    f"[CropQueryEnricher] No crop data for state {only_key!r} (empty list); "
                    "skipping enrichment (no LLM call).",
                    flush=True,
                )
                return query
            dict_for_prompt = sliced
            only_key = next(iter(sliced.keys()))
            print(
                f"[CropQueryEnricher] Using crop data for state only: {only_key!r} "
                f"(shorter context for enrichment).",
                flush=True,
            )

        dict_json: str
        try:
            raw = json.dumps(dict_for_prompt, ensure_ascii=False)
            if len(raw) > _MAX_DICT_JSON_CHARS:
                raw = raw[:_MAX_DICT_JSON_CHARS] + "\n…[truncated]"
        except (TypeError, ValueError) as e:
            print(f"[CropQueryEnricher] dictionary serialize failed: {e}; using original query.", flush=True)
            return query

        allowlist_names = _extract_crop_names_from_dictionary(dict_for_prompt)
        allowlist_text = _format_allowlist_for_prompt(allowlist_names)

        user_content = (
            f"ALLOWED_CROP_NAMES (you may ONLY insert names from this set; exact spelling):\n{allowlist_text}\n\n"
            "Crop dictionary JSON for this location only (reference; crop keys must match ALLOWED_CROP_NAMES):\n"
            + raw
            + "\n\nUser question body — output enriched_body with at most insertions of allowed crop names, "
            "or the same text unchanged:\n"
            + body
        )

        try:
            # New OpenAI client per request: no shared client state; each call is an isolated chat completion.
            client = OpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=self._timeout_seconds,
            )
            completion = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_completion_tokens=1024,
            )
            message = completion.choices[0].message
            raw_text = (message.content or "").strip()
        except Exception as e:
            print(f"[CropQueryEnricher] LLM call failed: {e}; using original query.", flush=True)
            return query

        enriched = _parse_enriched_body(raw_text)
        if enriched is None:
            print("[CropQueryEnricher] Could not parse JSON enriched_body; using original query.", flush=True)
            return query

        enriched_stripped = enriched.strip()
        if not enriched_stripped:
            print("[CropQueryEnricher] Model returned empty enriched_body; using original query.", flush=True)
            return query

        if enriched_stripped != body and not _body_is_subsequence_of_enriched(body, enriched_stripped):
            print(
                "[CropQueryEnricher] Rejected model output: only insertions of text are allowed "
                "(no deletions, rewrites, or substitutions). Using original question body.",
                flush=True,
            )
            return query

        out = prefix + enriched_stripped
        if enriched_stripped == body:
            print("[CropQueryEnricher] Enrichment finished: question body unchanged (no crop injection).", flush=True)
        else:
            print(
                "[CropQueryEnricher] Enrichment finished: inserted allowed crop name(s) "
                f"({len(body)} -> {len(enriched_stripped)} chars).",
                flush=True,
            )
            print(f"[CropQueryEnricher]   Before: {body[:200]}{'…' if len(body) > 200 else ''}", flush=True)
            print(f"[CropQueryEnricher]   After:  {enriched_stripped[:200]}{'…' if len(enriched_stripped) > 200 else ''}", flush=True)
        return out
