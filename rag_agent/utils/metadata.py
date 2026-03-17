from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_STATE_ABBR_TO_NAME: Dict[str, str] = {
    "AL": "ALABAMA",
    "AK": "ALASKA",
    "AZ": "ARIZONA",
    "AR": "ARKANSAS",
    "CA": "CALIFORNIA",
    "CO": "COLORADO",
    "CT": "CONNECTICUT",
    "DE": "DELAWARE",
    "FL": "FLORIDA",
    "GA": "GEORGIA",
    "HI": "HAWAII",
    "ID": "IDAHO",
    "IL": "ILLINOIS",
    "IN": "INDIANA",
    "IA": "IOWA",
    "KS": "KANSAS",
    "KY": "KENTUCKY",
    "LA": "LOUISIANA",
    "ME": "MAINE",
    "MD": "MARYLAND",
    "MA": "MASSACHUSETTS",
    "MI": "MICHIGAN",
    "MN": "MINNESOTA",
    "MS": "MISSISSIPPI",
    "MO": "MISSOURI",
    "MT": "MONTANA",
    "NE": "NEBRASKA",
    "NV": "NEVADA",
    "NH": "NEW HAMPSHIRE",
    "NJ": "NEW JERSEY",
    "NM": "NEW MEXICO",
    "NY": "NEW YORK",
    "NC": "NORTH CAROLINA",
    "ND": "NORTH DAKOTA",
    "OH": "OHIO",
    "OK": "OKLAHOMA",
    "OR": "OREGON",
    "PA": "PENNSYLVANIA",
    "RI": "RHODE ISLAND",
    "SC": "SOUTH CAROLINA",
    "SD": "SOUTH DAKOTA",
    "TN": "TENNESSEE",
    "TX": "TEXAS",
    "UT": "UTAH",
    "VT": "VERMONT",
    "VA": "VIRGINIA",
    "WA": "WASHINGTON",
    "WV": "WEST VIRGINIA",
    "WI": "WISCONSIN",
    "WY": "WYOMING",
    "DC": "DISTRICT OF COLUMBIA",
    "PR": "PUERTO RICO",
}

_STATE_NAME_TO_ABBR: Dict[str, str] = {v: k for k, v in _STATE_ABBR_TO_NAME.items()}
_HARDINESS_LOOKUP_CACHE: Optional[Tuple[Dict[Tuple[str, str], str], Dict[str, str]]] = None
_LAND_GRANT_STATE_TO_DOMAINS: Optional[Dict[str, List[str]]] = None
_HARDINESS_ZONE_TO_DOMAINS: Optional[Dict[str, List[str]]] = None


def _land_grant_csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "Datasets" / "land_grant_universities.csv"


def _hardiness_zone_edu_csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "Datasets" / "hardiness_zone_edu_domain.csv"


def _load_land_grant_state_to_domains() -> Dict[str, List[str]]:
    """Load state (canonical) -> list of edu domains from land grant universities CSV."""
    global _LAND_GRANT_STATE_TO_DOMAINS
    if _LAND_GRANT_STATE_TO_DOMAINS is not None:
        return _LAND_GRANT_STATE_TO_DOMAINS
    state_to_domains: Dict[str, List[str]] = {}
    csv_path = _land_grant_csv_path()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = (row.get("Email Domain") or "").strip().lstrip("@").lower()
                raw_state = (row.get("State") or "").strip()
                if not domain or not raw_state:
                    continue
                state_name = _canonical_state_name(raw_state)
                if state_name:
                    if state_name not in state_to_domains:
                        state_to_domains[state_name] = []
                    state_to_domains[state_name].append(domain)
    _LAND_GRANT_STATE_TO_DOMAINS = state_to_domains
    return _LAND_GRANT_STATE_TO_DOMAINS


def _load_hardiness_zone_to_domains() -> Dict[str, List[str]]:
    """Load phz (hardiness zone) -> list of edu domains from hardiness_zone_edu_domain.csv.

    CSV columns: state, state_abbrev, county_name, university, city, main_domain,
    extension_domains, phz
    """
    global _HARDINESS_ZONE_TO_DOMAINS
    if _HARDINESS_ZONE_TO_DOMAINS is not None:
        return _HARDINESS_ZONE_TO_DOMAINS
    zone_to_domains: Dict[str, List[str]] = {}
    csv_path = _hardiness_zone_edu_csv_path()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                zone = (row.get("phz") or "").strip()
                domain_raw = (row.get("main_domain") or "").strip().lstrip("@").lower()
                if not zone or not domain_raw:
                    continue
                if zone not in zone_to_domains:
                    zone_to_domains[zone] = []
                zone_to_domains[zone].append(domain_raw)
    _HARDINESS_ZONE_TO_DOMAINS = zone_to_domains
    return _HARDINESS_ZONE_TO_DOMAINS


def _extract_state_from_location(location: str) -> str:
    """Extract canonical state name from location string (State, County or State or County, State)."""
    location_text = (location or "").strip()
    if not location_text:
        return ""
    parts = [p.strip() for p in location_text.split(",") if p.strip()]
    if not parts:
        return ""
    if len(parts) >= 2:
        state_name = _canonical_state_name(parts[0])
        if state_name:
            return state_name
        return _canonical_state_name(parts[1])
    return _canonical_state_name(parts[0])


def get_edu_domains_for_state(location: str) -> List[str]:
    """Return edu domains for the state in the given location string."""
    state_name = _extract_state_from_location(location)
    if not state_name:
        return []
    state_to_domains = _load_land_grant_state_to_domains()
    return state_to_domains.get(state_name, [])


def get_edu_domains_for_hardiness_zone(zone: str) -> List[str]:
    """Return edu domains for the given hardiness zone."""
    zone = (zone or "").strip()
    if not zone:
        return []
    zone_to_domains = _load_hardiness_zone_to_domains()
    return zone_to_domains.get(zone, [])


def get_filtered_edu_domains_for_search(location: Optional[str]) -> List[str]:
    """
    Return edu domains filtered by location (state) and hardiness zone.
    If location is None/empty, returns [] (caller uses broad site:.edu).
    If union of state + zone domains has more than 6: use intersection (stricter).
    Otherwise: use union (broader). All results are deduplicated.
    """
    location_text = (location or "").strip()
    if not location_text:
        return []
    state_domains = get_edu_domains_for_state(location_text)
    zone = extract_hardiness_zone_for_location(location_text)
    zone_domains = get_edu_domains_for_hardiness_zone(zone) if zone else []
    union = list(dict.fromkeys(state_domains + zone_domains))
    if len(union) > 6:
        zone_set = set(zone_domains)
        result = [d for d in state_domains if d in zone_set]
        result = list(dict.fromkeys(result)) if result else list(dict.fromkeys(state_domains))
        return result
    return union


def _normalize_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9\s]", " ", (value or "").lower())
    token = re.sub(r"\s+", " ", token).strip()
    return token


def _normalize_county(county: str) -> str:
    county_norm = _normalize_token(county)
    county_norm = re.sub(r"\bcounty\b$", "", county_norm).strip()
    return county_norm


def _normalize_state_name(state: str) -> str:
    return _normalize_token(state).upper()


def _canonical_state_name(state_or_abbr: str) -> str:
    token = _normalize_state_name(state_or_abbr)
    if not token:
        return ""
    if len(token) == 2 and token in _STATE_ABBR_TO_NAME:
        return _STATE_ABBR_TO_NAME[token]
    if token in _STATE_NAME_TO_ABBR:
        return token
    return ""


def _dataset_csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "Datasets" / "county_state_hardiness_zone.csv"


def _load_hardiness_lookups() -> Tuple[Dict[Tuple[str, str], str], Dict[str, str]]:
    global _HARDINESS_LOOKUP_CACHE
    if _HARDINESS_LOOKUP_CACHE is not None:
        return _HARDINESS_LOOKUP_CACHE

    county_state_to_zone: Dict[Tuple[str, str], str] = {}
    per_state_zone_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    csv_path = _dataset_csv_path()
    if not csv_path.exists():
        _HARDINESS_LOOKUP_CACHE = (county_state_to_zone, {})
        return _HARDINESS_LOOKUP_CACHE

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_county = (row.get("county") or "").strip()
            raw_state = (row.get("state") or "").strip()
            raw_state_abbr = (row.get("state_abbr") or "").strip()
            hardiness_zone = (row.get("hardiness_zone") or "").strip()

            state_name = _canonical_state_name(raw_state_abbr) or _canonical_state_name(raw_state)
            county_name = _normalize_county(raw_county)
            if not state_name or not hardiness_zone:
                continue

            if county_name:
                county_state_to_zone[(county_name, state_name)] = hardiness_zone
            per_state_zone_counts[state_name][hardiness_zone] += 1

    state_to_modal_zone = {
        state_name: zone_counts.most_common(1)[0][0]
        for state_name, zone_counts in per_state_zone_counts.items()
        if zone_counts
    }
    _HARDINESS_LOOKUP_CACHE = (county_state_to_zone, state_to_modal_zone)
    return _HARDINESS_LOOKUP_CACHE


def extract_hardiness_zone_for_location(location: str) -> str:
    """
    Resolve hardiness zone from a single location string.

    Expected formats include:
      - "State, County"
      - "State" / "ST" (falls back to state's modal hardiness zone)
      - Legacy compatibility: "County, State" / "County, ST"
    """
    location_text = (location or "").strip()
    if not location_text:
        return ""

    county_state_to_zone, state_to_modal_zone = _load_hardiness_lookups()
    if not county_state_to_zone and not state_to_modal_zone:
        return ""

    parts = [p.strip() for p in location_text.split(",") if p.strip()]
    if not parts:
        return ""

    if len(parts) >= 2:
        # Preferred format: "State, County"
        state_name = _canonical_state_name(parts[0])
        county_name = _normalize_county(parts[1])
        if state_name:
            if county_name:
                zone = county_state_to_zone.get((county_name, state_name))
                if zone:
                    return zone
            return state_to_modal_zone.get(state_name, "")

        # Legacy compatibility: "County, State"
        legacy_county = _normalize_county(parts[0])
        legacy_state = _canonical_state_name(parts[1])
        if legacy_state:
            if legacy_county:
                zone = county_state_to_zone.get((legacy_county, legacy_state))
                if zone:
                    return zone
            return state_to_modal_zone.get(legacy_state, "")
        return ""

    state_name = _canonical_state_name(parts[0])
    if state_name:
        return state_to_modal_zone.get(state_name, "")
    return ""


def build_canonical_chunk_metadata(
    *,
    source_type: str,
    source_id: str,
    title: str,
    url: str,
    page: int,
    chunk_index: int,
    location: str,
    month_year: str,
    content_hash: str,
    language: str,
    hardiness_zone: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the canonical chunk metadata payload used across ingestion paths.

    Canonical keys:
      source_type, source_id, title, url, page, chunk_index,
      location, month_year, content_hash, language
    """
    metadata: Dict[str, Any] = {
        "source_type": source_type,
        "source_id": source_id,
        "title": title,
        "url": url,
        "page": page,
        "chunk_index": chunk_index,
        "location": location,
        "month_year": month_year,
        "content_hash": content_hash,
        "language": language,
        "hardiness_zone": (hardiness_zone or extract_hardiness_zone_for_location(location or "")).strip(),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata
