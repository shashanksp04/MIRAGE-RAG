import csv
import hashlib
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import re
import trafilatura
from bs4 import BeautifulSoup

from rag_agent.utils.metadata import build_canonical_chunk_metadata

_LAND_GRANT_CSV = Path(__file__).resolve().parents[2] / "Datasets" / "land_grant_universities.csv"
_EDU_DOMAIN_TO_STATE: dict[str, str] | None = None


def _load_edu_domain_to_state() -> dict[str, str]:
    """Load domain -> state mapping from land grant universities CSV."""
    global _EDU_DOMAIN_TO_STATE
    if _EDU_DOMAIN_TO_STATE is not None:
        return _EDU_DOMAIN_TO_STATE
    mapping: dict[str, str] = {}
    if _LAND_GRANT_CSV.exists():
        with open(_LAND_GRANT_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = (row.get("Email Domain") or "").strip().lstrip("@")
                state = (row.get("State") or "").strip()
                if domain and state:
                    mapping[domain.lower()] = state
    _EDU_DOMAIN_TO_STATE = mapping
    return mapping


class WebAddition:
    """
    Agent tool for ingesting web page content into a vector database.
    """

    def __init__(
        self,
        collection,
        content_utils,
        null_str: str = "",
        null_int: int = -1,
    ):
        """
        Args:
            collection: Vector database collection
            content_utils: Instance of ContentUtils
            null_str: Placeholder for missing string metadata
            null_int: Placeholder for missing integer metadata
        """
        self.collection = collection
        self.content_utils = content_utils
        self.null_str = null_str
        self.null_int = null_int

    def clean_webpage_text(self, text: str) -> str:

        # Normalize white space
        text = re.sub(r"[ \t]+"," ", text)

        # Normalize newlines
        text = re.sub(r"\n{2,}", "\n\n", text)

        # removes boilerplate
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if len(line) < 20:
                continue

            lines.append(line)

        return "\n\n".join(lines)

    def extract_title(self, html: str) -> str | None:
        soup = BeautifulSoup(html, "lxml")

        if soup.title and soup.title.string:
            return soup.title.string.strip()

        h1 = soup.find("h1")
        if h1 and h1.string:
            return h1.string.strip()

        return None

    def extract_web_page(self, *, html: str) -> dict:
        """
        Extracts clean, semantic text from a web page HTML.

        Use this tool when:
        - The agent has fetched raw HTML from a web request
        - The content may be relevant for ingestion into a vector database

        Returns:
        - status: "success" | "failed"
        - title: extracted page title (if available)
        - text: cleaned main content (if successful)
        - reason: failure reason (if failed)
        """

        if not html or not html.strip():
            return {
                "status": "failed",
                "reason": "empty_html"
            }

        title = self.extract_title(html)

        extracted = trafilatura.extract(
            html,
            include_comments = False,
            include_tables = False,
            include_links = False
            )

        if not extracted:
            return {
                "status": "failed",
                "reason": "no_main_content",
                "title": title or "Untitled"
            }

        cleaned_text = self.clean_webpage_text(extracted)

        if not cleaned_text or len(cleaned_text) < 30:
            return {
                "status": "failed",
                "reason": "content_too_short",
                "title": title or "Untitled"
            }

        return {
            "status": "success",
            "title": title or "Untitled",
            "text": cleaned_text,
            "char_count": len(cleaned_text)
        }

    def extract_data(self, URL: str) -> str:
        downloaded = trafilatura.fetch_url(URL)
        if not downloaded:
            return None

        # clean_text = trafilatura.extract(
        #     downloaded,
        #     include_comments=False,
        #     include_tables=False,
        #     include_links=False,
        #     favor_recall=False
        # )

        return downloaded

    @staticmethod
    def _normalize_month_year(month_year: Optional[str]) -> str:
        raw = (month_year or "").strip()
        if not raw:
            return ""
        return raw[:7] if re.match(r"^\d{4}-\d{2}", raw) else ""

    @staticmethod
    def _extract_domain_from_url(url: str) -> str | None:
        """Extract hostname from URL, e.g. www.berkeley.edu -> berkeley.edu."""
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or parsed.path or "").strip()
            if not host:
                return None
            host = host.lower()
            if host.startswith("www."):
                host = host[4:]
            return host if host else None
        except Exception:
            return None

    @staticmethod
    def _get_state_from_edu_url(url: str) -> str | None:
        """If URL is from a .edu domain in the land grant CSV, return the state."""
        domain = WebAddition._extract_domain_from_url(url)
        if not domain or not domain.endswith(".edu"):
            return None
        mapping = _load_edu_domain_to_state()
        return mapping.get(domain)

    def add_web_content(
        self,
        *,
        url: str,
        location: Optional[str] = None,
        month_year: Optional[str] = None,
        language: str = "en",
    ) -> dict:
        """Adds web page content to the vector database.

        Use when:
        - New web content is discovered
        - Search results should be indexed for retrieval

        The tool:
        - Extracts and cleans main content from HTML
        - Generates a stable source_id
        - Normalizes metadata
        - Prevents duplicates
        - Is safe to call multiple times

        Args:
            url: Canonical page URL
            location: Optional geographic context
            month_year: Optional publication date
            language: Language code

        Returns:
            Success:
            {
              "status": "success",
              "source_type": "web",
              "url": str,
              "source_id": str,
              "chunks_added": int,
              "chunks_skipped_as_duplicates": int
            }

            Error:
            {
              "status": "error",
              "error_message": str
            }
        """
        html_url = self.extract_data(url)

        if not html_url:
            return {"status": "error", "error_message": f"Failed to fetch URL: {url}"}

        extraction = self.extract_web_page(html=html_url)

        if extraction["status"] != "success":
            return {
                "status": "error",
                "error_message": f"Web extraction failed: {extraction.get('reason')}",
            }

        text = extraction["text"]
        title = extraction["title"]

        if not text or not text.strip():
            return {
                "status": "error",
                "error_message": "Extracted web content is empty",
            }

        if location:
            location = location.upper()
        else:
            state_from_url = self._get_state_from_edu_url(url)
            location = state_from_url.upper() if state_from_url else self.null_str
        resolved_month_year = self._normalize_month_year(month_year)
        month_year = resolved_month_year if resolved_month_year else self.null_str

        source_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        token_count = len(
            self.content_utils.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

        if token_count > self.content_utils.chunk_config["web"]["chunk_if_over"]:
            chunks = self.content_utils.chunk_by_tokens(
                text,
                max_tokens=self.content_utils.chunk_config["web"]["max_tokens"],
                overlap=self.content_utils.chunk_config["web"]["overlap"],
            )
        else:
            chunks = [text]

        documents = []
        metadatas = []
        ids = []

        added = 0
        skipped = 0
        seen_hashes = set()

        for chunk_index, chunk in enumerate(chunks):
            content_hash = self.content_utils.compute_content_hash(chunk)

            if content_hash in seen_hashes:
                skipped += 1
                continue

            if self.content_utils.content_hash_exists(
                self.collection,
                content_hash
            ):
                skipped += 1
                continue

            seen_hashes.add(content_hash)
            added += 1

            doc_id = f"{source_id}_c{chunk_index}"

            documents.append(f"Title: {title}\n\n{chunk}")
            metadatas.append(
                build_canonical_chunk_metadata(
                    source_type="web",
                    source_id=source_id,
                    title=title,
                    url=url,
                    page=self.null_int,
                    chunk_index=chunk_index,
                    location=location,
                    month_year=month_year,
                    content_hash=content_hash,
                    language=language,
                )
            )
            ids.append(doc_id)

        if not documents:
            return {
                "status": "error",
                "error_message": "No new content to add after deduplication",
            }

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        return {
            "status": "success",
            "source_type": "web",
            "url": url,
            "source_id": source_id,
            "chunks_added": added,
            "chunks_skipped_as_duplicates": skipped,
        }
