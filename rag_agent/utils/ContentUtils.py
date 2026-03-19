from typing import Any, List, Dict, Optional, Tuple
import re
import hashlib
from transformers import AutoTokenizer
from rag_agent.utils.metadata import extract_hardiness_zone_for_location


class ContentUtils:
    """
    Utility class for content processing tasks such as
    hashing, chunking, and deduplication checks.
    """

    def __init__(
        self,
        embed_model: str = "BAAI/bge-base-en-v1.5",
        chunk_config: Dict | None = None,
    ):
        self.embed_model = embed_model
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)

        self.chunk_config = chunk_config or {
            "pdf": {
                "max_tokens": 500,
                "overlap": 90,
            },
            "web": {
                "chunk_if_over": 800,
                "max_tokens": 700,
                "overlap": 80,
            },
        }

    # -------------------------
    # Hashing
    # -------------------------

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Computes a normalized SHA-256 hash for text content."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # -------------------------
    # Deduplication
    # -------------------------

    @staticmethod
    def content_hash_exists(collection, content_hash: str) -> bool:
        """Checks whether a content hash already exists in the collection."""
        result = collection.get(where={"content_hash": content_hash})
        return len(result.get("ids", [])) > 0

    # -------------------------
    # Chunking
    # -------------------------

    def chunk_by_tokens(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap: int,
    ) -> List[str]:
        """Splits text into overlapping chunks based on token count."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: List[str] = []

        start = 0
        total_tokens = len(tokens)

        while start < total_tokens:
            end = min(start + max_tokens, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True
            )
            chunks.append(chunk_text)

            if end == total_tokens:
                break

            start = max(end - overlap, 0)

        return chunks
        
    def retrieve_with_priority_filters(
        self,
        *,
        query: str,
        collection,
        location: Optional[str] = None,
        month_year: Optional[str] = None,
        title: Optional[str] = None,
        k: int = 5,
        min_results: int = 1,
    ) -> Tuple[Optional[Dict], str, List[Dict]]:
        """
        Performs semantic retrieval with progressive metadata filtering.

        Returns:
            used_filter: The metadata filter that succeeded (or None)
            strategy: Name of the retrieval strategy used
            results: List of retrieved chunks with text, metadata, and distance
        """

        def _clean(value: Optional[str], *, upper: bool = False) -> Optional[str]:
            """Normalize incoming metadata values and treat NULL-like strings as missing."""
            if value is None:
                return None
            if not isinstance(value, str):
                value = str(value)

            v = value.strip()
            if not v:
                return None

            # Treat these as missing
            if v.upper() in {"NULL", "NONE", "N/A", "NA", "UNKNOWN"}:
                return None

            return v.upper() if upper else v

        def _eq(field: str, val: str) -> Dict:
            """Single equality clause in Chroma where syntax."""
            return {field: {"$eq": val}}

        def _make_where(**kwargs: Optional[str]) -> Optional[Dict]:
            """
            Build a valid Chroma where filter:
            - None if no filters
            - single clause dict if exactly one
            - {"$and": [...]} if multiple
            """
            clauses = []
            for field, val in kwargs.items():
                if val is not None:
                    clauses.append(_eq(field, val))

            if not clauses:
                return None
            if len(clauses) == 1:
                return clauses[0]
            return {"$and": clauses}

        # Clean inputs (and treat "NULL" as None)
        # Note: `location` is only used to derive `hardiness_zone` for metadata filtering.
        location = _clean(location, upper=True)
        month_year = _clean(month_year, upper=False)
        title = _clean(title, upper=False)
        hardiness_zone = _clean(
            extract_hardiness_zone_for_location(location or ""),
            upper=False,
        )

        filter_attempts: List[Tuple[str, Optional[Dict]]] = []

        # Most specific -> least specific -> semantic only
        if hardiness_zone and month_year and title:
            filter_attempts.append((
                "hardiness_zone+month_year+title",
                _make_where(
                    hardiness_zone=hardiness_zone,
                    month_year=month_year,
                    title=title,
                ),
            ))

        if hardiness_zone and title:
            filter_attempts.append((
                "hardiness_zone+title",
                _make_where(hardiness_zone=hardiness_zone, title=title),
            ))

        if title:
            filter_attempts.append(("title", _make_where(title=title)))

        if month_year:
            filter_attempts.append(("month_year", _make_where(month_year=month_year)))

        if hardiness_zone and month_year:
            filter_attempts.append((
                "hardiness_zone+month_year",
                _make_where(hardiness_zone=hardiness_zone, month_year=month_year),
            ))

        if hardiness_zone:
            filter_attempts.append(("hardiness_zone", _make_where(hardiness_zone=hardiness_zone)))

        filter_attempts.append(("semantic_only", None))

        k = int(k)
        min_results = int(min_results)


        def _format_results(
            docs: List[str],
            metadatas: List[Dict[str, Any]],
            distances: List[float],
        ) -> List[Dict[str, Any]]:
            return [
                {"text": doc, "metadata": metadata, "distance": distance}
                for doc, metadata, distance in zip(docs, metadatas, distances)
            ]

        strategy_evaluations: List[Dict[str, Any]] = []

        for strategy_name, where_filter in filter_attempts:
            query_args = {
                "query_texts": [query],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"],
            }

            if where_filter is not None:
                query_args["where"] = where_filter

            results = collection.query(**query_args)

            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            similarity_scores = []
            for distance in distances:
                # Chroma distances are lower-is-better; convert to a higher-is-better score.
                similarity_scores.append(1.0 / (1.0 + max(float(distance), 0.0)))

            doc_count = len(docs)
            normalized_score = (
                sum(similarity_scores) / doc_count
                if doc_count > 0
                else 0.0
            )

            strategy_evaluations.append(
                {
                    "strategy_name": strategy_name,
                    "where_filter": where_filter,
                    "doc_count": doc_count,
                    "docs": docs,
                    "metadatas": metadatas,
                    "distances": distances,
                    "similarity_scores": similarity_scores,
                    "normalized_score": normalized_score,
                }
            )

        valid_strategies = [
            s for s in strategy_evaluations
            if s["doc_count"] >= min_results
        ]

        for s in valid_strategies:
            print(
                f"Strategy passed filter: name={s.get('strategy_name')} "
                f"score={float(s.get('normalized_score', 0.0)):.4f} "
                f"doc_count={int(s.get('doc_count', 0))} (min_results={min_results})"
            )

        if valid_strategies:
            best_strategy = max(valid_strategies, key=lambda s: s["normalized_score"])
            return (
                best_strategy["where_filter"],
                best_strategy["strategy_name"],
                _format_results(
                    best_strategy["docs"],
                    best_strategy["metadatas"],
                    best_strategy["distances"],
                ),
            )

        return None, "no_results", []

