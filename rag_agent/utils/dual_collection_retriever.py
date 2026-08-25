from __future__ import annotations


class DualCollectionRetriever:
    """One retrieval interface for optional curated base plus active runtime."""

    def __init__(self, base_store, runtime_store, content_utils):
        self.base_store = base_store
        self.runtime_store = runtime_store
        self.content_utils = content_utils

    @staticmethod
    def _dedupe(results):
        selected = {}
        for result in results:
            metadata = result.get("metadata", {})
            key = metadata.get("content_hash") or metadata.get("chunk_id") or result.get("text")
            previous = selected.get(key)
            if previous is None or (result.get("retrieval_source") == "base" and previous.get("retrieval_source") != "base"):
                selected[key] = result
        return sorted(selected.values(), key=lambda r: r.get("distance", 1.0))

    def retrieve_with_priority_filters(self, *, query, location=None, month_year=None,
                                       title=None, k=5, min_results=1,
                                       use_progressive_filtering=True):
        stores = [("runtime", self.runtime_store)]
        if self.base_store is not None:
            stores.insert(0, ("base", self.base_store))
        evaluations = []
        for source, store in stores:
            used_filter, strategy, results = self.content_utils.retrieve_with_priority_filters(
                query=query, store=store, location=location, month_year=month_year,
                title=title, k=k, min_results=min_results,
                use_progressive_filtering=use_progressive_filtering,
            )
            for result in results:
                result["retrieval_source"] = source
            evaluations.append((used_filter, strategy, results))
        merged = self._dedupe([r for _, _, results in evaluations for r in results])[:k]
        if not merged:
            return None, "no_results", []
        # Keep the existing strategy naming while confidence sees the merged evidence.
        strategy = max(evaluations, key=lambda e: len(e[2]))[1]
        used_filter = next((e[0] for e in evaluations if e[1] == strategy), None)
        return used_filter, strategy, merged


class CrossCollectionDeduplicator:
    def __init__(self, base_store, runtime_store):
        self.base_store = base_store
        self.runtime_store = runtime_store

    def find_duplicate(self, content_hash):
        if self.base_store is not None and self.base_store.content_hash_exists(content_hash):
            return "base"
        if self.runtime_store.content_hash_exists(content_hash):
            return "runtime"
        return None
