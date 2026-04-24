import statistics
from typing import Optional, Dict


class ConfidenceEvaluator:
    """
    Agent tool for evaluating confidence in retrieved evidence.
    """

    def __init__(self, collection, content_utils):
        """
        Args:
            collection: Vector database collection
            content_utils: Instance of ContentUtils
        """
        self.collection = collection
        self.content_utils = content_utils

    def evaluate_retrieval_confidence(
        self,
        *,
        query: str,
        location: Optional[str] = None,
        month_year: Optional[str] = None,
        title: Optional[str] = None,
        k: int = 5,
        use_progressive_filtering: bool = True,
    ) -> dict:
        """Evaluates confidence of retrieved evidence for a query.

        Use this tool when:
        - The agent needs to decide whether retrieved data is reliable
        - The agent wants to qualify its answer (high / medium / low confidence)

        Args:
            query: User query
            location: Optional geographic filter
            month_year: Optional temporal filter
            title: Optional document title filter
            k: Number of chunks to retrieve
            use_progressive_filtering: When True, evaluate confidence using progressive
                metadata retrieval strategies. When False, force semantic-only retrieval.

        Returns:
            Success:
            {
              "status": "success",
              "confidence_score": float,
              "confidence_level": "high" | "medium" | "low",
              "diagnostics": {
                    "similarity_score": float,
                    "coverage_score": float,
                    "consistency_score": float,
                    "scope_score": float,
                    "strategy_used": str,
                    "num_chunks": int
                }
            }

            Error:
            {
                "status": "error",
                "error_message": str
            }
        """

        if not query or not query.strip():
            return {
                "status": "error",
                "error_message": "Empty query provided"
            }

        used_filter, strategy, results = self.content_utils.retrieve_with_priority_filters(
            query=query,
            collection=self.collection,
            location=location,
            month_year=month_year,
            title=title,
            k=k,
            use_progressive_filtering=use_progressive_filtering,
        )

        if not results:
            return {
                "status": "success",
                "confidence_score": 0.0,
                "confidence_level": "low",
                "diagnostics": {
                    "reason": "no_relevant_evidence",
                    "strategy_used": strategy,
                    "num_chunks": 0
                }
            }

        similarities = [1 - r["distance"] for r in results]
        similarity_score = sum(similarities) / len(similarities)

        coverage_score = min(len(results) / 5, 1.0)

        if len(similarities) > 1:
            variance = statistics.pvariance(similarities)
            consistency_score = max(0.0, 1 - variance * 5)
        else:
            consistency_score = 0.7

        scope_weights = {
            "hardiness_zone+month_year+title": 1.0,
            "hardiness_zone+month_year": 0.9,
            "hardiness_zone+title": 0.85,
            "hardiness_zone": 0.8,
            "month_year": 0.75,
            "title": 0.7,
            "semantic_only": 0.4,
            "no_results": 0.0,
        }

        scope_score = scope_weights.get(strategy, 0.4)

        confidence_score = round(
            0.50 * similarity_score
            + 0.20 * coverage_score
            + 0.20 * consistency_score
            + 0.10 * scope_score,
            3
        )

        if confidence_score >= 0.75:
            confidence_level = "high"
        elif confidence_score >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            "status": "success",
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "diagnostics": {
                "similarity_score": similarity_score,
                "coverage_score": coverage_score,
                "consistency_score": consistency_score,
                "scope_score": scope_score,
                "strategy_used": strategy,
                "num_chunks": len(results),
            }
        }
