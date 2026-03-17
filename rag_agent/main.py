import chromadb
import time
import re
from .tools.pdf_addition import PDFAddition
from .tools.web_search import WebSearch
from .tools.web_addition import WebAddition
from .tools.confidence_evaluator import ConfidenceEvaluator
from .tools.keyword_extractor import KeywordExtractor
from .utils.ContentUtils import ContentUtils
from .utils.Embedding import SentenceTransformerEmbeddingFunction
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.models.lite_llm import LiteLlm
from typing import Optional, Dict, List, Any


class MainAgent:
    def __init__(self, test_model: str = "Qwen2.5-VL-3B-Instruct", embed_model_name: str = "BAAI/bge-base-en-v1.5", device: str = "None", api_base: str = "http://127.0.0.1:11434/v1"):
        self.test_model = test_model
        self.api_base = api_base
        self.embedding_function = SentenceTransformerEmbeddingFunction(embed_model_name, device)
        self.client = chromadb.PersistentClient(path="./chroma_database/chroma_db") # path has to be a valid path to a directory
        self.collection = self.client.get_or_create_collection(name="meta-mirage_collection", embedding_function=self.embedding_function)
        self.null_str = "__null__"
        self.null_int = -1
        self.content_utils = ContentUtils(embed_model=embed_model_name)
        self.pdf_addition = PDFAddition(self.collection, self.content_utils, self.null_str)
        self.web_search = WebSearch()
        self.web_addition = WebAddition(self.collection, self.content_utils, self.null_str, self.null_int)
        self.confidence_evaluator = ConfidenceEvaluator(self.collection, self.content_utils)
        self.keyword_extractor = KeywordExtractor(model_name=test_model, openai_api_base=api_base)
        
        # Store tools for debugging
        self.tools_list = [
            self._tracked_retrieve_content,
            self._tracked_evaluate_confidence,
            self._tracked_web_search,
            self._tracked_add_web_content,
            self._tracked_add_pdf_content,
            self._tracked_extract_keywords,
        ]
    
    def _tracked_retrieve_content(
        self,
        *,
        query: str,
        location: str | None = None,
        month_year: str | None = None,
        title: str | None = None,
    ) -> Dict:
        """Retrieves relevant content from the vector database using progressive metadata filtering.

        Use this tool FIRST for every query to retrieve relevant information from the knowledge base.
        """
        print(f"[RAG Tools] retrieve_content: CALLED", flush=True)

        try:
            result = self.retrieve_content(
                query=query,
                location=location,
                month_year=month_year,
                title=title,
            )
        except Exception as e:
            # Self-heal: stale collection handle after another worker reset/deleted the collection
            msg = str(e).lower()
            if "does not exist" in msg or "not exist" in msg:
                print(
                    f"[RAG Tools] retrieve_content: Detected stale collection handle ({e}). "
                    f"Re-binding collection + dependent tools and retrying once...",
                    flush=True,
                )
                try:
                    self.collection = self.client.get_or_create_collection(
                        name="meta-mirage_collection",
                        embedding_function=self.embedding_function,
                    )
                    # Rebind components that depend on collection
                    self.pdf_addition = PDFAddition(self.collection, self.content_utils, self.null_str)
                    self.web_addition = WebAddition(self.collection, self.content_utils, self.null_str, self.null_int)
                    self.confidence_evaluator = ConfidenceEvaluator(self.collection, self.content_utils)

                    # Retry once after rebind
                    result = self.retrieve_content(
                        query=query,
                        location=location,
                        month_year=month_year,
                        title=title,
                    )
                except Exception as e2:
                    print(f"[RAG Tools] ✗ retrieve_content: FAILED after rebind - {e2}", flush=True)
                    return {
                        "status": "error",
                        "error_message": f"Collection stale handle; retry after rebind failed: {e2}",
                        "results": [],
                    }
            else:
                print(f"[RAG Tools] ✗ retrieve_content: EXCEPTION - {e}", flush=True)
                return {
                    "status": "error",
                    "error_message": str(e),
                    "results": [],
                }

        status = result.get("status", "unknown")
        if status == "success":
            results_count = len(result.get("results", []))
            print(f"[RAG Tools] ✓ retrieve_content: SUCCESS ({results_count} results)", flush=True)
        else:
            print(
                f"[RAG Tools] ✗ retrieve_content: FAILED - {result.get('error_message', 'Unknown error')}",
                flush=True,
            )
        return result
    
    def reset_collection(self) -> None:
        """Drop and recreate the collection (clean slate)."""
        name = "meta-mirage_collection"

        # Delete if exists (tolerate missing / concurrent deletion)
        try:
            self.client.delete_collection(name=name)
            print(f"[RAG reset_collection] Deleted collection: {name}")
        except Exception as e:
            msg = str(e).lower()
            if "does not exist" in msg or "not exist" in msg:
                print(f"[RAG reset_collection] Collection missing (ok): {e}")
            else:
                # If another process deleted it between checks, some clients throw differently.
                # You can choose to re-raise, but for testing it's better to proceed.
                print(f"[RAG reset_collection] Delete failed (continuing for test): {e}")

        # Always recreate (get_or_create is idempotent)
        try:
            self.collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
            print(f"[RAG reset_collection] Created/loaded collection: {name}")
        except Exception as e:
            # Rare case: if DB is mid-delete in another process, a short retry helps.
            print(f"[RAG reset_collection] get_or_create failed, retrying once: {e}")
            time.sleep(0.5)
            self.collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function
            )

        self.pdf_addition = PDFAddition(self.collection, self.content_utils, self.null_str)
        self.web_addition = WebAddition(self.collection, self.content_utils, self.null_str, self.null_int)
        self.confidence_evaluator = ConfidenceEvaluator(self.collection, self.content_utils)

    def _tracked_evaluate_confidence(self, *, query: str, location: Optional[str] = None,
                                      month_year: Optional[str] = None, title: Optional[str] = None, k: int = 5) -> Dict:
        """Evaluates confidence of retrieved evidence for a query.
        
        Use this tool AFTER calling retrieve_content to determine if the retrieved information is reliable.
        This is MANDATORY - you MUST call this after every retrieval attempt.
        
        Args:
            query: User query
            location: Optional geographic filter
            month_year: Optional temporal filter
            title: Optional document title filter
            k: Number of chunks to retrieve (default: 5)
            
        Returns:
            Dict with status, confidence_level ("high"/"medium"/"low"), confidence_score, and diagnostics
        """
        import sys
        print(f"[RAG Tools] evaluate_retrieval_confidence: CALLED", flush=True)
        result = self.confidence_evaluator.evaluate_retrieval_confidence(
            query=query, location=location, month_year=month_year, title=title, k=k
        )
        status = result.get("status", "unknown")
        if status == "success":
            confidence_level = result.get("confidence_level", "unknown")
            print(f"[RAG Tools] ✓ evaluate_retrieval_confidence: SUCCESS (confidence: {confidence_level})", flush=True)
        else:
            print(f"[RAG Tools] ✗ evaluate_retrieval_confidence: FAILED - {result.get('error_message', 'Unknown error')}", flush=True)
        return result
    
    def _tracked_web_search(self, *, query: str, results_to_extract_count: int = 10) -> Dict:
        """Searches the web for relevant information and extracts clean text.
        
        Use this tool ONLY when confidence_level is "low" after evaluating retrieval confidence.
        This tool searches the web for up-to-date information not available in the knowledge base.
        
        Args:
            query: The search query (use extract_keywords first to optimize the query)
            results_to_extract_count: Number of web results to retrieve and process (default: 10)
            
        Returns:
            Dict with status, query, results (list of dicts with title, url), and error_message if failed
        """
        import sys
        print(f"[RAG Tools] web_search: CALLED (query: {query[:50]}...)", flush=True)
        result = self.web_search.web_search(query, results_to_extract_count)
        status = result.get("status", "unknown")
        if status == "success":
            results_count = len(result.get("results", []))
            print(f"[RAG Tools] ✓ web_search: SUCCESS ({results_count} results)", flush=True)
        else:
            print(f"[RAG Tools] ✗ web_search: FAILED - {result.get('error_message', 'Unknown error')}", flush=True)
        return result
    
    def _tracked_add_web_content(self, *, url: str, location: Optional[str] = None,
                                 month_year: Optional[str] = None, language: str = "en") -> Dict:
        """Wrapper around add_web_content that prints success/failure"""
        if not month_year or not re.match(r"^\d{4}-\d{2}$", month_year.strip()):
            return {
                "status": "error",
                "error_message": "month_year is required for web ingestion and must be in YYYY-MM format.",
            }
        before = self.collection.count()
        result = self.web_addition.add_web_content(url=url, location=location, 
                month_year=month_year, language=language)
        after = self.collection.count()
        print(f"[RAG Tools] add_web_content: count delta={after-before} (before={before}, after={after})", flush=True)
        status = result.get("status", "unknown")
        if status == "success":
            print(f"[RAG Tools] ✓ add_web_content: SUCCESS")
        else:
            print(f"[RAG Tools] ✗ add_web_content: FAILED - {result.get('error_message', 'Unknown error')}")
        return result
    
    def _tracked_add_pdf_content(
        self,
        *,
        pdf_path: str,
        source_id: str,
        title: str,
        location: Optional[str] = None,
        month_year: Optional[str] = None,
        language: str = "en",
    ) -> Dict:
        """Wrapper around add_pdf_content that prints success/failure"""
        result = self.pdf_addition.add_pdf_content(
            pdf_path=pdf_path,
            source_id=source_id,
            title=title,
            location=location,
            month_year=month_year,
            language=language,
        )
        status = result.get("status", "unknown")
        if status == "success":
            print(f"[RAG Tools] ✓ add_pdf_content: SUCCESS")
        else:
            print(f"[RAG Tools] ✗ add_pdf_content: FAILED - {result.get('error_message', 'Unknown error')}")
        return result
    
    def _tracked_extract_keywords(self, *, query: str) -> Dict:
        """Extracts search-optimized keywords from a query.
        
        Use this tool ONLY when confidence_level is "low" to prepare a query for web_search.
        This tool extracts the most important keywords to improve web search results.
        
        Args:
            query: The original user query
            
        Returns:
            Dict with status, keywords (list of strings), and error_message if failed
        """
        result = self.keyword_extractor.extract_keywords(query=query)
        status = result.get("status", "unknown")
        if status == "success":
            keywords_count = len(result.get("keywords", []))
            print(f"[RAG Tools] ✓ extract_keywords: SUCCESS ({keywords_count} keywords)")
        else:
            print(f"[RAG Tools] ✗ extract_keywords: FAILED - {result.get('raw_text_preview', 'Unknown error')}")
        return result

    def retrieve_content(self,
            *,
            query: str,
            location: str | None = None,
            month_year: str | None = None,
            title: str | None = None,
        ) -> dict:
        """Retrieves relevant content using progressive metadata filtering."""

        if not query or not query.strip():
            return {
                "status": "error",
                "error_message": "Empty query provided",
                "results": [],
            }

        print(f"[RAG Tools] collection.count()={self.collection.count()}", flush=True)
        used_filter, strategy, results = self.content_utils.retrieve_with_priority_filters(
            query=query,
            collection=self.collection,
            location=location,
            month_year=month_year,
            title=title,
        )

        if not results:
            return {
                "status": "error",
                "error_message": "No results found",
                "results": [],
            }

        return {
            "status": "success",
            "used_filter": used_filter,
            "strategy": strategy,
            "results": results,
        }

    def main(self):
        import os
        # Set API base URL for OpenAI-compatible endpoints (vLLM)
        # google-adk uses OPENAI_API_BASE environment variable
        os.environ["OPENAI_API_BASE"] = self.api_base
        os.environ["OPENAI_API_KEY"] = "EMPTY"  # vLLM ignores this
        
        # Format model name for google-adk: "openai/model_name" for OpenAI-compatible APIs
        model_name = f"openai/{self.test_model}"
        
        # Debug: Print tool information
        print(f"[RAG Agent Init] Creating agent with model: {model_name}")
        print(f"[RAG Agent Init] Tools to register: {len(self.tools_list)} tools")
        for i, tool in enumerate(self.tools_list):
            tool_name = getattr(tool, '__name__', 'unknown')
            print(f"[RAG Agent Init]   Tool {i+1}: {tool_name}")
        
        SGLANG_BASE_URL = self.api_base
        SGLANG_MODEL = self.test_model
        API_KEY = "EMPTY"

        model_litellm = LiteLlm(
            model=f"openai/{SGLANG_MODEL}",
            api_base=SGLANG_BASE_URL,
            api_key=API_KEY,
            additional_kwargs={
                "tool_choice": "auto",
            },
        )
        
        rag_agent = LlmAgent(
            name="Rag_Agent",
            model=model_litellm,
            description="An agent that retrieves, evaluates, and ingests knowledge.",
            instruction=
            """You are a retrieval-augmented evidence runner. Your job is NOT to answer the user's question.
            Your only job is to run the retrieval pipeline and return the exact retrieved text passages (verbatim)
            that are relevant to the user query, so they can be appended to the user query and sent to another model.

            CRITICAL: You MUST use function calling to invoke tools. Do NOT write text responses that look like tool outputs.
            You MUST actually call the tools using the function calling mechanism provided by the system.

            You have access to tools for:
            - Extracting search-optimized keywords from a user query (_tracked_extract_keywords)
            - Retrieving information from a vector database (_tracked_retrieve_content)
            - Evaluating confidence in retrieved evidence (_tracked_evaluate_confidence)
            - Searching the web (_tracked_web_search)
            - Ingesting new web content into the database (_tracked_add_web_content)
            
            ====================
            CORE RULES (MANDATORY)
            ====================

            1. NEVER answer the user's question directly.
            2. YOU MUST USE TOOLS. Do NOT generate text responses without calling tools first.
            3. ALWAYS call _tracked_retrieve_content FIRST (vector database). DO NOT skip this step. DO NOT guess or make up results.
            4. AFTER calling _tracked_retrieve_content, you MUST call _tracked_evaluate_confidence. DO NOT guess confidence levels. DO NOT write "CONFIDENCE: low" without actually calling the tool.
            5. You MUST follow the confidence-based decision rules below.
            6. Output MUST contain only retrieved text (verbatim) or nothing. No paraphrases, no summaries, no extra facts.
            7. If evidence is insufficient or not found, explicitly admit it and return no evidence.
            8. CRITICAL: You MUST call tools using function calling. Do NOT write text responses that look like tool outputs without actually calling the tools.

            ===========================
            CONFIDENCE-BASED DECISIONS
            ===========================

            CRITICAL: You MUST call _tracked_evaluate_confidence FIRST before making any decisions.
            Do NOT write "CONFIDENCE: low" without actually calling the _tracked_evaluate_confidence tool.
            You MUST use function calling to invoke tools - do NOT just write text that looks like tool outputs.

            After calling _tracked_evaluate_confidence, follow these rules:

            - If confidence_level is "high":
            - Do NOT perform web search.
            - Return the retrieved passages exactly as-is (verbatim), with minimal structure (see Output Format).
            - Do NOT add analysis, explanation, or answers.

            - If confidence_level is "medium":
            - Do NOT answer the question.
            - Return the retrieved passages exactly as-is (verbatim).
            - Include a brief note: "Confidence: medium" (and nothing else besides the evidence).

            - If confidence_level is "low":
            - Do NOT return evidence yet
            - You MUST call _tracked_extract_keywords ONCE to prepare for web search.
            - Join extracted keywords into a single query string.
            - You MUST call _tracked_web_search with the extracted keywords.
            - From the returned object, use both `url` and `month_year` fields inside each item of `results`.
            - You MUST call _tracked_add_web_content for URLs from those web_search results ONLY, and you MUST pass `month_year` with each call.
            - You MUST ingest at least 5 successful URLs (status="success"), up to 10 total attempts.
            - If _tracked_add_web_content fails for a URL, try the next URL from `results` until you reach 5 successes or you run out of results.
            - You MUST call _tracked_retrieve_content again from the vector database.
            - You MUST call _tracked_evaluate_confidence again.

            - If confidence remains "low" after ingestion:
            - Do NOT guess.
            - Respond exactly with:
                "No sufficient reliable information available to return."
            - Return no evidence (empty).

            ===================
            TOOL USAGE RULES
            ===================

            - Use tools only when needed.
            - Do not call the same tool repeatedly with the same arguments.
            - Do not perform web search unless confidence is low.
            - Do not ingest content unless it comes from web_search results.
            - Do not call one tool from inside another tool.
            - Do not fabricate sources, passages, titles, URLs, or citations.

            ===================
            KEYWORD EXTRACTION
            ===================

            Use _tracked_extract_keywords ONLY when:
            - confidence_level is "low", AND
            - you are preparing a query for web_search.

            Rules:
            - Do NOT use _tracked_extract_keywords if confidence is "high" or "medium".
            - Call _tracked_extract_keywords at most once per user query.
            - If keyword extraction fails, fall back to the original user query for web_search.

            ================
            OUTPUT FORMAT
            ================

            Your output must be structured and STRICT.

            If confidence is high or medium and you have relevant evidence:

            Return:

            CONFIDENCE: <high|medium>
            EVIDENCE:
            <verbatim retrieved text passage 1>
            ...

            Rules:
            - Only include passages that were actually retrieved.
            - Do not edit, paraphrase, or “clean up” the text.
            - Preserve original punctuation, casing, line breaks, and any citations included in the retrieved text.
            - Do not add your own citations or commentary.
            - Do not include anything outside the template.

            If no relevant information is found OR confidence remains low after web ingestion:

            Return exactly:

            "No sufficient reliable information available to return."

            And DO NOT include an EVIDENCE section (i.e., return nothing else).

            ===================
            FINAL REMINDER
            ===================

            Accuracy is more important than completeness.
            It is always acceptable to return no evidence.
            It is never acceptable to hallucinate.
            """,
            tools=self.tools_list,
        )
        
        # Debug: Verify tools were registered
        try:
            # Check if tools attribute exists (tools list passed to LlmAgent)
            if hasattr(rag_agent, 'tools'):
                tools_attr = rag_agent.tools
                if tools_attr is not None:
                    tool_count = len(tools_attr) if isinstance(tools_attr, (list, tuple)) else 0
                    print(f"[RAG Agent Init] Agent has {tool_count} tool(s) in tools attribute")
                    if tool_count == 0:
                        print(f"[RAG Agent Init] WARNING: Tools list is empty! Expected {len(self.tools_list)} tools")
                        print(f"[RAG Agent Init] Tools list passed: {[getattr(t, '__name__', 'unknown') for t in self.tools_list]}")
                else:
                    print(f"[RAG Agent Init] Agent tools attribute is None")
            else:
                print(f"[RAG Agent Init] WARNING: Agent does not have 'tools' attribute")
            
            # List all attributes of the agent to debug
            print(f"[RAG Agent Init] Agent attributes: {[attr for attr in dir(rag_agent) if not attr.startswith('_')][:20]}")
        except Exception as e:
            print(f"[RAG Agent Init] Could not verify tool registration: {e}")
            import traceback
            print(f"[RAG Agent Init] Traceback: {traceback.format_exc()}")

        runner = InMemoryRunner(agent=rag_agent)

        return runner

if __name__ == "__main__":
    main_agent = MainAgent()
    runner = main_agent.main()
    response = runner.run_debug(
        "What is Agent Development Kit from Google? What languages is the SDK available in?"
    )
    print(response)