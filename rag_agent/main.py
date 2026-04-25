import chromadb
import time
import re
import json
from pathlib import Path
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
    def __init__(self, test_model: str = "Qwen2.5-VL-3B-Instruct", embed_model_name: str = "BAAI/bge-base-en-v1.5", device: str = "None", api_base: str = "http://127.0.0.1:11434/v1", ablation_id: str = "default"):
        self.test_model = test_model
        self.api_base = api_base
        self.ablation_id = (ablation_id or "").strip() or "default"
        self.embedding_function = SentenceTransformerEmbeddingFunction(embed_model_name, device)
        persist_path = "/work/nvme/bfox/ssingh38/chroma_database/chroma_db"
        self.client = chromadb.PersistentClient(path=persist_path) # path has to be a valid path to a directory, shifted to nvme for storage reasons
        self.collection = self.client.get_or_create_collection(name="meta-mirage_collection", embedding_function=self.embedding_function)
        print(f"[RAG Init] Chroma persist path: {persist_path}", flush=True)
        try:
            collections = self.client.list_collections()
            collection_names = [c.name if hasattr(c, "name") else str(c) for c in collections]
            print(f"[RAG Init] Chroma collections: {collection_names}", flush=True)
        except Exception as e:
            print(f"[RAG Init] Failed to list collections: {e}", flush=True)
        print("[RAG Init] Skipping startup collection.count() debug check.", flush=True)
        self.null_str = "__null__"
        self.null_int = -1
        self.content_utils = ContentUtils(embed_model=embed_model_name)
        self.pdf_addition = PDFAddition(self.collection, self.content_utils, self.null_str)
        self.web_search = WebSearch()
        self.web_addition = WebAddition(self.collection, self.content_utils, self.null_str, self.null_int)
        self.confidence_evaluator = ConfidenceEvaluator(self.collection, self.content_utils)
        self.keyword_extractor = KeywordExtractor(model_name=test_model, openai_api_base=api_base)
        self.current_location: Optional[str] = None
        # Ablation toggle: set False to disable location-aware domain filtering for all web searches.
        self.use_domain_filter: bool = True
        # Ablation toggle: set False to disable progressive metadata filtering (semantic-only retrieval).
        self.use_progressive_filtering: bool = True
        # Ablation toggle: set False to disable web search and ingestion behavior in ablation runs.
        self.use_web_search: bool = True
        # Ablation toggle: set False to disable ingestion loop (add_web_content/add_pdf_content) in ablation runs.
        self.use_ingestion_loop: bool = True
        # Ablation toggle: set False to disable confidence evaluation in retrieval flow.
        self.use_confidence_eval: bool = True
        self.ablation_settings = self._extract_ablation_settings(self.ablation_id)
        self.applied_instruction_key: str = ""
        self._apply_ablation_settings()

        # Store tools for debugging
        self.tools_list = self._build_tools_list()

    def _load_instruction_templates(self) -> Dict[str, str]:
        """Loads instruction templates from model_instructions.md."""
        instruction_path = Path(__file__).resolve().parent / "model_instructions.md"
        try:
            content = instruction_path.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to read instruction template file: {instruction_path} ({e})") from e

        sections = re.findall(
            r"<!--\s*instruction:([a-z0-9_]+)\s*-->\s*\n(.*?)(?=\n<!--\s*instruction:|\Z)",
            content,
            flags=re.DOTALL,
        )
        templates = {name.strip(): body.strip() for name, body in sections}
        required = {"confidence_on", "confidence_off"}
        missing = required - set(templates.keys())
        if missing:
            raise RuntimeError(
                f"Missing instruction template section(s): {sorted(missing)} in {instruction_path}"
            )
        return templates

    def _load_ablation_configs(self) -> Dict[str, Dict[str, Any]]:
        """Loads ablation settings map from ablation_configs.json."""
        config_path = Path(__file__).resolve().parent / "ablation_configs.json"
        try:
            raw = config_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[RAG Ablation] Failed to read ablation config file: {config_path} ({e})", flush=True)
            return {}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[RAG Ablation] Failed to parse JSON in {config_path}: {e}", flush=True)
            return {}

        if not isinstance(parsed, dict):
            print(f"[RAG Ablation] Invalid ablation config format in {config_path}: expected object", flush=True)
            return {}
        return parsed

    def _extract_ablation_settings(self, ablation_id: str) -> Optional[Dict[str, Any]]:
        """Extracts settings object for a given ablation_id, if configured."""
        configs = self._load_ablation_configs()
        settings = configs.get(ablation_id)
        if settings is None:
            return None
        if not isinstance(settings, dict):
            print(f"[RAG Ablation] Invalid settings entry for ablation_id={ablation_id!r}: expected object", flush=True)
            return None
        return settings

    def _apply_ablation_settings(self) -> None:
        """Applies ablation settings to runtime toggles when available."""
        if self.ablation_settings is None:
            return

        self.use_progressive_filtering = bool(
            self.ablation_settings.get("progressive_filtering_on", self.use_progressive_filtering)
        )
        self.use_confidence_eval = bool(
            self.ablation_settings.get("confidence_on", self.use_confidence_eval)
        )
        self.use_web_search = bool(
            self.ablation_settings.get("web_search_on", self.use_web_search)
        )
        self.use_domain_filter = bool(
            self.ablation_settings.get("domain_filter_on", self.use_domain_filter)
        )
        self.use_ingestion_loop = bool(
            self.ablation_settings.get("ingestion_loop_on", self.use_ingestion_loop)
        )

    def _build_tools_list(self) -> List[Any]:
        """Builds tool list deterministically from resolved toggles."""
        tools: List[Any] = [self._tracked_retrieve_content]
        if self.use_confidence_eval:
            tools.append(self._tracked_evaluate_confidence)
        if self.use_web_search:
            tools.append(self._tracked_web_search)
            tools.append(self._tracked_extract_keywords)
        if self.use_ingestion_loop:
            tools.append(self._tracked_add_web_content)
            tools.append(self._tracked_add_pdf_content)
        return tools

    def _get_agent_instruction(self) -> str:
        """Returns the instruction variant using ablation_id with fallback."""
        templates = self._load_instruction_templates()
        if self.ablation_id in templates:
            self.applied_instruction_key = self.ablation_id
            return templates[self.ablation_id]

        fallback_key = "confidence_on" if self.use_confidence_eval else "confidence_off"
        self.applied_instruction_key = fallback_key
        print(
            f"[RAG Ablation] Instruction template not found for ablation_id={self.ablation_id!r}; "
            f"falling back to {fallback_key!r}.",
            flush=True,
        )
        return templates[fallback_key]

    def _get_ablation_context(self) -> Dict[str, Any]:
        """Returns current ablation context (plumbing only; no behavior mapping yet)."""
        return {
            "ablation_id": self.ablation_id,
            "ablation_settings": self.ablation_settings,
            "use_confidence_eval": self.use_confidence_eval,
            "use_web_search": self.use_web_search,
            "use_ingestion_loop": self.use_ingestion_loop,
            "use_progressive_filtering": self.use_progressive_filtering,
            "use_domain_filter": self.use_domain_filter,
        }
    
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
        effective_location = location or getattr(self, "current_location", None)

        try:
            result = self.retrieve_content(
                query=query,
                location=effective_location,
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
                        location=effective_location,
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

    def reload_existing_collection(self):
        """Reload existing Chroma collection and rebind ALL dependent components."""

        name = "meta-mirage_collection"

        print(f"[RAG reload] Reloading existing collection: {name}", flush=True)

        # 🔥 IMPORTANT: use get_collection (NOT get_or_create)
        self.collection = self.client.get_collection(
            name=name,
            embedding_function=self.embedding_function,
        )

        print(f"[RAG reload] Collection count: {self.collection.count()}", flush=True)

        # 🔥 CRITICAL: rebind ALL components that depend on collection
        self.pdf_addition = PDFAddition(self.collection, self.content_utils, self.null_str)
        self.web_addition = WebAddition(self.collection, self.content_utils, self.null_str, self.null_int)
        self.confidence_evaluator = ConfidenceEvaluator(self.collection, self.content_utils)

        print(f"[RAG reload] Rebinding complete", flush=True)
        
    def _tracked_evaluate_confidence(self, *, query: str, location: Optional[str] = None,
                                      month_year: Optional[str] = None, title: Optional[str] = None, k: int = 5,
                                      use_progressive_filtering: Optional[bool] = None) -> Dict:
        """Evaluates confidence of retrieved evidence for a query.
        
        Use this tool AFTER calling retrieve_content to determine if the retrieved information is reliable.
        This is MANDATORY - you MUST call this after every retrieval attempt.
        
        Args:
            query: User query
            location: Optional geographic filter
            month_year: Optional temporal filter
            title: Optional document title filter
            k: Number of chunks to retrieve (default: 5)
            use_progressive_filtering: Optional per-call override. If omitted,
                this uses the class-level ablation setting `self.use_progressive_filtering`.
            
        Returns:
            Dict with status, confidence_level ("high"/"medium"/"low"), confidence_score, and diagnostics
        """
        import sys
        print(f"[RAG Tools] evaluate_retrieval_confidence: CALLED", flush=True)
        effective_location = location or getattr(self, "current_location", None)
        effective_use_progressive_filtering = (
            use_progressive_filtering
            if use_progressive_filtering is not None
            else self.use_progressive_filtering
        )
        result = self.confidence_evaluator.evaluate_retrieval_confidence(
            query=query,
            location=effective_location,
            month_year=month_year,
            title=title,
            k=k,
            use_progressive_filtering=effective_use_progressive_filtering,
        )
        status = result.get("status", "unknown")
        if status == "success":
            confidence_level = result.get("confidence_level", "unknown")
            print(f"[RAG Tools] ✓ evaluate_retrieval_confidence: SUCCESS (confidence: {confidence_level})", flush=True)
        else:
            print(f"[RAG Tools] ✗ evaluate_retrieval_confidence: FAILED - {result.get('error_message', 'Unknown error')}", flush=True)
        return result
    
    def _tracked_web_search(
        self,
        *,
        query: str,
        results_to_extract_count: int = 10,
        location: Optional[str] = None,
        use_domain_filter: Optional[bool] = None,
    ) -> Dict:
        """Searches the web for relevant information and extracts clean text.
        
        Use this tool ONLY when confidence_level is "low" after evaluating retrieval confidence.
        This tool searches the web for up-to-date information not available in the knowledge base.
        
        Args:
            query: The search query (use extract_keywords first to optimize the query)
            results_to_extract_count: Number of web results to retrieve and process (default: 10)
            location: Optional geographic context (e.g. "Minnesota, Stearns County").
                When provided, restricts results to .edu domains in that location and hardiness zone.
            use_domain_filter: Optional per-call override for domain filtering. If omitted,
                this uses the class-level ablation setting `self.use_domain_filter`.
            
        Returns:
            Dict with status, query, results (list of dicts with title, url), and error_message if failed
        """
        import sys
        print(f"[RAG Tools] web_search: CALLED (query: {query[:50]}...)", flush=True)
        effective_location = location or getattr(self, "current_location", None)
        effective_use_domain_filter = (
            use_domain_filter if use_domain_filter is not None else self.use_domain_filter
        )
        result = self.web_search.web_search(
            query,
            results_to_extract_count,
            location=effective_location,
            use_domain_filter=effective_use_domain_filter,
        )
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
            use_progressive_filtering: Optional[bool] = None,
        ) -> dict:
        """Retrieves relevant content with optional progressive metadata filtering.

        Args:
            query: Query text to retrieve against.
            location: Optional geographic location used to derive hardiness zone.
            month_year: Optional month/year metadata filter.
            title: Optional title metadata filter.
            use_progressive_filtering: Optional per-call override. If omitted,
                this uses `self.use_progressive_filtering`.
        """

        if not query or not query.strip():
            return {
                "status": "error",
                "error_message": "Empty query provided",
                "results": [],
            }

        print(f"[RAG Tools] collection.count()={self.collection.count()}", flush=True)
        effective_use_progressive_filtering = (
            use_progressive_filtering
            if use_progressive_filtering is not None
            else self.use_progressive_filtering
        )
        used_filter, strategy, results = self.content_utils.retrieve_with_priority_filters(
            query=query,
            collection=self.collection,
            location=location,
            month_year=month_year,
            title=title,
            use_progressive_filtering=effective_use_progressive_filtering,
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
        instruction_text = self._get_agent_instruction()
        
        # Debug: Print tool information
        print(f"[RAG Agent Init] Creating agent with model: {model_name}")
        print(f"[RAG Agent Init] ablation_id={self.ablation_id}")
        if self.ablation_settings is None:
            print(f"[RAG Agent Init] No ablation settings found for ablation_id={self.ablation_id}; using current defaults.", flush=True)
        else:
            print(f"[RAG Agent Init] Ablation settings: {self.ablation_settings}", flush=True)
        print(f"[RAG Agent Init] Instruction template key: {self.applied_instruction_key}", flush=True)
        print(f"[RAG Agent Init] Toggles: use_confidence_eval={self.use_confidence_eval}, use_web_search={self.use_web_search}, use_ingestion_loop={self.use_ingestion_loop}, use_progressive_filtering={self.use_progressive_filtering}, use_domain_filter={self.use_domain_filter}")
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
            instruction=instruction_text,
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