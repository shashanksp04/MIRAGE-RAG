import json
import re
import ast
from typing import List, Dict
from chat_models.Client import Client


class KeywordExtractor:
    """
    Agent tool for extracting search-optimized keywords from a user query.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", openai_api_base: str = "http://127.0.0.1:11434/v1"):
        """
        Args:
            model_name: LLM used for keyword extraction
            openai_api_base: Base URL of the OpenAI-compatible API
        """
        self.model_name = model_name
        self.openai_api_base = openai_api_base
        # Don't create client here - create fresh one for each call to avoid message accumulation

    def _extract_first_json_blob(self, text: str) -> str | None:
        """
        Try to pull out the first JSON array/object from a messy model response.
        """
        # remove code fences
        t = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

        # prefer an array if present
        m = re.search(r"\[[\s\S]*?\]", t)
        if m:
            return m.group(0)

        # else try object
        m = re.search(r"\{[\s\S]*?\}", t)
        if m:
            return m.group(0)

        return None


    def _parse_keywords_any(self, raw_text: str):
        """
        Accepts:
        - ["a","b"]
        - {"keywords":["a","b"]}
        - "['a','b']" (stringified)
        - single quotes / trailing commas (fallback via ast.literal_eval)
        """
        blob = self._extract_first_json_blob(raw_text) or raw_text.strip()

        # First try strict JSON
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # common fixes: single quotes, trailing commas → use python literal eval
            try:
                obj = ast.literal_eval(blob)
            except Exception as e:
                raise ValueError(f"Could not parse keywords JSON: {e}")

        # If model returned {"keywords": [...]}
        if isinstance(obj, dict):
            obj = obj.get("keywords") or obj.get("keyphrases") or obj.get("terms")

        # If model returned a *string* that itself contains JSON
        if isinstance(obj, str):
            obj = obj.strip()
            try:
                obj = json.loads(obj)
            except Exception:
                try:
                    obj = ast.literal_eval(obj)
                except Exception as e:
                    raise ValueError(f"Keywords were a string but not parseable: {e}")

        if not isinstance(obj, list):
            raise ValueError("Parsed output was not a list")

        # normalize + filter
        out = []
        for kw in obj:
            if isinstance(kw, str):
                s = kw.strip()
                if s:
                    out.append(s)

        return out

    def extract_keywords(self, *, query: str) -> Dict:
        """Extracts search-optimized keywords from a query.

        Use this tool when:
        - Preparing a query for web search
        - Improving recall for SerpAPI / Google-style search

        The tool:
        - Extracts important keywords and phrases
        - Combines related terms
        - Quotes multi-word concepts
        - Orders keywords from general to specific

        Args:
            query: User query to extract keywords from

        Returns:
            Success:
            {
              "status": "success",
              "keywords": [str, ...]
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
                "error_message": "Empty query provided",
            }

        prompt = f"""
            You are an intelligent keyword extraction assistant.

            Your task is to extract the most important and relevant keywords or short phrases
            from the user query below, and organize them in a logical order suitable for
            Google-style web search.

            Return ONLY a JSON array (top-level list) of strings.
            No prose. No markdown. No code fences.

            Rules:
            - Combine related words into phrases when appropriate.
            - Include entities such as crops, pests, locations, years, organizations, or events.
            - If a keyword or phrase contains multiple words forming a fixed concept,
            enclose it in double quotes (" ").
            - Do NOT quote single words.
            - Order keywords from general to specific.

            Example:
            Input: "impact of drought on corn and soybean pests in Maryland 2022"
            Output: ["drought impact", "corn", "soybean pests", "Maryland", "2022"]

            User Query:
            {query}
            """

        try:
            client = Client(model_name=self.model_name, openai_api_base=self.openai_api_base, messages=[])
            raw_text = client.chat(prompt=prompt)
            print("RAW:", repr(raw_text))

            try:
                keywords = self._parse_keywords_any(raw_text)
            except Exception:
                return {"status": "error", 
                        "error_message": "Model did not return a valid JSON list",
                        "raw_text_preview": (raw_text or "")[:500],}

            if not keywords:
                return {
                    "status": "error",
                    "error_message": "No keywords extracted",
                }

            return {
                "status": "success",
                "keywords": keywords,
            }

        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Keyword extraction failed: {str(e)}",
            }
