from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from tools.web_search_tool import web_search, web_fetch

# ---- CONFIG FOR SGLANG + QWEN2.5-VL ----
SGLANG_BASE_URL = "http://127.0.0.1:11434/v1"
SGLANG_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
API_KEY = "EMPTY"
# ---------------------------------


def build_agent() -> LlmAgent:
    model = LiteLlm(
        model=f"openai/{SGLANG_MODEL}",
        api_base=SGLANG_BASE_URL,
        api_key=API_KEY,
        additional_kwargs={
            "tool_choice": "auto",
        },
    )

    instruction = """You are a helpful assistant that uses tools to answer questions.

You have access to:
- web_search(query, results_to_extract_count) - searches the web
- web_fetch(url) - fetches content from a URL

When you need current information, immediately use the tools. Do not describe what you will do - just do it.

After using tools, provide a clear answer with sources."""

    return LlmAgent(
        name="websearch_agent",
        model=model,
        instruction=instruction.strip(),
        tools=[web_search, web_fetch],
    )