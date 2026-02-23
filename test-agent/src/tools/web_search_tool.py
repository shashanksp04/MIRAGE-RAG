from typing import Dict
from web_search import WebSearch

# ---- CONFIG (NO ENV REQUIRED) ----
YOU_API_KEY = "ydc-sk-988fe646a127e2ca-zHOgmT2slT02L28HZttsS5FuHH8VH3Nk-2c7423e1"
YOU_BASE_URL = "https://ydc-index.io/v1/search"
# ---------------------------------

_search_client = WebSearch(api_key=YOU_API_KEY, base_url=YOU_BASE_URL)


def web_search(query: str, results_to_extract_count: int = 10) -> Dict:
    out = _search_client.web_search(query=query, results_to_extract_count=results_to_extract_count)
    if out.get("status") == "error":
        out["should_retry"] = False
    else:
        out["should_retry"] = False  # generally you want the model to STOP after it gets results
    out["query_used"] = query
    return out

def web_fetch(url: str) -> Dict:
    """
    Fetch and extract content from a specific URL.
    
    Args:
        url: The URL to fetch content from
    
    Returns:
        Dictionary with status, url, and extracted content
    """
    content = _search_client.extract_data(url)
    
    if content:
        # Limit content to avoid token overflow (adjust as needed)
        max_length = 3000
        truncated_content = content[:max_length] if len(content) > max_length else content
        
        return {
            "status": "success",
            "url": url,
            "content": truncated_content,
            "truncated": len(content) > max_length,
            "original_length": len(content)
        }
    else:
        return {
            "status": "error",
            "url": url,
            "error_message": "Failed to extract content from URL"
        }
