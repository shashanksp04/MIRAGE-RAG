from typing import Dict, Optional
import requests
import json
import trafilatura


class WebSearch:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ydc-index.io/v1/search",
    ):
        self.base_url = base_url
        self.api_key = api_key

    def extract_data(self, URL: str) -> Optional[str]:
        downloaded = trafilatura.fetch_url(URL)
        if not downloaded:
            return None
        return downloaded

    def web_search(self, query: str, results_to_extract_count: int = 10) -> Dict:
        """Search the web (You.com index API) and return titles + URLs."""
        if not self.api_key:
            return {"status": "error", "error_message": "Missing You.com API key"}

        params = {"query": query, "count": results_to_extract_count}
        headers = {"X-API-Key": self.api_key}

        try:
            response = requests.get(self.base_url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            return {"status": "error", "error_message": str(e)}

        try:
            data = response.json()
        except json.JSONDecodeError:
            return {"status": "error", "error_message": "Failed to parse JSON response"}

        results = []
        results_data = data.get("results", {})
        for item in results_data.get("web", []):
            url = item.get("url")
            if not url:
                continue
            results.append({"title": item.get("title"), "url": url})

        return {"status": "success", "query": query, "results": results}
