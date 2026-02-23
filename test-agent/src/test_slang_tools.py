#!/usr/bin/env python3
"""Test if SGLang server is properly handling tool calls"""

import requests
import json

SGLANG_URL = "http://127.0.0.1:11434/v1/chat/completions"

# Test payload with tools
payload = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
        {"role": "user", "content": "What's the weather in Chicago?"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "results_to_extract_count": {
                            "type": "integer",
                            "description": "Number of results",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ],
    "tool_choice": "auto"
}

print("Testing SGLang tool calling...")
print(f"URL: {SGLANG_URL}")
print(f"Payload: {json.dumps(payload, indent=2)}\n")

try:
    response = requests.post(SGLANG_URL, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")