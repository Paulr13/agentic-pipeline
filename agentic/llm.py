#!/usr/bin/env python3
"""agentic.llm — OpenAI-compatible chat-completions client, stdlib only.

Works against llama.cpp (llama-server), vLLM, Ollama's OpenAI shim, or any
OpenAI-style endpoint. Env overrides: LLM_BASE, LLM_API_KEY, LLM_MODEL.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class LLM:
    """Thin chat-completions client. `chat()` normalizes the response to
    {"content": str|None, "tool_calls": [{"id", "name", "arguments": dict}]}."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or os.environ.get("LLM_BASE", "http://127.0.0.1:8080/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "") or None
        self.model = model or os.environ.get("LLM_MODEL", "local-model")
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.7, max_tokens: int = 1024) -> dict:
        payload: dict = {"model": self.model, "messages": messages,
                         "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
        data = self._post("/chat/completions", payload)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {e}") from e
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
            calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""),
                          "arguments": args})
        return {"content": msg.get("content"), "tool_calls": calls}

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code}: {e.read().decode(errors='ignore')[:300]}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"cannot reach {self.base_url}: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise LLMError(f"non-JSON response from {self.base_url}") from e


def to_openai_tool(name: str, description: str, parameters: dict) -> dict:
    """Adapter: (name, description, JSON-schema) -> OpenAI tools[] entry."""
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": parameters}}