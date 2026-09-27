#!/usr/bin/env python3
"""agentic.agent — tool-calling agent loop.

Run(task) drives the model until it stops calling tools or `max_steps` is hit.
Tool errors are fed back as results instead of crashing the loop.
"""
from __future__ import annotations

import json

from .llm import LLM, to_openai_tool


class Tool:
    """A callable the agent may invoke. `parameters` is a JSON-schema dict."""

    def __init__(self, name: str, description: str, parameters: dict, fn):
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.name, self.description, self.parameters, self.fn = name, description, parameters, fn

    def schema(self) -> dict:
        return to_openai_tool(self.name, self.description, self.parameters)


class Agent:
    def __init__(self, llm: LLM, tools: list[Tool] | None = None,
                 system: str = "", max_steps: int = 8, temperature: float = 0.7):
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.system = system
        self.max_steps = max_steps
        self.temperature = temperature

    def run(self, task: str, max_steps: int | None = None) -> str:
        limit = max_steps if max_steps is not None else self.max_steps
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": task})

        tool_defs = [t.schema() for t in self.tools.values()] or None
        for _ in range(limit):
            resp = self.llm.chat(messages, tools=tool_defs, temperature=self.temperature)
            calls = resp["tool_calls"]
            if not calls:
                return resp["content"] or ""
            messages.append(self._assistant_message(calls, resp["content"]))
            for call in calls:
                messages.append(self._tool_message(call))
        return "(max steps reached without a final answer)"

    # --- internals ---

    def _assistant_message(self, calls: list[dict], content: str | None) -> dict:
        return {"role": "assistant", "content": content,
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"],
                                             "arguments": json.dumps(c["arguments"])}}
                               for c in calls]}

    def _tool_message(self, call: dict) -> dict:
        tool = self.tools.get(call["name"])
        if tool is None:
            result = f"error: unknown tool '{call['name']}'"
        else:
            try:
                result = tool.fn(**call["arguments"])
            except TypeError as e:
                result = f"error: bad arguments for '{call['name']}': {e}"
            except Exception as e:  # tool crash must not kill the loop
                result = f"error: {type(e).__name__}: {e}"
        return {"role": "tool", "tool_call_id": call["id"],
                "name": call["name"], "content": str(result)}