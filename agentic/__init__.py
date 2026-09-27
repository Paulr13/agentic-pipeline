"""agentic — minimal agentic pipeline framework (stdlib only).

Agent loop, resumable pipelines, persistent memory. Bring your own
OpenAI-compatible endpoint (llama.cpp, vLLM, ...).
"""
from .llm import LLM, LLMError, to_openai_tool
from .agent import Agent, Tool
from .pipeline import Pipeline
from .memory import Memory

__all__ = ["Agent", "Tool", "LLM", "LLMError", "Pipeline", "Memory",
           "to_openai_tool"]
__version__ = "0.1.0"