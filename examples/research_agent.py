#!/usr/bin/env python3
"""examples/research_agent.py — end-to-end example: agent + tools + memory.

A research agent that answers a question by taking notes into persistent
memory, then answers from memory on repeat runs. Fully offline-safe: swap in
any OpenAI-compatible endpoint via env (LLM_BASE, LLM_MODEL, LLM_API_KEY).

usage: python3 examples/research_agent.py "why is kv cache quantization safe"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic import Agent, LLM, Memory, Tool

MEM_PATH = Path(os.environ.get("AGENT_MEM_PATH", "agent_memory.jsonl"))


def build_tools(mem: Memory) -> list[Tool]:
    def take_note(topic: str, note: str) -> str:
        """Record a research note. Call after each finding you want to keep."""
        mem.add(note, kind="note", topic=topic)
        return f"noted: {note}"

    def recall(topic: str) -> str:
        """Show notes already taken, filtered by keyword."""
        hits = mem.recall(topic, k=10) or mem.all(kind="note")
        if not hits:
            return "(no notes yet)"
        return "\n".join(f"- {h['text']}" for h in hits)

    return [
        Tool("take_note", "Save a research note (topic, note).",
             {"type": "object",
              "properties": {"topic": {"type": "string"},
                             "note": {"type": "string"}},
              "required": ["topic", "note"]}, take_note),
        Tool("recall", "List notes taken so far (optional keyword filter).",
             {"type": "object",
              "properties": {"topic": {"type": "string"}},
              "required": []}, recall),
    ]


def main() -> int:
    question = " ".join(sys.argv[1:]) or "What should I research?"
    mem = Memory(MEM_PATH)
    llm = LLM()
    agent = Agent(
        llm,
        tools=build_tools(mem),
        system=("You are a research assistant. Investigate the user's question "
                "step by step, saving each finding with take_note, then give a "
                "short final answer. Cite your notes by topic."),
        max_steps=10,
    )
    # Seed context from prior runs — memory across sessions.
    prior = mem.recall(question, k=5)
    task = question if not prior else question + "\n\nNotes from earlier sessions:\n" + \
        "\n".join(f"- {h['text']}" for h in prior)

    print(f"task: {question}")
    print(f"memory: {MEM_PATH} ({len(mem.all())} records)")
    answer = agent.run(task)
    mem.add(f"Q: {question} -> A: {answer}", kind="answer")
    print("\n--- answer ---\n" + answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())