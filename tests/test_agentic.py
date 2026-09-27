#!/usr/bin/env python3
"""Unit tests: run with `python3 -m unittest discover -s tests -v`.

No network and no LLM required — the LLM is stubbed with a scripted fake.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic import Agent, LLM, Memory, Pipeline, Tool


class FakeLLM(LLM):
    """Scripted responses — exercises the loop without any endpoint."""

    def __init__(self, script: list[dict]):
        super().__init__(base_url="http://127.0.0.1:9/v1")  # never contacted
        self.script = list(script)
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=1024):
        self.calls += 1
        return self.script.pop(0)


class TestAgentLoop(unittest.TestCase):
    def test_tool_call_then_answer(self):
        def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(FakeLLM([
            {"content": None, "tool_calls": [
                {"id": "1", "name": "add", "arguments": {"a": 2, "b": 3}}]},
            {"content": "the sum is 5", "tool_calls": []},
        ]), tools=[Tool("add", "add two ints",
                        {"type": "object", "properties": {"a": {"type": "integer"},
                                                          "b": {"type": "integer"}},
                         "required": ["a", "b"]}, add)])
        self.assertEqual(agent.run("what is 2+3"), "the sum is 5")

    def test_unknown_tool_and_bad_args_become_tool_errors(self):
        seen = []

        def echo(**kw):
            seen.append(kw)
            return "ok"

        agent = Agent(FakeLLM([
            {"content": None, "tool_calls": [
                {"id": "1", "name": "nope", "arguments": {}}]},
            {"content": None, "tool_calls": [
                {"id": "2", "name": "echo", "arguments": {"x": "hi"}}]},
            {"content": "done", "tool_calls": []},
        ]), tools=[Tool("echo", "echo kwargs",
                        {"type": "object", "properties": {}}, echo)])
        out = agent.run("go")
        self.assertEqual(out, "done")
        self.assertEqual(seen, [{"x": "hi"}])  # loop survived the unknown tool

    def test_max_steps_guard(self):
        boom = {"content": None, "tool_calls": [
            {"id": "1", "name": "noop", "arguments": {}}]}
        agent = Agent(FakeLLM([boom] * 10),
                      tools=[Tool("noop", "do nothing",
                                  {"type": "object", "properties": {}}, lambda: "")],
                      max_steps=3)
        self.assertEqual(agent.run("loop forever"), "(max steps reached without a final answer)")


class TestPipeline(unittest.TestCase):
    def test_run_and_resume(self):
        calls = {"n": 0}

        def step_a(prev):
            calls["n"] += 1
            return {"a": 1}

        def step_b(prev):
            calls["n"] += 1
            return {"b": prev["a"] + 1}

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.sqlite"
            p1 = Pipeline(db, run_id="r1", max_retries=0)
            res = p1.run([step_a, step_b])
            self.assertTrue(all(r.ok for r in res))
            self.assertEqual(calls["n"], 2)

            # Re-run same run_id: everything checkpointed, zero new calls.
            p2 = Pipeline(db, run_id="r1")
            res2 = p2.run([step_a, step_b])
            self.assertTrue(all(r.ok for r in res2))
            self.assertEqual(calls["n"], 2)

            # Fresh run_id re-executes.
            p3 = Pipeline(db, run_id="r2")
            p3.run([step_a])
            self.assertEqual(calls["n"], 3)

    def test_failure_stops_and_retry_counts(self):
        attempts = {"n": 0}

        def flaky(prev):
            attempts["n"] += 1
            raise ValueError("boom")

        with tempfile.TemporaryDirectory() as td:
            p = Pipeline(Path(td) / "p.sqlite", run_id="r1", max_retries=2, retry_delay=0)
            res = p.run([flaky])
            self.assertFalse(res[-1].ok)
            self.assertEqual(attempts["n"], 3)   # 1 + 2 retries
            self.assertIn("boom", res[-1].error or "")
            self.assertEqual(p.summary().count("FAILED"), 1)


class TestMemory(unittest.TestCase):
    def test_add_recall_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            m = Memory(Path(td) / "mem.jsonl")
            m.add("the user prefers q8_0 kv cache", kind="note")
            m.add("postgres wants btree indexes", kind="note")
            hits = m.recall("kv cache quantization", k=1)
            self.assertEqual(hits[0]["text"], "the user prefers q8_0 kv cache")

    def test_corrupt_line_survives(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mem.jsonl"
            m = Memory(path)
            m.add("good line", kind="note")
            with path.open("a") as f:
                f.write("{not json\n")
            self.assertEqual(len(m.all()), 1)  # skips the corrupt line


if __name__ == "__main__":
    unittest.main()