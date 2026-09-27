#!/usr/bin/env python3
"""agentic.pipeline — resumable step pipelines with SQLite checkpointing.

Each step's output is committed to SQLite before the next step starts, so a
crashed pipeline resumes exactly where it left off: re-run with the same
run_id and finished steps are skipped.

Steps are plain callables receiving the previous step's output. Wrap model
calls in a step; the pipeline stays dumb and auditable.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class StepResult:
    __slots__ = ("name", "output", "ok", "error", "attempts")

    def __init__(self, name: str, output, ok: bool, error: str | None, attempts: int):
        self.name, self.output, self.ok, self.error, self.attempts = name, output, ok, error, attempts


class Pipeline:
    def __init__(self, db_path: str | Path, run_id: str, max_retries: int = 2,
                 retry_delay: float = 2.0):
        self.db_path = str(db_path)
        self.run_id = run_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS steps ("
            "run_id TEXT NOT NULL, name TEXT NOT NULL, idx INTEGER NOT NULL,"
            "output TEXT, ok INTEGER NOT NULL, attempts INTEGER NOT NULL,"
            "ts REAL NOT NULL, PRIMARY KEY (run_id, idx))")

    def run(self, steps: list) -> list[StepResult]:
        """Steps: list of callables (prev_output) -> output. Output must be
        JSON-serializable to be checkpointed; non-serializable values are
        still passed along in-process (marked as such in the DB)."""
        results: list[StepResult] = []
        prev = None
        for i, step in enumerate(steps):
            name = getattr(step, "__name__", f"step_{i}")
            done = self._conn.execute(
                "SELECT output FROM steps WHERE run_id=? AND idx=? AND ok=1",
                (self.run_id, i)).fetchone()
            if done is not None:
                prev = _maybe_json(done[0])
                results.append(StepResult(name, prev, True, None, 0))
                continue
            attempts, err = 0, None
            while attempts <= self.max_retries:
                attempts += 1
                try:
                    prev = step(prev)
                    err = None
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    if attempts <= self.max_retries:
                        time.sleep(self.retry_delay)
            ok = err is None
            out_repr = _safe_json(prev) if ok else None
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO steps VALUES (?,?,?,?,?,?,?)",
                    (self.run_id, name, i, out_repr, int(ok), attempts, time.time()))
            results.append(StepResult(name, prev, ok, err, attempts))
            if not ok:
                break  # stop at first failed step; resume from here on re-run
        return results

    def summary(self) -> str:
        rows = self._conn.execute(
            "SELECT idx, name, ok, attempts FROM steps WHERE run_id=? ORDER BY idx",
            (self.run_id,)).fetchall()
        return "\n".join(f"  [{i}] {n}: {'ok' if o else 'FAILED'} ({a} attempt(s))"
                         for i, n, o, a in rows)


def _safe_json(value) -> str:
    import json
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps({"_unserializable": repr(value)[:200]})


def _maybe_json(text: str):
    import json
    try:
        v = json.loads(text)
        if isinstance(v, dict) and "_unserializable" in v:
            return None
        return v
    except (json.JSONDecodeError, TypeError):
        return text