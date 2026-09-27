#!/usr/bin/env python3
"""agentic.memory — tiny persistent memory: append-only JSONL + keyword recall.

Good enough for single-host agents: facts survive restarts, recall scores by
term overlap. Swap for a vector DB when scale demands it.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class Memory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def add(self, text: str, kind: str = "fact", **meta) -> dict:
        rec = {"ts": time.time(), "kind": kind, "text": text, **meta}
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def recall(self, query: str, k: int = 5) -> list[dict]:
        """Score by distinct-term overlap; ties broken by recency."""
        if not self.path.exists():
            return []
        terms = {t.strip(".,!?;:").lower() for t in query.split() if len(t) > 2}
        scored = []
        with self._lock, self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rterms = {t.strip(".,!?;:").lower() for t in rec.get("text", "").split()}
                score = len(terms & rterms)
                if score:
                    scored.append((score, rec["ts"], rec))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        return [rec for _, _, rec in scored[:k]]

    def all(self, kind: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        recs = []
        with self._lock, self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # corrupt line: skip, don't crash readers
        return [r for r in recs if kind is None or r.get("kind") == kind]