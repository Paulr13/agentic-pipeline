# agentic-pipeline

Minimal agentic pipeline framework in pure Python stdlib — no pip installs. Agent loop with tool calling, resumable pipelines with SQLite checkpointing, persistent memory, and a worked example you can run against any OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama).

## Why another agent framework

Most agent frameworks are heavy: pydantic trees, async everywhere, provider lock-in. This is the opposite — ~450 lines you can read in one sitting, that still cover the three things a production agent needs:

- **Agent loop** that calls tools, survives tool errors, and stops cleanly
- **Pipelines** that checkpoint every step to SQLite and resume after a crash
- **Memory** that survives restarts (append-only JSONL, keyword recall)

## Layout

| File | Lines | What |
|---|---|---|
| `agentic/llm.py` | ~80 | OpenAI-compatible client (urllib, no deps) |
| `agentic/agent.py` | ~90 | Tool-calling loop: unknown tools and bad args become tool results, not crashes |
| `agentic/pipeline.py` | ~110 | Step runner: retries, SQLite checkpoints, resumable by `run_id` |
| `agentic/memory.py` | ~60 | JSONL append + term-overlap recall, thread-safe |
| `examples/research_agent.py` | ~80 | End-to-end: agent takes notes into memory across sessions |
| `tests/test_agentic.py` | ~120 | Full suite, scripted fake LLM — no network needed |

## Quick start

```bash
export LLM_BASE=http://127.0.0.1:8080/v1   # llama-server, vLLM, Ollama, ...
export LLM_MODEL=your-model
python3 examples/research_agent.py "why is kv cache quantization safe"
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

## Design notes

- **No `requirements.txt` on purpose.** urllib + sqlite3 + json cover everything; if it installs, it runs anywhere — including locked-down boxes.
- **Failures are data.** Tool exceptions and unknown tools come back to the model as results, so the loop self-corrects instead of dying mid-task.
- **Checkpoints before side effects.** The pipeline commits a step's output to SQLite *before* the next step starts — a crashed run resumes with finished steps skipped, verified by the test suite.
- **Swap memory later, not now.** Term-overlap recall is deliberately dumb; the interface (`add`/`recall`) is the seam where a vector DB plugs in.

## Status

Running the patterns it packages on my own boxes. MIT.