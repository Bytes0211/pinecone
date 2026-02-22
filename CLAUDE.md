# CLAUDE.md

## Project Overview
This is a Pinecone vector database ingestion pipeline written in Python. It embeds text records using Pinecone's hosted inference API and upserts them into a serverless index. It serves as a minimal RAG ingestion backbone.

## Architecture
- **Single-script pipeline** — All logic lives in `main.py`, organized into decorated functions
- **AST-based record loading** — `records.txt` is a Python file parsed via `ast` to safely extract a `records` list
- **Pinecone serverless** — The index is deployed on AWS us-east-1 using `ServerlessSpec`
- **Embedding model** — `llama-text-embed-v2` via `pc.inference.embed()`

## Key Files
- `main.py` — Entry point; contains all pipeline steps (load, embed, ensure index, upsert, fetch)
- `records.txt` — Dataset; Python file defining a `records = [...]` list of dicts with `_id`, `chunk_text`, and metadata fields
- `.env` — Stores `PINECONE_API_KEY` (never commit to version control)
- `requirements.txt` — Pinned Python dependencies
- `docs/developer-notes.md` — Detailed walkthrough of each function and step
- `test_main.py` — Pytest suite covering all pipeline functions and the main workflow
- `main-diagram.md` — Mermaid flowchart of the pipeline

## Conventions
- **Logging** — Use the existing `colorlog`-based logger (`logger`), not `print()`, for operational output
- **Timing** — Use the `@timed_step("Step Name")` decorator for any new pipeline steps
- **Configuration** — Constants are module-level (`INDEX_NAME`, `METRIC`, `CLOUD`, `REGION`, `MODEL`, `RECORDS_PATH`); do not hardcode values inside functions
- **Error handling** — Fail fast with descriptive errors; validate inputs early (e.g. missing API key, missing record fields)
- **Type hints** — Functions use type annotations from `typing`
- **Testing** — Tests live in `test_main.py`; run with `pytest test_main.py -v`. Use `unittest.mock` to mock the Pinecone client; never call the real API in tests

## Record Schema
Each record in `records.txt` must have:
- `_id` (str) — Unique vector ID
- `chunk_text` (str) — Text to embed
- Any additional keys become Pinecone vector metadata

## Environment
- Python 3.10+
- Dependencies managed via `requirements.txt` and a local `.venv`
- API key loaded from `.env` via `python-dotenv`

## Common Tasks
- **Add records** — Append dicts to the `records` list in `records.txt`
- **Change embedding model** — Update the `MODEL` constant in `main.py`
- **Change index settings** — Update `INDEX_NAME`, `METRIC`, `CLOUD`, or `REGION` constants
- **Run the pipeline** — `python main.py`
- **Run tests** — `pytest test_main.py -v`
