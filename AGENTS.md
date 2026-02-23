# AGENTS.md

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
- `test_main.py` — Pytest suite covering sync pipeline functions (`load_records`, `timed_step`)
- `process-flow.md` — Mermaid flowchart of the pipeline
- `CLAUDE.md` — Project conventions for Claude AI assistants
- `README.md` — Project overview, setup, usage, and configuration reference
- `check_pc.py` — Utility script to inspect `PineconeAsyncio` attributes
- `logs/pipeline.log` — Rotating log file (auto-created at runtime; git-ignored)

## Conventions
- **Logging** — Use the existing `colorlog`-based logger (`logger`), not `print()`, for operational output
- **Timing** — Use the `@timed_step("Step Name")` decorator for any new pipeline steps
- **Configuration** — Constants are module-level (`INDEX_NAME`, `METRIC`, `CLOUD`, `REGION`, `MODEL`, `RECORDS_PATH`, `UPSERT_BATCH_SIZE`, `UPSERT_CONCURRENCY`, `EMBED_BATCH_SIZE`, `MAX_RETRIES`, `BACKOFF_BASE`, `BACKOFF_JITTER`, `LOG_DIR`, `LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`); do not hardcode values inside functions
- **Resource management** — Use `async with` for `PineconeAsyncio` and `IndexAsyncio` to ensure aiohttp sessions are closed properly
- **Retries** — Use `run_with_retries()` for any remote API call (embed, upsert); it applies exponential backoff with jitter
- **Error handling** — Fail fast with descriptive errors; validate inputs early (e.g. missing API key, missing record fields)
- **Type hints** — Functions use type annotations from `typing`
- **Testing** — Tests live in `test_main.py`; run with `pytest test_main.py -v`. Use `unittest.mock` to mock the Pinecone client; never call the real API in tests

## Record Schema
Each record in `records.txt` must have:
- `_id` (str) — Unique vector ID
- `chunk_text` (str) — Text to embed
- Any additional keys become Pinecone vector metadata

## Logging
- **Console** — Colored output via `colorlog` at `INFO` level and above
- **File** — Plain-text rotating log at `logs/pipeline.log` at `DEBUG` level and above
- **Rotation** — Controlled by `LOG_MAX_BYTES` (default 5 MB) and `LOG_BACKUP_COUNT` (default 3 backups: `pipeline.log.1`, `.2`, `.3`)
- **Directory** — `logs/` is created automatically via `os.makedirs` on startup and is listed in `.gitignore`

## Environment
- Python 3.12+ (managed via `uv`)
- Dependencies managed via `requirements.txt` and a local `.venv` (created with `uv venv`)
- API key loaded from `.env` via `python-dotenv`

## Common Tasks
- **Add records** — Append dicts to the `records` list in `records.txt`
- **Change embedding model** — Update the `MODEL` constant in `main.py`
- **Change index settings** — Update `INDEX_NAME`, `METRIC`, `CLOUD`, or `REGION` constants
- **Change log settings** — Update `LOG_DIR`, `LOG_FILE`, `LOG_MAX_BYTES`, or `LOG_BACKUP_COUNT` constants in `main.py`
- **Run the pipeline** — `python main.py`
- **Run tests** — `pytest test_main.py -v`
