# Pinecone Vector Ingestion Pipeline

A minimal end-to-end RAG ingestion pipeline that loads text records, generates embeddings via Pinecone's inference API, and upserts them into a serverless Pinecone index.

## Overview

This pipeline performs the following workflow:

1. **Load records** — Parses `records.txt` (a Python-formatted file) using AST for safe literal evaluation
2. **Embed text** — Generates vector embeddings in batches via Pinecone's hosted `llama-text-embed-v2` model with retry logic
3. **Ensure index** — Creates or recreates a serverless Pinecone index with the correct dimension
4. **Upsert vectors** — Writes vectors (id + embedding + metadata) to the index using concurrent batched upserts
5. **Fetch & verify** — Retrieves a single vector to confirm ingestion succeeded

## Prerequisites

- Python 3.10+
- A [Pinecone](https://www.pinecone.io/) account and API key

## Setup

1. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**

   Create a `.env` file in the project root:

   ```
   PINECONE_API_KEY=<your-pinecone-api-key>
   ```

## Usage

```bash
python main.py
```

The script runs an async pipeline using `PineconeAsyncio`. It logs each step with colored, timestamped output to the console and plain-text rotating logs to `logs/pipeline.log`. Progress bars track embedding and upserting. Async clients are managed via `async with` context managers to ensure clean session teardown.

## Logging

- **Console** — Colored output via `colorlog` at `INFO` level and above
- **File** — Plain-text rotating log at `logs/pipeline.log` at `DEBUG` level and above
- **Rotation** — Controlled by `LOG_MAX_BYTES` (default 5 MB) and `LOG_BACKUP_COUNT` (default 3 backups: `pipeline.log.1`, `.2`, `.3`)
- **Directory** — `logs/` is created automatically on startup and is listed in `.gitignore`

## Testing

```bash
pytest test_main.py -v
```

The test suite uses `unittest.mock` to mock all Pinecone API calls — no API key or network access is required. Tests cover the synchronous components: `load_records` and `timed_step`.

## Configuration

Constants are defined at the top of `main.py`:

| Constant | Default | Description |
|---|---|---|
| `INDEX_NAME` | `records-index` | Name of the Pinecone index |
| `METRIC` | `cosine` | Similarity metric |
| `CLOUD` / `REGION` | `aws` / `us-east-1` | Serverless deployment target |
| `MODEL` | `llama-text-embed-v2` | Embedding model |
| `RECORDS_PATH` | `records.txt` | Path to the input records file |
| `UPSERT_BATCH_SIZE` | `50` | Vectors per upsert API call |
| `UPSERT_CONCURRENCY` | `8` | Max parallel upsert tasks |
| `EMBED_BATCH_SIZE` | `32` | Texts per embedding API call |
| `MAX_RETRIES` | `3` | Retry attempts for API calls |
| `BACKOFF_BASE` | `0.5` | Base delay (seconds) for exponential backoff |
| `BACKOFF_JITTER` | `0.3` | Max random jitter (seconds) added to backoff |
| `LOG_DIR` | `logs` | Directory for log files |
| `LOG_FILE` | `logs/pipeline.log` | Path to the rotating log file |
| `LOG_MAX_BYTES` | `5242880` (5 MB) | Max log file size before rotation |
| `LOG_BACKUP_COUNT` | `3` | Number of rotated log backups to keep |

## Record Format

Records are defined in `records.txt` as a Python list of dicts. Each record requires:

- `_id` — Unique vector identifier
- `chunk_text` — Text content to embed
- Additional fields become vector metadata (e.g. `category`)

```python
records = [
    {"_id": "rec1", "chunk_text": "The Eiffel Tower was completed in 1889.", "category": "history"},
    ...
]
```

## Project Structure

```
pinecone/
├── .env                     # Environment variables (API key; git-ignored)
├── .gitignore               # Git ignore rules
├── CLAUDE.md                # Project conventions for AI assistants
├── README.md                # This file
├── main.py                  # Ingestion pipeline entry point
├── test_main.py             # Pytest suite for sync pipeline functions
├── records.txt              # Input dataset (Python list of dicts)
├── requirements.txt         # Python dependencies
├── process-flow.md          # Mermaid flow diagram of the pipeline
├── docs/
│   └── developer-notes.md   # Detailed developer walkthrough
└── logs/                    # Rotating log files (auto-created; git-ignored)
    └── pipeline.log
```

## Key Dependencies

- `pinecone` — Pinecone vector database client and inference API
- `python-dotenv` — Environment variable loading
- `colorlog` — Colored log output
- `tqdm` — Progress bars
- `pytest` — Test framework (dev dependency)