# Pinecone Vector Ingestion Pipeline

A minimal end-to-end RAG ingestion pipeline that loads text records, generates embeddings via Pinecone's inference API, and upserts them into a serverless Pinecone index.

## Overview

This pipeline performs the following workflow:

1. **Load records** — Parses `records.txt` (a Python-formatted file) using AST for safe literal evaluation
2. **Embed text** — Generates vector embeddings via Pinecone's hosted `llama-text-embed-v2` model
3. **Ensure index** — Creates or recreates a serverless Pinecone index with the correct dimension
4. **Upsert vectors** — Writes vectors (id + embedding + metadata) to the index
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

The script will log each step with colored, timestamped output and progress bars for embedding and upserting.

## Configuration

Constants are defined at the top of `main.py`:

- `INDEX_NAME` — Name of the Pinecone index (`records-index`)
- `METRIC` — Similarity metric (`cosine`)
- `CLOUD` / `REGION` — Serverless deployment target (`aws` / `us-east-1`)
- `MODEL` — Embedding model (`llama-text-embed-v2`)
- `RECORDS_PATH` — Path to the input records file (`records.txt`)

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
├── .env                 # Environment variables (API key)
├── main.py              # Ingestion pipeline entry point
├── records.txt          # Input dataset (Python list of dicts)
├── requirements.txt     # Python dependencies
├── developer-notes.md   # Detailed developer walkthrough
└── main-diagram.md      # Mermaid flow diagram of the pipeline
```

## Key Dependencies

- `pinecone` — Pinecone vector database client and inference API
- `python-dotenv` — Environment variable loading
- `colorlog` — Colored log output
- `tqdm` — Progress bars
