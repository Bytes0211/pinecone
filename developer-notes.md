
# ⭐ High‑Level Summary

This script performs a **complete Pinecone vector database workflow**:

1. **Load records** from a file (`records.txt`)  
2. **Embed text** using Pinecone’s hosted embedding model  
3. **Ensure the index exists** with the correct vector dimension  
4. **Upsert vectors** (id + embedding + metadata)  
5. **Fetch one vector** to verify everything worked  

It’s a minimal, end‑to‑end RAG ingestion pipeline.

---

# 📌 Imports and Setup

```python
import ast
import asyncio
import functools
import inspect
import logging
import os
import time
from typing import Any, List, Sequence

from colorlog import ColoredFormatter
from dotenv import load_dotenv
from pinecone import PineconeAsyncio, ServerlessSpec
from tqdm import tqdm
```

- `ast` is used to safely parse a Python file containing a `records = [...]` list.
- `asyncio` drives the async event loop.
- `functools` / `inspect` support the `@timed_step` decorator for both sync and async functions.
- `logging` + `colorlog` for colored, structured logs.
- `dotenv` loads environment variables from `.env`.
- `PineconeAsyncio` is the async Pinecone client.
- `ServerlessSpec` configures Pinecone's serverless index.
- `tqdm` provides progress bars for embedding and upserting.

---

# 📌 Load API Key

```python
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
if not API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")
```

Loads the Pinecone API key from environment variables.  
If missing → fail early.

---

# 📌 Configuration Constants

```python
INDEX_NAME = "records-index"
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"
MODEL = "llama-text-embed-v2"
RECORDS_PATH = "records.txt"
```

These define:

- Index name  
- Similarity metric  
- Cloud + region for serverless index  
- Embedding model  
- Path to the input records file  

---

# 📌 `load_records()` — Load Records from a Python File

```python
def load_records(path: str) -> List[dict]:
```

This function loads a Python file containing:

```python
records = [
    {"_id": "...", "chunk_text": "...", ...},
    ...
]
```

### Why AST?

Using `ast.parse()` ensures:

- The file is parsed safely  
- Only literal values are evaluated  
- No arbitrary code execution  

### What it does:

1. Reads the file  
2. Parses it into an AST  
3. Searches for a top‑level assignment named `records`  
4. Extracts its literal value  
5. Returns the list of dicts  

If no `records` variable is found → raises an error.

---

# 📌 `ensure_index()` — Create or Recreate the Index

```python
async def ensure_index(pc: PineconeAsyncio, name: str, dimension: int):
```

This function ensures the Pinecone index exists **with the correct vector dimension**.

### Why this matters:

Vector DBs require a **fixed dimension**.  
If your embeddings are 1024‑dim, the index must also be 1024‑dim.

### Steps:

1. Try to describe the index  
2. If it exists:
   - Check its dimension  
   - If mismatched → delete and recreate  
   - If correct → skip creation  
3. If it doesn’t exist → create it  

### Uses `ServerlessSpec`:

```python
spec=ServerlessSpec(cloud=CLOUD, region=REGION)
```

This configures Pinecone’s serverless index.

---

# 📌 `embed_texts()` — Generate Embeddings

```python
async def embed_texts(pc: PineconeAsyncio, texts: Sequence[str]) -> List[List[float]]:
```

This function calls Pinecone’s **inference API** to embed text.

### What it does:

1. Calls:

```python
pc.inference.embed(
    model=MODEL,
    inputs=list(texts),
    parameters={"input_type": "passage"},
)
```

2. Extracts the embedding vectors from the response  
3. Validates that each item contains `values`  
4. Returns a list of float vectors  

### Error handling:

- If no data returned → error  
- If any embedding missing → error  

This ensures the ingestion pipeline fails fast.

---

# 📌 `main()` — The Full Workflow

This is the orchestrator.

---

## Step 1 — Initialize Pinecone Client

```python
async with PineconeAsyncio(api_key=API_KEY) as pc:
```

The client is opened as an async context manager so its underlying aiohttp sessions are closed automatically when the block exits.

---

## Step 2 — Load Records

```python
records = load_records(RECORDS_PATH)
```

Each record must contain:

- `_id` → vector ID  
- `chunk_text` → text to embed  
- other fields → metadata  

---

## Step 3 — Extract Texts, IDs, Metadata

```python
texts.append(chunk_text)
ids.append(rec_id)
metadatas.append({k: v for k, v in r.items() if k not in ("_id", "chunk_text")})
```

Metadata is everything except `_id` and `chunk_text`.

---

## Step 4 — Embed the Texts

```python
embeddings = embed_texts(pc, texts)
```

Validates:

- Embeddings exist  
- Count matches number of records  

---

## Step 5 — Ensure Index Exists

```python
dimension = len(embeddings[0])
ensure_index(pc, INDEX_NAME, dimension)
```

The dimension is derived from the first embedding.

---

## Step 6 — Upsert Vectors

```python
desc = await pc.describe_index(INDEX_NAME)
async with pc.IndexAsyncio(host=desc.host) as index:
    vectors = [
        {"id": vid, "values": vec, "metadata": meta}
        for vid, vec, meta in zip(ids, embeddings, metadatas)
    ]
    await upsert_vectors(index, vectors)
```

`IndexAsyncio` is also opened as an async context manager for clean session cleanup. Vectors are upserted concurrently in batches controlled by `UPSERT_BATCH_SIZE` and `UPSERT_CONCURRENCY`.

Each vector contains:

- `id`  
- `values` (embedding)  
- `metadata`  

This is the core ingestion step.

---

## Step 7 — Fetch One Vector

```python
first_id = ids[0]
fetch_result = await index.fetch(ids=[first_id])
```

This verifies the index is working.

---

# ⭐ What This Script Actually Accomplishes

This is a **complete RAG ingestion pipeline**:

1. Load raw text chunks  
2. Embed them  
3. Create a vector index  
4. Insert vectors  
5. Validate retrieval  

It’s the minimal backbone of:

- Document ingestion  
- Knowledge base creation  
- RAG pipelines  
- Semantic search systems  
- Chat-with-your-data apps  

---
