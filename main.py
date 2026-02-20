# Auto-generated: Basic Pinecone workflow (async version)
# - Embed texts via Pinecone inference
# - Create index with derived dimension
# - Upsert vectors concurrently
# - Fetch one vector
# - Author: scotton

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

# ---------------------------------------------------------
# Logging Setup: Colored Logs + Timestamp + Level Coloring
# ---------------------------------------------------------
handler = logging.StreamHandler()
handler.setFormatter(
    ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(message_log_color)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={
            "message": {
                "INFO": "white",
                "DEBUG": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        },
    )
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
if not API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")

# Config
INDEX_NAME = "records-index"
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"
MODEL = "llama-text-embed-v2"
RECORDS_PATH = "records.txt"
UPSERT_BATCH_SIZE = 50
UPSERT_CONCURRENCY = 8


# ---------------------------------------------------------
# Utility: timing decorator (supports sync + async)
# ---------------------------------------------------------
def timed_step(step_name: str):
    """Decorator to measure execution time of sync or async functions."""

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger.info(f"⏳ Starting: {step_name}")
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                end = time.perf_counter()
                logger.info(f"✅ Completed: {step_name} in {end - start:.3f}s")
                return result

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.info(f"⏳ Starting: {step_name}")
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            logger.info(f"✅ Completed: {step_name} in {end - start:.3f}s")
            return result

        return sync_wrapper

    return decorator


# ---------------------------------------------------------
# Load records
# ---------------------------------------------------------
@timed_step("Load Records")
def load_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=path)
    records: List[dict] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "records":
                    records = ast.literal_eval(node.value)
                    break
        if records:
            break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


# ---------------------------------------------------------
# Ensure index exists (async)
# ---------------------------------------------------------
@timed_step("Ensure Index")
async def ensure_index(pc: PineconeAsyncio, name: str, dimension: int):
    desc: Any = None
    try:
        desc = await pc.describe_index(name)
    except Exception as e:
        if getattr(e, "status", None) not in (None, 404):
            raise

    if desc:
        current_dim = (
            desc.get("dimension") or desc.get("config", {}).get("dimension")
            if isinstance(desc, dict)
            else getattr(desc, "dimension", None)
        )
        if current_dim and current_dim != dimension:
            logger.warning(
                f"⚠️ Index '{name}' exists with dimension {current_dim}; recreating with {dimension}."
            )
            await pc.delete_index(name)
        else:
            logger.info(f"Index '{name}' already exists with correct dimension.")
            return

    logger.info(f"Creating index '{name}' with dimension {dimension} ...")
    await pc.create_index(
        name=name,
        dimension=dimension,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )


# ---------------------------------------------------------
# Embed texts (async)
# ---------------------------------------------------------
@timed_step("Embed Texts")
async def embed_texts(pc: PineconeAsyncio, texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []

    logger.info("🧠 Generating embeddings asynchronously...")
    resp = await pc.inference.embed(
        model=MODEL,
        inputs=list(texts),
        parameters={"input_type": "passage"},
    )

    data = getattr(resp, "data", None) or resp.get("data")
    if not data:
        raise RuntimeError("No embedding data returned from inference")

    vectors: List[List[float]] = []
    for i, item in enumerate(tqdm(data, desc="Embedding", unit="vec")):
        vals = getattr(item, "values", None) or item.get("values")
        if vals is None:
            raise RuntimeError(f"Missing embedding values at position {i}")
        vectors.append(list(vals))

    return vectors


# ---------------------------------------------------------
# Upsert vectors concurrently
# ---------------------------------------------------------
@timed_step("Upsert Vectors")
async def upsert_vectors(index: Any, vectors: List[dict]):
    if not vectors:
        logger.info("No vectors to upsert.")
        return

    sem = asyncio.Semaphore(UPSERT_CONCURRENCY)
    progress = tqdm(total=len(vectors), desc="Upserting", unit="vec")

    async def upsert_batch(batch: List[dict]):
        async with sem:
            await index.upsert(vectors=batch)
            progress.update(len(batch))

    tasks = [
        asyncio.create_task(upsert_batch(vectors[i : i + UPSERT_BATCH_SIZE]))
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE)
    ]

    await asyncio.gather(*tasks)
    progress.close()


# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------
async def main():
    logger.info("🚀 Starting Pinecone ingestion workflow (async)")

    async with PineconeAsyncio(api_key=API_KEY) as pc:
        # Load records (off the event loop to avoid blocking)
        records = await asyncio.to_thread(load_records, RECORDS_PATH)
        logger.info(f"📄 Loaded {len(records)} records")

        # Extract fields
        texts, ids, metadatas = [], [], []
        for i, r in enumerate(records):
            chunk_text = r.get("chunk_text")
            rec_id = r.get("_id")
            if not chunk_text:
                raise ValueError(f"Record {i} missing 'chunk_text'")
            if not rec_id:
                raise ValueError(f"Record {i} missing '_id'")
            texts.append(chunk_text)
            ids.append(rec_id)
            metadatas.append({k: v for k, v in r.items() if k not in ("_id", "chunk_text")})

        # Embed
        embeddings = await embed_texts(pc, texts)
        if len(embeddings) != len(ids):
            raise RuntimeError("Embedding count does not match record count")

        # Ensure index
        dimension = len(embeddings[0])
        await ensure_index(pc, INDEX_NAME, dimension)

        # Upsert
        desc = await pc.describe_index(INDEX_NAME)
        async with pc.IndexAsyncio(host=desc.host) as index:
            vectors = [
                {"id": vid, "values": vec, "metadata": meta}
                for vid, vec, meta in zip(ids, embeddings, metadatas)
            ]

            await upsert_vectors(index, vectors)

            # Fetch one
            first_id = ids[0]
            fetch_start = time.perf_counter()
            fetch_result = await index.fetch(ids=[first_id])
            logger.info(
                f"🔍 Fetch completed in {time.perf_counter() - fetch_start:.3f}s — "
                f"Fetched record for id '{first_id}'"
            )
            print(fetch_result)

    logger.info("🎉 Workflow completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
