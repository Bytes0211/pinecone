"""Async Pinecone vector database ingestion pipeline.

Embeds text records using Pinecone's hosted inference API (llama-text-embed-v2)
and upserts them into a serverless index. Serves as a minimal RAG ingestion
backbone.

Pipeline steps:
    1. Load records from a Python-formatted file via AST parsing.
    2. Embed text fields in batches through Pinecone inference.
    3. Ensure a serverless index exists with the correct dimension.
    4. Upsert vectors concurrently with batching and retry logic.
    5. Fetch one vector to verify the ingestion.

Author:
    scotton
"""

import ast
import asyncio
import functools
import inspect
import logging
import os
import random
import time
from logging.handlers import RotatingFileHandler
from typing import Any, List, Sequence

from colorlog import ColoredFormatter
from dotenv import load_dotenv
from pinecone import PineconeAsyncio, ServerlessSpec
from tqdm import tqdm

# ---------------------------------------------------------
# Logging Setup: Colored Logs + Timestamp + Level Coloring
# ---------------------------------------------------------

# File logging config
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3

os.makedirs(LOG_DIR, exist_ok=True)

# Console handler — colored, INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
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

# File handler — plain text, DEBUG level, rotating
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

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
EMBED_BATCH_SIZE = 32
MAX_RETRIES = 3
BACKOFF_BASE = 0.5
BACKOFF_JITTER = 0.3


async def run_with_retries(label: str, operation, max_retries: int = MAX_RETRIES):
    """Execute an async operation with exponential backoff retries.

    Retries the given async callable up to ``max_retries`` times. On each
    failure the delay doubles from ``BACKOFF_BASE`` with random jitter
    added via ``BACKOFF_JITTER``.

    Args:
        label: A human-readable name for the operation, used in log messages.
        operation: A zero-argument async callable to execute. Typically a
            lambda wrapping the real call so arguments are captured.
        max_retries: Maximum number of attempts before raising. Defaults
            to the module-level ``MAX_RETRIES`` constant.

    Returns:
        The return value of ``operation()`` on the first successful attempt.

    Raises:
        Exception: Re-raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return await operation()
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            delay = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(
                0, BACKOFF_JITTER
            )
            logger.warning(
                f"Retrying {label} (attempt {attempt}/{max_retries}) after error: {exc!r}. "
                f"Sleeping {delay:.2f}s"
            )
            await asyncio.sleep(delay)
    if last_exc:
        raise last_exc


# ---------------------------------------------------------
# Utility: timing decorator (supports sync + async)
# ---------------------------------------------------------
def timed_step(step_name: str):
    """Decorator factory that logs the execution time of a pipeline step.

    Automatically detects whether the wrapped function is synchronous or
    asynchronous and applies the appropriate wrapper. Logs a start message
    before execution and a completion message (with elapsed seconds) after.

    Args:
        step_name: A descriptive label for the step, shown in log output.

    Returns:
        A decorator that wraps the target function with timing and logging.
    """

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
    """Load records from a Python-formatted file using AST parsing.

    Reads the file at ``path``, parses it as a Python module, and extracts
    the first top-level variable named ``records``. The value is safely
    evaluated with ``ast.literal_eval``, so only literal Python structures
    (lists, dicts, strings, numbers, etc.) are allowed.

    Args:
        path: Filesystem path to the records file (e.g. ``records.txt``).

    Returns:
        A list of dicts, each containing at minimum ``_id`` and
        ``chunk_text`` keys.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        SyntaxError: If the file is not valid Python syntax.
        ValueError: If no ``records`` variable is found in the file.
    """
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
    """Ensure a Pinecone serverless index exists with the required dimension.

    Checks whether an index with the given ``name`` already exists. If it
    exists with a mismatched dimension, the index is deleted and recreated.
    If it exists with the correct dimension, the function returns immediately.
    If no index exists, a new one is created using the module-level
    ``METRIC``, ``CLOUD``, and ``REGION`` constants.

    Args:
        pc: An initialised ``PineconeAsyncio`` client instance.
        name: The name of the index to create or verify.
        dimension: The required vector dimension for the index.

    Raises:
        Exception: Any Pinecone API error other than a 404 (not found) when
            describing the index.
    """
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
    """Generate vector embeddings for a sequence of text strings.

    Texts are processed in batches of ``EMBED_BATCH_SIZE`` using the
    Pinecone hosted inference endpoint (model configured by the ``MODEL``
    constant). Each batch call is wrapped in ``run_with_retries`` for
    resilience. A ``tqdm`` progress bar tracks per-vector progress.

    Args:
        pc: An initialised ``PineconeAsyncio`` client instance.
        texts: The text strings to embed. Each string is treated as a
            "passage" input type for the embedding model.

    Returns:
        A list of embedding vectors (each a list of floats), in the same
        order as the input ``texts``.

    Raises:
        RuntimeError: If the inference API returns no data or if any
            individual embedding is missing its ``values`` field.
    """
    if not texts:
        return []

    logger.info("🧠 Generating embeddings asynchronously...")
    progress = tqdm(total=len(texts), desc="Embedding", unit="vec")

    vectors: List[List[float]] = []
    for offset in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[offset : offset + EMBED_BATCH_SIZE])
        resp = await run_with_retries(
            "embed",
            lambda b=batch: pc.inference.embed(
                model=MODEL,
                inputs=b,
                parameters={"input_type": "passage"},
            ),
        )

        data = getattr(resp, "data", None) or resp.get("data")
        if not data:
            raise RuntimeError("No embedding data returned from inference")

        for j, item in enumerate(data):
            vals = getattr(item, "values", None) or item.get("values")
            if vals is None:
                raise RuntimeError(f"Missing embedding values at position {offset + j}")
            vectors.append(list(vals))
            progress.update(1)

    progress.close()
    return vectors


# ---------------------------------------------------------
# Upsert vectors concurrently
# ---------------------------------------------------------
@timed_step("Upsert Vectors")
async def upsert_vectors(index: Any, vectors: List[dict]):
    """Upsert vectors into a Pinecone index with concurrent batching.

    Splits ``vectors`` into chunks of ``UPSERT_BATCH_SIZE`` and upserts
    them concurrently, limiting parallelism with an ``asyncio.Semaphore``
    set to ``UPSERT_CONCURRENCY``. Each batch is retried on failure via
    ``run_with_retries``. A ``tqdm`` progress bar tracks per-vector progress.

    Args:
        index: A connected Pinecone ``IndexAsyncio`` instance.
        vectors: A list of vector dicts, each containing ``id`` (str),
            ``values`` (list of floats), and ``metadata`` (dict) keys.

    Raises:
        Exception: Any Pinecone API error that persists after all retries
            in a batch upsert call.
    """
    if not vectors:
        logger.info("No vectors to upsert.")
        return

    sem = asyncio.Semaphore(UPSERT_CONCURRENCY)
    progress = tqdm(total=len(vectors), desc="Upserting", unit="vec")

    async def upsert_batch(batch: List[dict]):
        async with sem:
            await run_with_retries(
                "upsert_batch", lambda b=batch: index.upsert(vectors=b)
            )
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
    """Run the full Pinecone ingestion workflow.

    Orchestrates the end-to-end pipeline:
        1. Initialises a ``PineconeAsyncio`` client via async context manager.
        2. Loads records from ``RECORDS_PATH`` on a background thread.
        3. Validates that every record has ``_id`` and ``chunk_text`` fields.
        4. Embeds all ``chunk_text`` values in batches.
        5. Ensures the target index exists with the correct dimension.
        6. Connects to the index and upserts all vectors concurrently.
        7. Fetches the first vector back as a verification step.

    Raises:
        RuntimeError: If ``PINECONE_API_KEY`` is not set, embedding counts
            do not match record counts, or the inference API returns no data.
        ValueError: If any record is missing ``_id`` or ``chunk_text``.
    """
    logger.info("🚀 Starting Pinecone ingestion workflow (async)")

    async with PineconeAsyncio(api_key=API_KEY) as pc:
        # Load records (off the event loop to avoid blocking)
        records = await asyncio.to_thread(load_records, RECORDS_PATH)
        logger.info(f"📄 Loaded {len(records)} records")

        # Extract fields
        texts, ids, metadatas = [], [], []
        ir_check = 0
        for i, r in enumerate(records):
            if ir_check < 4:
                print(f"i = {i}\nr = {r}\n\n")
                ir_check += 1
            chunk_text = r.get("chunk_text")
            rec_id = r.get("_id")
            if not chunk_text:
                raise ValueError(f"Record {i} missing 'chunk_text'")
            if not rec_id:
                raise ValueError(f"Record {i} missing '_id'")
            texts.append(chunk_text)
            ids.append(rec_id)
            metadatas.append(
                {k: v for k, v in r.items() if k not in ("_id", "chunk_text")}
            )

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
