"""Tests for main.py — Pinecone ingestion pipeline.

Only covers synchronous functions (load_records, timed_step). Async pipeline
functions require dedicated async test infrastructure.
"""

import os
import tempfile
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# The module-level code in main.py calls load_dotenv() and checks for the API
# key immediately on import.  We patch the env *before* importing the module.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True, scope="session")
def _set_api_key():
    """Ensure PINECONE_API_KEY is set so main.py can be imported."""
    with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}):
        yield


# Re-import helpers after the env is patched (import is deferred so the
# autouse fixture above runs first).
@pytest.fixture(scope="session")
def main_module(_set_api_key):
    """Import main module with patched env."""
    import importlib

    with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}):
        mod = importlib.import_module("main")
    return mod


# ---- Fixtures -------------------------------------------------------------


@pytest.fixture()
def sample_records_file():
    """Create a temporary records file and return its path."""
    content = (
        "records = [\n"
        '    {"_id": "r1", "chunk_text": "Hello world", "category": "test"},\n'
        '    {"_id": "r2", "chunk_text": "Goodbye world", "category": "test"},\n'
        "]\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture()
def empty_records_file():
    """Create a temp file with no records variable."""
    content = "other_var = [1, 2, 3]\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


# ---- load_records ----------------------------------------------------------


class TestLoadRecords:
    def test_loads_valid_records(self, main_module, sample_records_file):
        records = main_module.load_records(sample_records_file)
        assert len(records) == 2
        assert records[0]["_id"] == "r1"
        assert records[1]["chunk_text"] == "Goodbye world"

    def test_preserves_metadata(self, main_module, sample_records_file):
        records = main_module.load_records(sample_records_file)
        assert records[0]["category"] == "test"

    def test_raises_on_missing_records(self, main_module, empty_records_file):
        with pytest.raises(ValueError, match="No records found"):
            main_module.load_records(empty_records_file)

    def test_raises_on_missing_file(self, main_module):
        with pytest.raises(FileNotFoundError):
            main_module.load_records("/tmp/nonexistent_records_file.txt")


# ---- timed_step ------------------------------------------------------------


class TestTimedStep:
    def test_returns_function_result(self, main_module):
        @main_module.timed_step("test step")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_logs_start_and_complete(self, main_module, caplog):
        @main_module.timed_step("my step")
        def noop():
            pass

        with caplog.at_level("INFO"):
            noop()

        messages = caplog.text
        assert "Starting: my step" in messages
        assert "Completed: my step" in messages
