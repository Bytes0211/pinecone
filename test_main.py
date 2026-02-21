"""Tests for main.py — Pinecone ingestion pipeline (sync version)."""

import ast
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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


class DualAccess:
    """Mock object supporting both attribute and dict-style (.get) access."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def get(self, key, default=None):
        return self.__dict__.get(key, default)


# ---- Fixtures -------------------------------------------------------------

@pytest.fixture()
def sample_records_file():
    """Create a temporary records file and return its path."""
    content = (
        'records = [\n'
        '    {"_id": "r1", "chunk_text": "Hello world", "category": "test"},\n'
        '    {"_id": "r2", "chunk_text": "Goodbye world", "category": "test"},\n'
        ']\n'
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


@pytest.fixture()
def mock_pc():
    """Return a MagicMock standing in for a Pinecone client."""
    pc = MagicMock()
    return pc


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


# ---- ensure_index ----------------------------------------------------------

class TestEnsureIndex:
    def test_creates_index_when_not_found(self, main_module, mock_pc):
        exc = Exception("not found")
        exc.status = 404
        mock_pc.describe_index.side_effect = exc

        main_module.ensure_index(mock_pc, "test-idx", 128)

        mock_pc.create_index.assert_called_once()
        kwargs = mock_pc.create_index.call_args
        assert kwargs[1]["name"] == "test-idx"
        assert kwargs[1]["dimension"] == 128

    def test_skips_when_dimension_matches(self, main_module, mock_pc):
        mock_pc.describe_index.return_value = SimpleNamespace(dimension=128)

        main_module.ensure_index(mock_pc, "test-idx", 128)

        mock_pc.create_index.assert_not_called()
        mock_pc.delete_index.assert_not_called()

    def test_recreates_when_dimension_mismatches(self, main_module, mock_pc):
        mock_pc.describe_index.return_value = SimpleNamespace(dimension=64)

        main_module.ensure_index(mock_pc, "test-idx", 128)

        mock_pc.delete_index.assert_called_once_with("test-idx")
        mock_pc.create_index.assert_called_once()

    def test_propagates_unexpected_errors(self, main_module, mock_pc):
        exc = Exception("server error")
        exc.status = 500
        mock_pc.describe_index.side_effect = exc

        with pytest.raises(Exception, match="server error"):
            main_module.ensure_index(mock_pc, "test-idx", 128)


# ---- embed_texts -----------------------------------------------------------

class TestEmbedTexts:
    def test_returns_empty_for_no_texts(self, main_module, mock_pc):
        result = main_module.embed_texts(mock_pc, [])
        assert result == []
        mock_pc.inference.embed.assert_not_called()

    def test_returns_vectors(self, main_module, mock_pc):
        mock_pc.inference.embed.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(values=[0.1, 0.2, 0.3]),
                SimpleNamespace(values=[0.4, 0.5, 0.6]),
            ]
        )

        result = main_module.embed_texts(mock_pc, ["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    def test_raises_on_empty_response(self, main_module, mock_pc):
        mock_pc.inference.embed.return_value = DualAccess(data=None)

        with pytest.raises(RuntimeError, match="No embedding data"):
            main_module.embed_texts(mock_pc, ["hello"])

    def test_raises_on_missing_values(self, main_module, mock_pc):
        mock_pc.inference.embed.return_value = SimpleNamespace(
            data=[DualAccess(values=None)]
        )

        with pytest.raises(RuntimeError, match="Missing embedding values"):
            main_module.embed_texts(mock_pc, ["hello"])

    def test_calls_embed_with_correct_params(self, main_module, mock_pc):
        mock_pc.inference.embed.return_value = SimpleNamespace(
            data=[SimpleNamespace(values=[0.1])]
        )

        main_module.embed_texts(mock_pc, ["test text"])

        mock_pc.inference.embed.assert_called_once_with(
            model=main_module.MODEL,
            inputs=["test text"],
            parameters={"input_type": "passage"},
        )


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


# ---- main workflow (integration) -------------------------------------------

class TestMainWorkflow:
    @patch("main.Pinecone")
    @patch("main.load_records")
    def test_full_pipeline(self, mock_load, mock_pinecone_cls, main_module):
        # Setup records
        mock_load.return_value = [
            {"_id": "r1", "chunk_text": "Hello", "cat": "a"},
            {"_id": "r2", "chunk_text": "World", "cat": "b"},
        ]

        # Setup Pinecone client mock
        mock_pc = MagicMock()
        mock_pinecone_cls.return_value = mock_pc

        # Embedding response
        mock_pc.inference.embed.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(values=[0.1, 0.2]),
                SimpleNamespace(values=[0.3, 0.4]),
            ]
        )

        # describe_index for ensure_index (404 → create)
        exc = Exception("not found")
        exc.status = 404
        mock_pc.describe_index.side_effect = exc

        # Index mock
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.fetch.return_value = {"vectors": {"r1": {}}}

        main_module.main()

        # Verify pipeline steps executed
        mock_load.assert_called_once()
        mock_pc.inference.embed.assert_called_once()
        mock_pc.create_index.assert_called_once()
        assert mock_index.upsert.call_count == 2
        mock_index.fetch.assert_called_once_with(ids=["r1"])

    @patch("main.Pinecone")
    @patch("main.load_records")
    def test_raises_on_missing_chunk_text(self, mock_load, mock_pinecone_cls, main_module):
        mock_load.return_value = [{"_id": "r1"}]
        mock_pinecone_cls.return_value = MagicMock()

        with pytest.raises(ValueError, match="missing 'chunk_text'"):
            main_module.main()

    @patch("main.Pinecone")
    @patch("main.load_records")
    def test_raises_on_missing_id(self, mock_load, mock_pinecone_cls, main_module):
        mock_load.return_value = [{"chunk_text": "hello"}]
        mock_pinecone_cls.return_value = MagicMock()

        with pytest.raises(ValueError, match="missing '_id'"):
            main_module.main()
