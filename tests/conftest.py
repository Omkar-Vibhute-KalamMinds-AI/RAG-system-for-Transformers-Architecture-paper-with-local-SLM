"""
Pytest bootstrap for LLMOps.

Why this file exists:
- `pyfiles/app.py` connects to LanceDB and loads SentenceTransformer at import time.
  Unit tests must not pull multi-GB weights or require a live vector DB on every run.
- We patch those dependencies once, before any test module imports `app` or `modelapi_app`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYFILES_DIR = PROJECT_ROOT / "pyfiles"

if str(PYFILES_DIR) not in sys.path:
    sys.path.insert(0, str(PYFILES_DIR))

_mock_table = MagicMock(name="lancedb_table")
_mock_table.count_rows.return_value = 0
_mock_table.to_pandas.return_value = __import__("pandas").DataFrame(
    columns=["id", "chunk_index", "source_file", "text"]
)

_mock_db = MagicMock(name="lancedb_connection")
_mock_db.open_table.return_value = _mock_table

_lance_patch = patch("lancedb.connect", return_value=_mock_db)
_lance_patch.start()

_embed_patch = patch(
    "EmbedModelLoader.EmbeddingManager",
    return_value=MagicMock(name="embedding_manager"),
)
_embed_patch.start()

import pytest  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow on-machine E2E smoke tests (LanceDB, embedder, optional LLM)",
    )


def pytest_configure(config):
    import os

    if config.getoption("--run-slow", default=False):
        os.environ["RUN_SLOW_TESTS"] = "1"


def _slow_tests_enabled() -> bool:
    import os

    return os.getenv("RUN_SLOW_TESTS") == "1"


@pytest.fixture(scope="module")
def e2e_unmock_heavy_deps():
    """
    Turn off global LanceDB / embedder mocks for slow smoke tests.

    Why module scope: loading SentenceTransformer once per file is expensive enough.
    """
    if not _slow_tests_enabled():
        pytest.skip("Slow E2E disabled. Run: RUN_SLOW_TESTS=1 pytest --run-slow tests/test_e2e_slow.py")
    _lance_patch.stop()
    _embed_patch.stop()
    yield
    _lance_patch.start()
    _embed_patch.start()


@pytest.fixture
def sample_chunk_record():
    """Minimal LanceDB row shape used across retrieval tests."""
    return {
        "id": "doc_chunk_0001",
        "client_id": "test_client",
        "doc_id": "doc",
        "source_file": "sample.pdf",
        "chunk_index": 1,
        "text": "The transformer uses multi-head attention.",
    }


@pytest.fixture
def real_lancedb_connect():
    """Temporarily use real LanceDB (integration / vector-store tests)."""
    _lance_patch.stop()
    yield
    _lance_patch.start()
