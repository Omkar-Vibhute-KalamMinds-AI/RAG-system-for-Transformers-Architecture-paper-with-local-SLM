"""
LLMOps integration tests.

These tests wire multiple modules together (unlike unit tests that mock neighbors).
They still avoid loading Gemma / SentenceTransformer so CI stays fast and offline.

Run:
    python -m pytest tests/test_integration.py -v
    python -m pytest tests/ -m integration -v
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.integration


class KeywordEmbeddingManager:
    """
    Deterministic embedder for integration tests.

    Why: real E5 weights are huge; we only need vectors aligned with seeded LanceDB rows
    so hybrid (BM25 + vector) retrieval returns predictable chunks.
    """
    def __init__(self, dim: int = 16):
        self.dim = dim
        self._attention = np.zeros(dim, dtype=np.float32)
        self._attention[0] = 1.0
        self._cooking = np.zeros(dim, dtype=np.float32)
        self._cooking[1] = 1.0

    def generate_embeddings(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "attention" in lower or "transformer" in lower:
                vectors.append(self._attention.copy())
            elif "recipe" in lower or "cooking" in lower:
                vectors.append(self._cooking.copy())
            else:
                vectors.append(np.full(self.dim, 0.01, dtype=np.float32))
        return np.vstack(vectors)


def _lance_table_names(db) -> list[str]:
    listed = db.list_tables()
    return listed.tables if hasattr(listed, "tables") else list(listed)

@pytest.fixture
def seeded_lance_table(tmp_path, real_lancedb_connect):
    """
    Real LanceDB table with two topical chunks + matching vectors.

    Why: proves ingestion schema, HybridSearch, and RAGRetriever work on disk, not mocks.
    """
    import lancedb

    from VectorStore import vectore_store

    dim = 16
    embedder = KeywordEmbeddingManager(dim=dim)
    rows = [
        {
            "text": "The Transformer model relies on multi-head attention instead of recurrence.",
            "source_file": "attention_paper.pdf",
            "chunk_index": 0,
            "id": "paper_chunk_0000",
        },
        {
            "text": "A simple recipe for pasta uses olive oil, garlic, and basil.",
            "source_file": "cookbook.pdf",
            "chunk_index": 0,
            "id": "cookbook_chunk_0000",
        },
    ]
    records = []
    for row in rows:
        vec = embedder.generate_embeddings([row["text"]])[0].astype(np.float32).tolist()
        records.append(
            {
                **row,
                "client_id": "integration_client",
                "doc_id": row["source_file"].replace(".pdf", ""),
                "vector": vec,
            }
        )

    db_path = tmp_path / "integration_lance"
    table_name = "integration_table"
    vectore_store(records, str(db_path), table_name, embedding_dim=dim)
    db = lancedb.connect(str(db_path))
    assert table_name in _lance_table_names(db)
    table = db.open_table(table_name)
    return table, KeywordEmbeddingManager(dim=dim)


@pytest.fixture
def integration_rag_retriever(seeded_lance_table):
    from RetrievalSystem import RAGRetriever

    table, embedder = seeded_lance_table
    return RAGRetriever(table, embedder)


@pytest.fixture
def integration_pipeline(integration_rag_retriever):
    """Pipeline with real retrieval + fake LLM (no GPU)."""
    from app import AdvancedRAGPipeline

    model = MagicMock()
    model.device = "cpu"
    processor = MagicMock()
    processor.tokenizer.eos_token_id = 0
    processor.apply_chat_template.return_value = {"input_ids": MagicMock(shape=[1, 8])}
    processor.tokenizer.decode.return_value = "Integrated answer from mock LLM."

    pipe = AdvancedRAGPipeline(
        integration_rag_retriever,
        model,
        processor,
        use_history=False,
        history_window=1,
    )
    pipe.default_stream = False
    # Bypass Gemma chat template / GPU generate; retrieval is what we integrate here.
    pipe.generate = MagicMock(return_value="Integrated answer from mock LLM.")
    return pipe


class TestRetrievalIntegration:
    def test_hybrid_search_returns_attention_chunk_for_ml_query(self, integration_rag_retriever):
        """
        End-to-end retrieval path: LanceDB → HybridSearch → ranked dicts.

        Guards against schema drift and broken ensemble wiring.
        """
        hits = integration_rag_retriever.retrieve(
            "Explain multi-head attention in Transformers",
            top_k=2,
        )
        assert hits, "expected at least one retrieved chunk"
        combined = " ".join(h["content"].lower() for h in hits)
        assert "attention" in combined or "transformer" in combined
        assert any(h["source_file"] == "attention_paper.pdf" for h in hits)

    def test_retrieval_prefers_on_topic_over_off_topic(self, integration_rag_retriever):
        """At least one top result should come from the ML paper, not only the cookbook."""
        hits = integration_rag_retriever.retrieve(
            "multi-head attention instead of recurrence",
            top_k=2,
        )
        sources = {h["source_file"] for h in hits}
        assert "attention_paper.pdf" in sources


class TestPipelineIntegration:
    def test_query_variations_merged_into_retrieval(self, integration_pipeline):
        """
        Pipeline should search original question + accepted variations (deduped).

        We stub variations to a paraphrase so _collect_hits runs multiple retrieve calls.
        """
        integration_pipeline.retriever.retrieve = MagicMock(
            return_value=[
                {
                    "id": "paper_chunk_0000",
                    "chunk_index": 0,
                    "content": "attention text",
                    "source_file": "attention_paper.pdf",
                    "rank": 1,
                }
            ]
        )
        with patch("app.query_variations", return_value=["transformer attention layers"]):
            integration_pipeline.query("What is attention?", stream=False)

        assert integration_pipeline.retriever.retrieve.call_count >= 2

    def test_full_query_returns_citations_from_real_retriever(self, integration_pipeline):
        """RAG answer must include citation block when LanceDB returns context."""
        with patch("app.query_variations", return_value=[]):
            result = integration_pipeline.query(
                "How does multi-head attention work?",
                stream=False,
            )
        assert "Citations:" in result["answer"]
        assert result["sources"]
        assert any(s["source"] == "attention_paper.pdf" for s in result["sources"])
        assert "Integrated answer" in result["answer"]


class TestApiIntegration:
    def test_query_http_roundtrip_with_loaded_pipeline(self, integration_pipeline):
        """
        HTTP layer + Pydantic + pipeline.query (same path as Retrieval Console Run button).

        Why: unit tests mock pipeline; this verifies modelapi_app wiring and JSON response shape.
        """
        from fastapi.testclient import TestClient
        import modelapi_app as api

        api.pipeline = integration_pipeline
        api.model = MagicMock(device="cpu")
        api.processor = MagicMock()
        api.tokenizer = MagicMock()

        with patch("app.query_variations", return_value=[]):
            client = TestClient(api.app)
            resp = client.post(
                "/query",
                json={
                    "question": "Describe transformer attention",
                    "top_k": 2,
                    "min_score": 0.0,
                    "use_history": False,
                    "summarize": False,
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "Describe transformer attention"
        assert body["sources"]
        assert "elapsed_seconds" in body
        assert isinstance(body["elapsed_seconds"], (int, float))
        assert any(s["source"] == "attention_paper.pdf" for s in body["sources"])

    def test_generate_stream_http_when_pipeline_loaded(self, integration_pipeline):
        """Streaming endpoint should return plain text body when pipeline is ready."""
        from fastapi.testclient import TestClient

        import modelapi_app as api

        api.pipeline = integration_pipeline
        api.model = MagicMock(device="cpu")
        api.processor = MagicMock()
        api.tokenizer = MagicMock()

        integration_pipeline.default_stream = True

        with patch("app.query_variations", return_value=[]), patch.object(
            integration_pipeline,
            "generate_streaming",
            return_value=iter(["stream", "ed"]),
        ):
            client = TestClient(api.app)
            resp = client.post(
                "/query/stream",
                json={"question": "attention", "top_k": 1, "use_history": False},
            )

        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        assert "streamed" in resp.text
        assert "Citations:" in resp.text

    def test_health_reports_ready_when_pipeline_injected(self, integration_pipeline):
        """Health should flip to ready once lifespan (or tests) attach a pipeline."""
        from fastapi.testclient import TestClient

        import modelapi_app as api

        api.pipeline = integration_pipeline
        api.model = MagicMock(device="cpu")

        client = TestClient(api.app)
        body = client.get("/health").json()
        assert body["pipeline"] == "ready"
        assert body["status"] == "ok"


class TestConfigRetrievalIntegration:
    def test_config_paths_match_retriever_expectations(self):
        """
        Integration sanity: config keys used at import time must stay in sync.

        Catches renamed YAML keys before production LanceDB connect fails.
        """
        from config_loader import load_config

        cfg = load_config()
        # Paths in config.yaml may be Windows-style; do not use Path.anchor
        # (empty on Linux CI for D:\... strings).
        assert isinstance(cfg["vectordb"]["path"], str) and cfg["vectordb"]["path"].strip()
        assert cfg["vectordb"]["table"]
        assert cfg["retriever"]["top_k"] >= 1
        min_score = float(cfg["retriever"]["min_score"])
        assert 0.0 <= min_score <= 1.0
