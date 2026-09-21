"""
LLMOps project unit tests (single suite file).

Design goals (why tests are structured this way):
1. Fast & offline — no Gemma weights, no real LanceDB paths from config.yaml in CI.
2. One file — easy to discover; sections mirror `pyfiles/` modules.
3. Comments — each test states *what* is guarded and *why* it matters for RAG/API behavior.
Run from repo root:
    python -m pytest tests/ -v
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
# ---------------------------------------------------------------------------
# config_loader — central YAML contract for LLM, embedder, retriever, LanceDB
# ---------------------------------------------------------------------------

class TestConfigLoader:
    def test_load_config_reads_project_yaml(self):
        """Production code assumes config.yaml exists at repo root; break early if layout changes."""
        from config_loader import load_config

        cfg = load_config()
        assert "llm" in cfg
        assert "embedding model" in cfg
        assert "vectordb" in cfg
        assert "retriever" in cfg
        assert "AdvancedRAGPipeline" in cfg

    def test_load_config_missing_file_raises(self, tmp_path, monkeypatch):
        """Misplaced config should fail loudly instead of silently using wrong models."""
        from config_loader import load_config

        monkeypatch.setattr("config_loader._project_root", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_config("nope.yaml")

# ---------------------------------------------------------------------------
# exception — operational errors must carry file/line for log triage
# ---------------------------------------------------------------------------
import sys 
sys.path.append(r'D:\LLMOps\pyfiles')

class TestExceptions:
    def test_custom_exception_includes_location(self):
        """Ops team relies on CustomException formatting in logs (see exception.py)."""
        from exception import CustomException

        with patch("exception.logger") as mock_logger:
            mock_logger.error = MagicMock()
            try:
                raise ValueError("unit-test failure")
            except ValueError as err:
                wrapped = CustomException(err, sys)
                text = str(wrapped)
                assert "unit-test failure" in text
                assert "Line No" in text
                assert "Script" in text

    def test_embed_model_loader_exception_prefix(self):
        """Embed failures should be identifiable in log grep (`EmbedModel` tag)."""
        from exception import EmbedModelLoaderException

        with patch("exception.logger") as mock_logger:
            mock_logger.error = MagicMock()
            try:
                raise RuntimeError("cuda oom")
            except RuntimeError as err:
                wrapped = EmbedModelLoaderException(err, sys)
                assert "EmbedModel" in str(wrapped)
# ---------------------------------------------------------------------------
# query_variationar — query expansion must parse LLM output safely
# ---------------------------------------------------------------------------
class TestQueryVariations:
    def test_query_variation_generator_parses_python_list(self):
        """LLM is asked for a Python list; parser must accept valid list literals only."""
        import query_variationar as qv

        with patch.object(qv, "_generate_text", return_value="['attention mechanism', 'self attention']"):
            out = qv.query_veriation_generator("What is attention?")
        assert out == ["attention mechanism", "self attention"]

    def test_query_variation_generator_fallback_lines_when_not_a_list(self):
        """If the model ignores instructions, we still get usable strings from line breaks."""
        import query_variationar as qv

        fake = "Here you go:\n- first variant\n- second variant"
        with patch.object(qv, "_generate_text", return_value=fake):
            out = qv.query_veriation_generator("query")
        assert any("first variant" in line for line in out)

    def test_query_variations_filters_by_embedding_similarity(self):
        """Variations too far from the original query should not pollute hybrid retrieval."""
        import query_variationar as qv

        with patch.object(
            qv,
            "query_veriation_generator",
            return_value=["close paraphrase", "unrelated topic"],
        ), patch.object(qv, "_get_embedding_manager") as get_emb:
            get_emb.return_value.generate_embeddings.return_value = np.array(
                [
                    [1.0, 0.0],
                    [0.99, 0.01],
                    [0.0, 1.0],
                ]
            )
            accepted = qv.query_variations("original question", min_sim=0.85)

        assert accepted == ["close paraphrase"]

    def test_query_variations_returns_empty_on_internal_error(self):
        """API/RAG path must not crash if variation step fails; pipeline continues with base query."""
        import query_variationar as qv

        with patch.object(qv, "query_veriation_generator", side_effect=RuntimeError("llm down")):
            assert qv.query_variations("hello") == []


# ---------------------------------------------------------------------------
# HybridSearch — vector retriever maps LanceDB rows → LangChain Documents
# ---------------------------------------------------------------------------
class TestHybridSearch:
    def test_lancedb_direct_retriever_maps_rows_to_documents(self, sample_chunk_record):
        """Ensemble retrieval depends on consistent metadata keys (id, chunk_index, source_file)."""
        from HybridSearch import LanceDBDirectRetriever

        row_df = pd.DataFrame([sample_chunk_record])
        mock_table = MagicMock()
        chain = mock_table.search.return_value.nprobes.return_value.limit.return_value
        chain.to_pandas.return_value = row_df

        mock_emb = MagicMock()
        mock_emb.generate_embeddings.return_value = np.array([[0.1, 0.2, 0.3]])

        retriever = LanceDBDirectRetriever(
            table=mock_table, embedding_manager=mock_emb, top_k=1
        )
        docs = retriever._get_relevant_documents("transformer attention")

        assert len(docs) == 1
        assert docs[0].page_content == sample_chunk_record["text"]
        assert docs[0].metadata["id"] == sample_chunk_record["id"]
        assert docs[0].metadata["source_file"] == sample_chunk_record["source_file"]


class TestLocalBM25Retriever:
    def test_bm25_finds_lexical_match_without_langchain_community(self):
        """rank_bm25 + BaseRetriever replaces deprecated community BM25Retriever."""
        from langchain_core.documents import Document

        from bm25_retriever import LocalBM25Retriever

        docs = [
            Document(page_content="multi-head attention in transformers", metadata={"id": "1"}),
            Document(page_content="pasta recipe with garlic", metadata={"id": "2"}),
            Document(page_content="weather forecast sunny skies", metadata={"id": "3"}),
        ]
        retriever = LocalBM25Retriever.from_documents(docs, k=1)
        hits = retriever.invoke("multi-head attention")
        assert hits[0].metadata["id"] == "1"


# ---------------------------------------------------------------------------
# RetrievalSystem — thin wrapper over hybrid search; shapes API-facing dicts
# ---------------------------------------------------------------------------
class TestRetrievalSystem:
    def test_retrieve_returns_ranked_dicts(self, sample_chunk_record):
        """`app.AdvancedRAGPipeline` expects dict hits with content + citation fields."""
        from langchain_core.documents import Document

        from RetrievalSystem import RAGRetriever

        doc = Document(
            page_content=sample_chunk_record["text"],
            metadata={
                "id": sample_chunk_record["id"],
                "chunk_index": sample_chunk_record["chunk_index"],
                "source_file": sample_chunk_record["source_file"],
            },
        )
        mock_emb = MagicMock()
        mock_table = MagicMock()

        with patch("RetrievalSystem.Hybrid_search", return_value=[doc]):
            retriever = RAGRetriever(mock_table, mock_emb)
            hits = retriever.retrieve("attention", top_k=1)

        assert hits[0]["rank"] == 1
        assert hits[0]["content"] == sample_chunk_record["text"]
        assert hits[0]["source_file"] == sample_chunk_record["source_file"]

# ---------------------------------------------------------------------------
# VectorStore — ingestion writes schema expected by HybridSearch / LanceDB
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_vectore_store_creates_table_with_expected_schema(self, tmp_path, real_lancedb_connect):
        """Wrong schema breaks vector search column `vector` and metadata joins."""
        import lancedb 
        from config_loader import load_config
        cfg = load_config() 
        from VectorStore import vectore_store

        dim = 8
        vector = np.zeros(dim, dtype=np.float32).tolist()
        records = [
            {
                "id": "doc_chunk_0000",
                "client_id": "client_a",
                "doc_id": "doc",
                "source_file": "paper.pdf",
                "chunk_index": 0,
                "text": "chunk text",
                "vector": vector,
            }
        ]
        table_name = cfg["vectordb"]["table"]
        db_path = tmp_path / "lance_unit"

        table = vectore_store(records, str(db_path), table_name, embedding_dim=dim)
        assert table is not None
        listed = lancedb.connect(str(db_path)).list_tables()
        table_names = listed.tables if hasattr(listed, "tables") else listed
        assert table_name in table_names

# ---------------------------------------------------------------------------
# app.AdvancedRAGPipeline — RAG orchestration (mocked LLM + retriever)
# ---------------------------------------------------------------------------

class TestAdvancedRAGPipeline:
    @pytest.fixture
    def pipeline(self):
        """Build pipeline without GPU; retriever/LLM are mocks."""
        from app import AdvancedRAGPipeline

        retriever = MagicMock()
        model = MagicMock()
        model.device = "cpu"
        processor = MagicMock()
        processor.tokenizer.eos_token_id = 0
        processor.apply_chat_template.return_value = {
            "input_ids": MagicMock(shape=[1, 4]),
        }
        processor.tokenizer.decode.return_value = "mock answer"

        pipe = AdvancedRAGPipeline(
            retriever, model, processor, use_history=False, history_window=1
        )
        pipe.default_stream = False
        return pipe

    def test_collect_hits_deduplicates_by_document_id(self, pipeline):
        """Query variations can retrieve the same chunk twice; dedupe keeps citations clean."""
        hit_a = {
            "id": "same_id",
            "chunk_index": 0,
            "content": "A",
            "source_file": "f.pdf",
            "rank": 1,
        }
        hit_b = {
            "id": "other_id",
            "chunk_index": 1,
            "content": "B",
            "source_file": "f.pdf",
            "rank": 1,
        }
        pipeline.retriever.retrieve.side_effect = [
            [hit_a, hit_b],
            [hit_a],
        ]

        merged = pipeline._collect_hits(["q1", "q2"], top_k=2)
        ids = [h["id"] for h in merged]
        assert ids.count("same_id") == 1
        assert "other_id" in ids

    def test_query_without_hits_returns_standard_message(self, pipeline):
        """Empty retrieval should short-circuit generation (saves LLM latency/cost)."""
        pipeline.retriever.retrieve.return_value = []
        with patch("app.query_variations", return_value=[]):
            result = pipeline.query("unknown topic?", stream=False)

        assert result["answer"] == "No relevant context found"
        assert result["sources"] == []

    def test_query_with_hits_adds_citations_block(self, pipeline):
        """Frontend and API consumers parse citation footer from answer string."""
        pipeline.retriever.retrieve.return_value = [
            {
                "id": "x",
                "chunk_index": 0,
                "content": "Context paragraph",
                "source_file": "paper.pdf",
                "rank": 1,
            }
        ]
        pipeline.generate = MagicMock(return_value="Answer body")

        with patch("app.query_variations", return_value=[]):
            result = pipeline.query("question?", stream=False)

        assert "Citations:" in result["answer"]
        assert "paper.pdf" in result["answer"]
# ---------------------------------------------------------------------------
# modelapi_app — FastAPI contract (health, config, validation, 503 when unloaded)
# ---------------------------------------------------------------------------
class TestModelApi:
    def test_health_before_model_load(self):
        """UI polls /health while weights load; must respond without raising."""
        from fastapi.testclient import TestClient

        import modelapi_app as api

        api.pipeline = None
        api.model = None
        client = TestClient(api.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "loading"
        assert body["pipeline"] == "unloaded"

    def test_public_config_exposes_defaults(self):
        """Retrieval Console sliders call /api/config on page load."""
        from fastapi.testclient import TestClient

        import modelapi_app as api

        client = TestClient(api.app)
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "temperature" in data["llm"]
        assert "top_k" in data["retriever"]

    def test_query_returns_503_when_pipeline_not_ready(self):
        """Prevent silent failures: client gets explicit service-unavailable."""
        from fastapi.testclient import TestClient

        import modelapi_app as api

        api.pipeline = None
        client = TestClient(api.app)
        resp = client.post("/query", json={"question": "What is transformer in macine learning?"})
        assert resp.status_code == 503

    def test_generate_request_validation_rejects_empty_prompt(self):
        """Pydantic guards stop empty prompts before they hit the GPU stack."""
        from pydantic import ValidationError

        import modelapi_app as api 

        with pytest.raises(ValidationError):
            api.GenerateRequest(prompt="")

    def test_index_route_serves_frontend(self):
        """Root URL must serve static Retrieval Console for local demos."""
        from fastapi.testclient import TestClient

        import modelapi_app as api

        client = TestClient(api.app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
