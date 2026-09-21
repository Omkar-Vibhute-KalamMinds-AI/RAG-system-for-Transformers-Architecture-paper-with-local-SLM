"""
Slow end-to-end smoke tests (your machine + real config.yaml paths).

Why separate from integration tests:
- Integration tests use a tiny temp LanceDB and a fake embedder (seconds, always offline).
- These tests validate *your* LanceDB, E5 model folder, and optionally Gemma — multi‑GB and GPU/CPU heavy.

Default: SKIPPED (so `pytest tests/` stays fast).

Enable retrieval + embedder smoke tests:
    set RUN_SLOW_TESTS=1
    python -m pytest tests/test_e2e_slow.py -v --run-slow

Also load Gemma for one short generation (runs when model path exists; ~minutes, GPU/RAM):
    set RUN_SLOW_TESTS=1
    python -m pytest tests/test_e2e_slow.py -v --run-slow

Skip LLM only (keep LanceDB + embedder tests):
    set SKIP_E2E_LLM=1
    python -m pytest tests/test_e2e_slow.py -v --run-slow
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("RUN_SLOW_TESTS") != "1",
        reason="Set RUN_SLOW_TESTS=1 or pass --run-slow",
    ),
]

def _require_exists(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        pytest.skip(f"{label} missing on this machine: {path}")
    return path


@pytest.fixture(scope="module")
def production_config(e2e_unmock_heavy_deps):
    from config_loader import load_config

    return load_config()


class TestProductionVectorStore:
    def test_lancedb_from_config_opens_and_has_rows(self, production_config):
        """
        Confirms config vectordb.path + table match a populated index on disk.

        Why: wrong path or empty table → RAG always returns 'No relevant context'.
        """
        import lancedb

        db_path = _require_exists(production_config["vectordb"]["path"], "LanceDB directory")
        table_name = production_config["vectordb"]["table"]

        db = lancedb.connect(str(db_path))
        listed = db.list_tables()
        names = listed.tables if hasattr(listed, "tables") else list(listed)
        if table_name not in names:
            pytest.skip(f"Table '{table_name}' not in LanceDB: {names}")

        table = db.open_table(table_name)
        row_count = table.count_rows()
        assert row_count > 0, f"table '{table_name}' is empty"


class TestProductionEmbedder:
    def test_sentence_transformer_from_config_loads_and_embeds(self, production_config):
        """
        Loads the real multilingual E5 folder from config (not the test double).

        Why: catches missing config.json, corrupt weights, or sentence-transformers breakage.
        """
        embed_path = _require_exists(
            production_config["embedding model"]["local_path"],
            "Embedding model directory",
        )
        from EmbedModelLoader import EmbeddingManager

        manager = EmbeddingManager(embedmodel_path=str(embed_path))
        vectors = manager.generate_embeddings(
            ["query: What is the transformer architecture?"]
        )
        assert vectors.shape[0] == 1
        assert vectors.shape[1] > 0


class TestProductionRetrieval:
    def test_hybrid_retrieval_against_production_index(self, production_config):
        """
        Full retrieval stack on production LanceDB + real embedder (no LLM).

        Why: proves HybridSearch + RAGRetriever work on the same data the API uses.
        """
        import lancedb

        from EmbedModelLoader import EmbeddingManager
        from RetrievalSystem import RAGRetriever

        db_path = _require_exists(production_config["vectordb"]["path"], "LanceDB directory")
        embed_path = _require_exists(
            production_config["embedding model"]["local_path"],
            "Embedding model directory",
        )
        table_name = production_config["vectordb"]["table"]

        db = lancedb.connect(str(db_path))
        table = db.open_table(table_name)
        embedder = EmbeddingManager(embedmodel_path=str(embed_path))
        retriever = RAGRetriever(table, embedder)

        hits = retriever.retrieve(
            "Explain multi-head attention in the Transformer encoder",
            top_k=min(3, production_config["retriever"]["top_k"]),
        )
        assert hits, "expected at least one chunk from production index"
        assert all(h.get("content") for h in hits)
        assert all(h.get("source_file") for h in hits)


class TestProductionLLM:
    @pytest.mark.skipif(
        os.getenv("SKIP_E2E_LLM") == "1",
        reason="LLM smoke test disabled via SKIP_E2E_LLM=1",
    )
    def test_gemma_generates_short_completion(self, production_config):
        """
        One minimal generation from config llm.local_path.

        Why: validates ModelLoader + processor chat template on your hardware.
        Runs with other slow tests when the Gemma folder exists (see config.yaml).
        Set SKIP_E2E_LLM=1 to skip only this test (e.g. CI without GPU).
        """
        model_path = _require_exists(production_config["llm"]["local_path"], "LLM directory")
        from ModelLoader import load_model

        processor, tokenizer, model = load_model(str(model_path))
        messages = [{"role": "user", "content": "Reply with one word: ok"}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        assert len(text) > 0
