from typing import List

import numpy as np
import pandas as pd

import pyfiles_path  # noqa: F401
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from bm25_retriever import LocalBM25Retriever

_SMALL_TABLE_ROWS = 64


def _cosine_search_pandas(table, query_embedding: np.ndarray, top_k: int):
    rows = table.to_pandas()
    if rows.empty:
        return rows

    q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    q_norm = float(np.linalg.norm(q)) or 1.0
    scored: list[tuple[float, object]] = []
    for idx, row in rows.iterrows():
        vec = np.asarray(row["vector"], dtype=np.float32).reshape(-1)
        v_norm = float(np.linalg.norm(vec)) or 1.0
        sim = float(np.dot(q, vec) / (q_norm * v_norm))
        scored.append((sim, idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_indices = [idx for _, idx in scored[:top_k]]
    return pd.DataFrame(rows.loc[top_indices]).reset_index(drop=True)


def _vector_search_dataframe(table, query_embedding: np.ndarray, top_k: int):
    """Native Lance search, with cosine fallback for tiny/unindexed tables (Linux CI)."""
    try:
        n_rows = table.count_rows()
    except Exception:
        n_rows = None

    if isinstance(n_rows, int) and n_rows <= _SMALL_TABLE_ROWS:
        return _cosine_search_pandas(table, query_embedding, top_k)

    try:
        return (
            table.search(query_embedding, vector_column_name="vector")
            .limit(top_k)
            .to_pandas()
        )
    except Exception:
        return _cosine_search_pandas(table, query_embedding, top_k)




class LanceDBDirectRetriever(BaseRetriever):
    table: object = Field(...)
    embedding_manager: object = Field(...)
    top_k: int = 1

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = np.asarray(
            self.embedding_manager.generate_embeddings([query])[0],
            dtype=np.float32,
        )
        results = _vector_search_dataframe(self.table, query_embedding, self.top_k)

        docs = []
        for _, row in results.iterrows():
          
            docs.append(
                Document(
                    page_content=row["text"] if "text" in row else row.get("page_content", ""),
                    metadata={
                        "id": row["id"],
                        "chunk_index": row["chunk_index"],
                        "source_file": row["source_file"],

                    },
                )
            )
        return docs


def _bm25_documents_from_table(table):
    all_rows = table.to_pandas()
    return [
        Document(
            page_content=row["text"],
            metadata={
                "id": row["id"],
                "chunk_index": row["chunk_index"],
                "source_file": row["source_file"],
            },
        )
        for _, row in all_rows.iterrows()
    ]


def Hybrid_search(query, table, embedding_manager, top_k=1):
    bm25_retriever = LocalBM25Retriever.from_documents(_bm25_documents_from_table(table))
    bm25_retriever.k = top_k

    try:
        vector_retriever = LanceDBDirectRetriever(
            table=table,
            embedding_manager=embedding_manager,
            top_k=top_k,
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],
        )
        return ensemble_retriever.invoke(query)
    except Exception:
        # Linux CI / tiny tables: keep lexical retrieval working if vector leg fails.
        return bm25_retriever.invoke(query)


