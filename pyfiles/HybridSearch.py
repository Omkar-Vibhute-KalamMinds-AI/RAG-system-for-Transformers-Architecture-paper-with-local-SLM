import sys
from typing import List

sys.path.append(r"D:\LLMOps\pyfiles")
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from bm25_retriever import LocalBM25Retriever


class LanceDBDirectRetriever(BaseRetriever):
    table: object = Field(...)
    embedding_manager: object = Field(...)
    top_k: int = 1

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = (
            self.table.search(query_embedding, vector_column_name="vector")
            .nprobes(2)
            .limit(self.top_k)
            .to_pandas()
        )

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
    vector_retriever = LanceDBDirectRetriever(
        table=table,
        embedding_manager=embedding_manager,
        top_k=top_k,
    )

    bm25_retriever = LocalBM25Retriever.from_documents(_bm25_documents_from_table(table))
    bm25_retriever.k = top_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )
    return ensemble_retriever.invoke(query)
