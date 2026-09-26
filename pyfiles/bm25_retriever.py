"""
Lexical BM25 retriever backed by rank_bm25.

Why: LangChain's BM25Retriever lives in langchain_community (deprecated re-exports
from langchain_classic). This module keeps the same from_documents / .k API used by
HybridSearch while depending only on rank_bm25 + langchain_core.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from exception import BM25RetrieverException
from logger import logger

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    logger.error("rank_bm25 is not installed", exc_info=True)
    raise BM25RetrieverException("Install rank_bm25: pip install rank_bm25", sys) from e


def _default_preprocess(text: str) -> List[str]:
    normalized = text.lower().replace("-", " ")
    return normalized.split()


class LocalBM25Retriever(BaseRetriever):
    """Okapi BM25 over in-memory documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorizer: BM25Okapi
    docs: List[Document]
    k: int = 4
    preprocess_func: Callable[[str], List[str]] = Field(default=_default_preprocess)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            logger.info(
                "BM25 retrieve: query_len=%s k=%s corpus=%s",
                len(query),
                self.k,
                len(self.docs),
            )
            tokenized = self.preprocess_func(query)
            scores = self.vectorizer.get_scores(tokenized)
            if len(scores) == 0:
                logger.warning("BM25 retrieve: empty score vector for query_len=%s", len(query))
                return []
            top_idx = np.argsort(scores)[::-1][: self.k]
            hits = [self.docs[i] for i in top_idx]
        
            return hits
        except Exception as e:
            logger.error(
                "BM25 retrieve failed for query_len=%s: %s",
                len(query),
                e,
                exc_info=True,
            )
            raise BM25RetrieverException(f"Error in _get_relevant_documents: {e}", sys) from e

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = _default_preprocess,
        **kwargs: Any,
    ) -> LocalBM25Retriever:
        try:
            docs = list(documents)
            if not docs:
                logger.error("BM25 from_documents called with an empty document list")
                raise ValueError(
                    "LocalBM25Retriever.from_documents requires at least one document"
                )
            logger.info("BM25 from_documents: indexing %s documents", len(docs))
            tokenized_corpus = [preprocess_func(d.page_content) for d in docs]
            vectorizer = BM25Okapi(tokenized_corpus, **(bm25_params or {}))
            retriever = cls(
                vectorizer=vectorizer,
                docs=docs,
                preprocess_func=preprocess_func,
                **kwargs,
            )
         
            return retriever
        except BM25RetrieverException:
            raise
        except Exception as e:
            logger.error("BM25 from_documents failed: %s", e, exc_info=True)
            raise BM25RetrieverException(f"Error in from_documents: {e}", sys) from e
