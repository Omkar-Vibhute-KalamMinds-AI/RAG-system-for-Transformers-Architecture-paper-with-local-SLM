import _bootstrap  # noqa: F401
from typing import List, Dict, Any 
import lancedb
from EmbedModelLoader import EmbeddingManager 
from exception import RetrievialSystemException
from logger import logger  
from HybridSearch import Hybrid_search
from config_loader import load_config 

config = load_config()

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, lancedb_table, embedding_manager):
        """
        Initialize the retriever

        Args:
            vectore_store (_type_): contains the embedding vectors of the documents.
            embedding_manager (EmbeddingManager class): for generating the query embeddings.
        """
        self.lancedb_table = lancedb_table  
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves relevant documents for the query

        Args:
            query (str): user query
            top_k (int): Number of top results for the query. Defaults to 1.
        
        Returns:
            List[Dict[str, Any]]: list of dicts containing retrieved documents and metadata.
        """
    
        try:
            docs = Hybrid_search(
                query,
                self.lancedb_table,
                self.embedding_manager,
                top_k=top_k,
            )

            retrieved_docs = []
            for i, doc in enumerate(docs):
                retrieved_docs.append({
                    "id": doc.metadata.get("id"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "content": doc.page_content,
                    "source_file": doc.metadata.get("source_file"),
                    "rank": i + 1,
                })
            return retrieved_docs

        except Exception as e:
            logger.error(f"Retrieval failed for query '{query} of length {len(query)}': {e}", exc_info=True)
            raise RetrievialSystemException(f"Retrieval failed for query '{query}': {e}") from e
