import pyfiles_path  # noqa: F401
from sentence_transformers import SentenceTransformer 
from exception import EmbedModelLoaderException
from logger import logger 
from typing import List, Dict, Any, Tuple 
import numpy as np 


embedding_model = r"D:\RAG Learn\Models\intfloat-multilingual-e5-large" # Best overall free/open-source: 
# Fully free, open-source (MIT-style license), self-hostable
# Non-Chinese origin (Microsoft) 
# Strong multilingual support (handles Hindi/regional Indian languages)
# Excellent retrieval accuracy on standard benchmarks (MTEB)

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""

    def __init__(self, embedmodel_path= embedding_model):
        self.model_name = embedmodel_path
        self.model = None 
        self.load_model()

    def load_model(self):
        """Load the SentenceTransformer model"""
    #    logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            dim = self.model.get_embedding_dimension()
        #    logger.info(f"Model loaded successfully — embedding dimension: {dim}")
        except Exception as e:
            logger.error(f"Model loading failed for '{self.model_name}': {e}", exc_info=True)
            raise EmbedModelLoaderException(e, sys) from e

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            logger.error("generate_embeddings called before model was loaded")
            raise ValueError('Model not loaded')

    #    logger.info(f"Generating embeddings for {len(texts)} texts...")

        import time
        start = time.time()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        elapsed = time.time() - start

    #    logger.info(
    #        f"Generated embeddings with shape {embeddings.shape} in {elapsed:.2f}s "
    #        f"({len(texts) / elapsed:.1f} texts/s)"
    #    )
        return embeddings 