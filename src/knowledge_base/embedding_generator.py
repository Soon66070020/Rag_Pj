"""Embedding generation module using BGE-M3 model.

This module provides embedding generation for Thai and mixed Thai-English
text using the BGE-M3 multilingual embedding model.

BGE-M3 generates both dense (semantic) and sparse (lexical) embeddings,
which are used for Weaviate hybrid search.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.core.exceptions import EmbeddingError
from src.utils.text_processing import clean_thai_text_for_embedding


logger = logging.getLogger(__name__)


class BGEEmbeddingGenerator:
    """BGE-M3 embedding generator for multilingual text.

    Generates both dense and sparse embeddings for Thai and mixed
    Thai-English text using the BAAI/bge-m3 model.

    Dense embeddings capture semantic meaning (1024-dim).
    Sparse embeddings capture lexical features (learned token weights).

    Attributes:
        model_name: HuggingFace model identifier.
        device: Computation device ('cuda' or 'cpu').
        batch_size: Batch size for encoding.
        use_fp16: Whether to use half precision (faster, less memory).
        max_length: Maximum sequence length for model.
        model: Loaded BGE-M3 model instance.

    Example:
        >>> generator = BGEEmbeddingGenerator()
        >>> texts = ["ฉันควรกินยาแก้ปวด", "หลังผ่าตัดควรพักผ่อน"]
        >>> embeddings = generator.encode(texts)
        >>> print(embeddings[0]['dense'].shape)  # (1024,)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
        use_fp16: bool = True,
        max_length: int = 512,
        normalize_embeddings: bool = True
    ):
        """Initialize BGE-M3 embedding generator.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
            batch_size: Batch size for encoding.
            use_fp16: Use half precision (FP16) for faster inference.
            max_length: Maximum sequence length.
            normalize_embeddings: Whether to L2-normalize dense embeddings.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # Determine device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing BGE-M3 model: {model_name}")
        logger.info(f"Device: {self.device}, FP16: {use_fp16}, Batch size: {batch_size}")

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Load BGE-M3 model from FlagEmbedding.

        Returns:
            Loaded BGEM3FlagModel instance.

        Raises:
            EmbeddingError: If model loading fails.
        """
        try:
            from FlagEmbedding import BGEM3FlagModel

            model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )

            logger.info(f"Successfully loaded BGE-M3 model on {self.device}")
            return model

        except ImportError:
            error_msg = (
                "FlagEmbedding library required for BGE-M3. "
                "Install with: pip install FlagEmbedding"
            )
            logger.error(error_msg)
            raise EmbeddingError(error_msg)

        except Exception as e:
            error_msg = f"Failed to load BGE-M3 model: {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

    def encode(
        self,
        texts: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        clean_text: bool = True
    ) -> List[Dict[str, Any]]:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.
            return_dense: Whether to return dense embeddings.
            return_sparse: Whether to return sparse embeddings.
            clean_text: Whether to clean Thai text before encoding.

        Returns:
            List of dictionaries containing embeddings:
            [
                {
                    'dense': np.ndarray,  # (1024,) dense vector
                    'sparse': Dict[str, float],  # {token: weight}
                    'lexical_weights': Dict[str, float]  # same as sparse
                },
                ...
            ]

        Raises:
            EmbeddingError: If encoding fails.

        Example:
            >>> embeddings = generator.encode(["ฉันควรกินยา", "พักผ่อน"])
            >>> dense = embeddings[0]['dense']
            >>> sparse = embeddings[0]['sparse']
        """
        if not texts:
            logger.warning("Empty text list provided for encoding")
            return []

        # Clean Thai text if enabled
        if clean_text:
            texts = [clean_thai_text_for_embedding(text) for text in texts]

        try:
            logger.debug(f"Encoding {len(texts)} texts...")

            # Encode with BGE-M3
            # BGE-M3 returns dict with 'dense_vecs' and 'lexical_weights'
            output = self.model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=False  # We don't need ColBERT vectors
            )

            # Process output into standardized format
            embeddings = []

            for i in range(len(texts)):
                embedding_dict = {}

                # Dense embeddings
                if return_dense and 'dense_vecs' in output:
                    dense_vec = output['dense_vecs'][i]

                    # Normalize if enabled
                    if self.normalize_embeddings:
                        dense_vec = self._normalize_vector(dense_vec)

                    embedding_dict['dense'] = dense_vec

                # Sparse embeddings (lexical weights)
                if return_sparse and 'lexical_weights' in output:
                    # BGE-M3 returns sparse as dict of {token_id: weight}
                    # We need to convert to {token: weight} format
                    sparse_weights = output['lexical_weights'][i]

                    # Convert to standard format
                    sparse_dict = self._process_sparse_weights(sparse_weights)

                    embedding_dict['sparse'] = sparse_dict
                    embedding_dict['lexical_weights'] = sparse_dict

                embeddings.append(embedding_dict)

            logger.info(f"Successfully encoded {len(texts)} texts")
            return embeddings

        except Exception as e:
            error_msg = f"Failed to encode texts: {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

    def encode_single(
        self,
        text: str,
        return_dense: bool = True,
        return_sparse: bool = True,
        clean_text: bool = True
    ) -> Dict[str, Any]:
        """Encode a single text into embeddings.

        Convenience method for encoding single text.

        Args:
            text: Text string to encode.
            return_dense: Whether to return dense embedding.
            return_sparse: Whether to return sparse embedding.
            clean_text: Whether to clean Thai text before encoding.

        Returns:
            Dictionary with 'dense' and 'sparse' keys.

        Example:
            >>> embedding = generator.encode_single("ฉันควรกินยา")
            >>> print(embedding['dense'].shape)
        """
        embeddings = self.encode(
            [text],
            return_dense=return_dense,
            return_sparse=return_sparse,
            clean_text=clean_text
        )
        return embeddings[0] if embeddings else {}

    def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Encode texts in batches for memory efficiency.

        Args:
            texts: List of texts to encode.
            batch_size: Batch size. Uses instance batch_size if None.

        Returns:
            List of embedding dictionaries.

        Example:
            >>> large_texts = ["text1", "text2", ..., "text1000"]
            >>> embeddings = generator.encode_batch(large_texts, batch_size=64)
        """
        batch_size = batch_size or self.batch_size

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch)
            all_embeddings.extend(batch_embeddings)

            if (i // batch_size + 1) % 10 == 0:
                logger.debug(
                    f"Processed {i + len(batch)}/{len(texts)} texts"
                )

        return all_embeddings

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector.

        Args:
            vec: Input vector.

        Returns:
            Normalized vector.
        """
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _process_sparse_weights(
        self,
        sparse_weights: Dict[int, float]
    ) -> Dict[str, float]:
        """Process sparse weights from BGE-M3 format.

        BGE-M3 returns sparse weights as {token_id: weight}.
        For simplicity and Weaviate compatibility, we convert to
        {token_string: weight}.

        Args:
            sparse_weights: Raw sparse weights from model.

        Returns:
            Processed sparse weights dictionary.
        """
        # Convert token IDs to strings and keep weights
        # In production, you might want to decode token IDs to actual tokens
        # using the tokenizer, but for now we use string IDs
        processed = {
            f"token_{token_id}": weight
            for token_id, weight in sparse_weights.items()
        }

        return processed

    def get_embedding_dimension(self) -> int:
        """Get dimension of dense embeddings.

        Returns:
            Embedding dimension (1024 for BGE-M3).
        """
        return 1024  # BGE-M3 produces 1024-dim embeddings

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BGEEmbeddingGenerator(model={self.model_name}, "
            f"device={self.device}, dim={self.get_embedding_dimension()})"
        )


def create_embedding_generator(config: Optional[Dict[str, Any]] = None):
    """Factory function to create embedding generator from config.

    Args:
        config: Configuration dictionary. If None, uses defaults.

    Returns:
        BGEEmbeddingGenerator instance.

    Example:
        >>> config = {
        ...     "model_name": "BAAI/bge-m3",
        ...     "device": "cuda",
        ...     "batch_size": 32
        ... }
        >>> generator = create_embedding_generator(config)
    """
    if config is None:
        config = {}

    return BGEEmbeddingGenerator(
        model_name=config.get("model_name", "BAAI/bge-m3"),
        device=config.get("device"),
        batch_size=config.get("batch_size", 32),
        use_fp16=config.get("use_fp16", True),
        max_length=config.get("max_length", 512),
        normalize_embeddings=config.get("normalize_embeddings", True)
    )


def encode_documents(
    texts: List[str],
    generator: Optional[BGEEmbeddingGenerator] = None,
    batch_size: int = 32
) -> List[Tuple[List[float], Dict[str, float]]]:
    """Convenience function to encode documents.

    Args:
        texts: List of document texts.
        generator: Embedding generator. Creates new if None.
        batch_size: Batch size for encoding.

    Returns:
        List of (dense_vector, sparse_dict) tuples.

    Example:
        >>> texts = ["doc1", "doc2", "doc3"]
        >>> embeddings = encode_documents(texts)
        >>> dense, sparse = embeddings[0]
    """
    if generator is None:
        generator = BGEEmbeddingGenerator(batch_size=batch_size)

    embeddings = generator.encode_batch(texts, batch_size=batch_size)

    # Convert to tuple format
    results = []
    for emb in embeddings:
        dense = emb.get('dense', [])
        sparse = emb.get('sparse', {})
        results.append((dense.tolist() if hasattr(dense, 'tolist') else dense, sparse))

    return results
