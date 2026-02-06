"""Embedding generation module using EmbeddingGemma-300M model.

This module provides embedding generation for Thai and mixed Thai-English
text using Google's EmbeddingGemma-300M via sentence-transformers.

EmbeddingGemma-300M uses asymmetric encoding with different prompt templates
for queries vs documents, producing 768-dimensional dense embeddings.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.core.exceptions import EmbeddingError
from src.utils.text_processing import clean_thai_text_for_embedding


logger = logging.getLogger(__name__)


class GemmaEmbeddingGenerator:
    """EmbeddingGemma-300M embedding generator for multilingual text.

    Generates dense embeddings for Thai and mixed Thai-English text
    using Google's EmbeddingGemma-300M via sentence-transformers.

    This model uses asymmetric encoding: queries and documents use
    different prompt templates via encode_query() and encode_document().

    Attributes:
        model_name: HuggingFace model identifier.
        device: Computation device ('cuda' or 'cpu').
        batch_size: Batch size for encoding.
        max_length: Maximum sequence length for model.
        model: Loaded SentenceTransformer instance.

    Example:
        >>> generator = GemmaEmbeddingGenerator()
        >>> texts = ["ฉันควรกินยาแก้ปวด", "หลังผ่าตัดควรพักผ่อน"]
        >>> embeddings = generator.encode(texts)
        >>> print(embeddings[0]['dense'].shape)  # (768,)
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: Optional[str] = None,
        batch_size: int = 32,
        use_fp16: bool = False,
        max_length: int = 2048,
        normalize_embeddings: bool = True
    ):
        """Initialize EmbeddingGemma-300M embedding generator.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
            batch_size: Batch size for encoding.
            use_fp16: Ignored. EmbeddingGemma does not support fp16.
            max_length: Maximum sequence length (up to 2048).
            normalize_embeddings: Whether to L2-normalize dense embeddings.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # Determine device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if use_fp16:
            logger.warning(
                "EmbeddingGemma does not support fp16. Using fp32/bfloat16 instead."
            )

        logger.info(f"Initializing EmbeddingGemma model: {model_name}")
        logger.info(f"Device: {self.device}, Batch size: {batch_size}")

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Load EmbeddingGemma model from sentence-transformers.

        Returns:
            Loaded SentenceTransformer instance.

        Raises:
            EmbeddingError: If model loading fails.
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            model.max_seq_length = self.max_length

            logger.info(f"Successfully loaded EmbeddingGemma model on {self.device}")
            return model

        except ImportError:
            error_msg = (
                "sentence-transformers library required for EmbeddingGemma. "
                "Install with: pip install sentence-transformers"
            )
            logger.error(error_msg)
            raise EmbeddingError(error_msg)

        except Exception as e:
            error_msg = f"Failed to load EmbeddingGemma model: {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

    def encode(
        self,
        texts: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        clean_text: bool = True,
        is_query: bool = False
    ) -> List[Dict[str, Any]]:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.
            return_dense: Whether to return dense embeddings.
            return_sparse: Kept for interface compatibility. Sparse always empty.
            clean_text: Whether to clean Thai text before encoding.
            is_query: If True, use query prompt template (encode_query).
                      If False, use document prompt template (encode_document).

        Returns:
            List of dictionaries containing embeddings:
            [
                {
                    'dense': np.ndarray,  # (768,) dense vector
                    'sparse': {},  # empty (no sparse for EmbeddingGemma)
                    'lexical_weights': {}  # empty
                },
                ...
            ]

        Raises:
            EmbeddingError: If encoding fails.

        Example:
            >>> embeddings = generator.encode(["ฉันควรกินยา", "พักผ่อน"])
            >>> dense = embeddings[0]['dense']
        """
        if not texts:
            logger.warning("Empty text list provided for encoding")
            return []

        # Clean Thai text if enabled
        if clean_text:
            texts = [clean_thai_text_for_embedding(text) for text in texts]

        try:
            logger.debug(f"Encoding {len(texts)} texts (is_query={is_query})...")

            # Use asymmetric encoding
            if is_query:
                embeddings_array = self.model.encode_query(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=False
                )
            else:
                embeddings_array = self.model.encode_document(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=False
                )

            # Process output into standardized format
            embeddings = []

            for i in range(len(texts)):
                embedding_dict = {
                    'dense': embeddings_array[i],
                    'sparse': {},
                    'lexical_weights': {}
                }
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
        clean_text: bool = True,
        is_query: bool = False
    ) -> Dict[str, Any]:
        """Encode a single text into embeddings.

        Convenience method for encoding single text.

        Args:
            text: Text string to encode.
            return_dense: Whether to return dense embedding.
            return_sparse: Kept for interface compatibility.
            clean_text: Whether to clean Thai text before encoding.
            is_query: If True, use query prompt template.

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
            clean_text=clean_text,
            is_query=is_query
        )
        return embeddings[0] if embeddings else {}

    def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        is_query: bool = False
    ) -> List[Dict[str, Any]]:
        """Encode texts in batches for memory efficiency.

        Args:
            texts: List of texts to encode.
            batch_size: Batch size. Uses instance batch_size if None.
            is_query: If True, use query prompt template.

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
            batch_embeddings = self.encode(batch, is_query=is_query)
            all_embeddings.extend(batch_embeddings)

            if (i // batch_size + 1) % 10 == 0:
                logger.debug(
                    f"Processed {i + len(batch)}/{len(texts)} texts"
                )

        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """Get dimension of dense embeddings.

        Returns:
            Embedding dimension (768 for EmbeddingGemma-300M).
        """
        return 768

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GemmaEmbeddingGenerator(model={self.model_name}, "
            f"device={self.device}, dim={self.get_embedding_dimension()})"
        )


# Backward-compatible alias
BGEEmbeddingGenerator = GemmaEmbeddingGenerator


def create_embedding_generator(config: Optional[Dict[str, Any]] = None):
    """Factory function to create embedding generator from config.

    Args:
        config: Configuration dictionary. If None, uses defaults.

    Returns:
        GemmaEmbeddingGenerator instance.

    Example:
        >>> config = {
        ...     "model_name": "google/embeddinggemma-300m",
        ...     "device": "cuda",
        ...     "batch_size": 32
        ... }
        >>> generator = create_embedding_generator(config)
    """
    if config is None:
        config = {}

    return GemmaEmbeddingGenerator(
        model_name=config.get("model_name", "google/embeddinggemma-300m"),
        device=config.get("device"),
        batch_size=config.get("batch_size", 32),
        use_fp16=config.get("use_fp16", False),
        max_length=config.get("max_length", 2048),
        normalize_embeddings=config.get("normalize_embeddings", True)
    )


def encode_documents(
    texts: List[str],
    generator: Optional[GemmaEmbeddingGenerator] = None,
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
        generator = GemmaEmbeddingGenerator(batch_size=batch_size)

    embeddings = generator.encode_batch(texts, batch_size=batch_size, is_query=False)

    # Convert to tuple format
    results = []
    for emb in embeddings:
        dense = emb.get('dense', [])
        sparse = emb.get('sparse', {})
        results.append((dense.tolist() if hasattr(dense, 'tolist') else dense, sparse))

    return results
