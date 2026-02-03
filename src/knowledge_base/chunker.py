"""Semantic chunking module for Thai text documents.

This module implements semantic chunking strategies that respect Thai
sentence boundaries and preserve context for medical terminology.

The chunking strategy is specifically designed for Thai medical documents
where maintaining semantic coherence is critical for accurate RAG retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core.exceptions import ChunkingError


logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a single chunk of text.

    Attributes:
        text: Chunk content.
        chunk_id: Unique identifier for this chunk.
        start_char: Starting character index in original text.
        end_char: Ending character index in original text.
        sentence_count: Number of sentences in chunk.
        metadata: Additional chunk metadata.
    """
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    sentence_count: int
    metadata: Dict[str, Any]


class SemanticChunker:
    """Semantic chunker for Thai text with sentence-boundary preservation.

    Implements a chunking strategy that:
    1. Tokenizes Thai text into sentences using pythainlp
    2. Groups sentences into semantically coherent chunks
    3. Respects size constraints while maintaining sentence integrity
    4. Adds overlap between chunks for context preservation

    This is particularly important for Thai medical text where splitting
    mid-sentence could break critical context.

    Example:
        >>> chunker = SemanticChunker(
        ...     chunk_size=512,
        ...     chunk_overlap=50,
        ...     min_chunk_size=100
        ... )
        >>> text = "ฉันควรกินยาแก้ปวด..."
        >>> chunks = chunker.chunk_text(text)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        sentence_tokenizer: str = "crfcut"
    ):
        """Initialize semantic chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
            min_chunk_size: Minimum chunk size in characters.
            max_chunk_size: Maximum chunk size in characters.
            sentence_tokenizer: pythainlp sentence tokenizer engine.
                Options: "crfcut" (default, most accurate), "whitespace+newline"
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_tokenizer = sentence_tokenizer

        # Validate parameters
        if chunk_size < min_chunk_size:
            raise ChunkingError(
                f"chunk_size ({chunk_size}) must be >= min_chunk_size ({min_chunk_size})"
            )

        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )

        logger.info(
            f"Initialized SemanticChunker with chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, min={min_chunk_size}, max={max_chunk_size}"
        )

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Chunk text into semantic segments.

        Args:
            text: Thai or mixed Thai-English text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            List of TextChunk objects.

        Raises:
            ChunkingError: If chunking fails.

        Example:
            >>> chunks = chunker.chunk_text("หลังผ่าตัดฟันคุณควร...")
            >>> print(f"Created {len(chunks)} chunks")
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        metadata = metadata or {}

        try:
            # Step 1: Tokenize into sentences
            sentences = self._tokenize_sentences(text)

            if not sentences:
                logger.warning("No sentences found after tokenization")
                return []

            logger.debug(f"Tokenized into {len(sentences)} sentences")

            # Step 2: Group sentences into chunks
            chunks = self._group_sentences_into_chunks(sentences)

            # Step 3: Create TextChunk objects with metadata
            text_chunks = []
            current_pos = 0

            for i, chunk_text in enumerate(chunks):
                # Find chunk position in original text
                start_char = text.find(chunk_text, current_pos)
                if start_char == -1:
                    # Fallback: approximate position
                    start_char = current_pos

                end_char = start_char + len(chunk_text)
                current_pos = start_char + 1  # Move forward for next search

                # Count sentences in this chunk
                chunk_sentences = [s for s in sentences if s in chunk_text]

                chunk_metadata = {
                    **metadata,
                    "chunking_strategy": "semantic",
                    "tokenizer": self.sentence_tokenizer,
                }

                text_chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_id=i,
                    start_char=start_char,
                    end_char=end_char,
                    sentence_count=len(chunk_sentences),
                    metadata=chunk_metadata
                ))

            logger.info(
                f"Created {len(text_chunks)} chunks from {len(sentences)} sentences"
            )
            return text_chunks

        except Exception as e:
            error_msg = f"Failed to chunk text: {e}"
            logger.error(error_msg)
            raise ChunkingError(error_msg) from e

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize Thai text into sentences.

        Uses pythainlp's sentence tokenization which handles Thai
        punctuation and sentence boundaries properly.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        try:
            from pythainlp.tokenize import sent_tokenize

            # Use pythainlp's sentence tokenizer
            sentences = sent_tokenize(text, engine=self.sentence_tokenizer)

            # Filter out empty sentences and strip whitespace
            sentences = [s.strip() for s in sentences if s.strip()]

            return sentences

        except ImportError:
            logger.warning(
                "pythainlp not available, falling back to basic sentence splitting"
            )
            return self._fallback_sentence_split(text)

    def _fallback_sentence_split(self, text: str) -> List[str]:
        """Fallback sentence splitter when pythainlp is not available.

        Uses basic rules for Thai punctuation: ။ (Thai period), ! ?
        and newlines.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        import re

        # Split on Thai/English sentence endings and newlines
        pattern = r'[။.!?]+|\n+'
        sentences = re.split(pattern, text)

        # Filter and clean
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks respecting size constraints.

        Strategy:
        1. Add sentences to current chunk until target size reached
        2. Ensure no chunk exceeds max_chunk_size
        3. Ensure all chunks meet min_chunk_size (combine if needed)
        4. Add overlap by including sentences from previous chunk

        Args:
            sentences: List of sentence strings.

        Returns:
            List of chunk strings.
        """
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            # Check if adding this sentence would exceed max size
            if current_size + sentence_len > self.max_chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk)

                # Only add chunk if it meets minimum size
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)

                    # Start new chunk with overlap from previous chunk
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk, self.chunk_overlap
                    )
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    # Chunk too small, keep accumulating
                    pass

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space

            # Check if we've reached target chunk size
            if current_size >= self.chunk_size:
                chunk_text = ' '.join(current_chunk)

                # Ensure chunk meets minimum size
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)

                    # Start new chunk with overlap
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk, self.chunk_overlap
                    )
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) for s in current_chunk)

        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)

            # Handle final chunk
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:
                # Too small for standalone, merge with previous chunk
                chunks[-1] = chunks[-1] + ' ' + chunk_text
            else:
                # Only chunk and it's too small, but include it anyway
                chunks.append(chunk_text)

        return chunks

    def _get_overlap_sentences(
        self,
        sentences: List[str],
        overlap_size: int
    ) -> List[str]:
        """Get sentences from end of list that fit within overlap size.

        Args:
            sentences: List of sentences.
            overlap_size: Target overlap size in characters.

        Returns:
            List of sentences for overlap.
        """
        overlap_sentences = []
        current_size = 0

        # Take sentences from end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_len = len(sentence)

            if current_size + sentence_len > overlap_size:
                break

            overlap_sentences.insert(0, sentence)
            current_size += sentence_len + 1  # +1 for space

        return overlap_sentences

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        preserve_page_boundaries: bool = True
    ) -> List[TextChunk]:
        """Chunk multiple pages of text.

        Args:
            pages: List of page dictionaries with 'text' and 'page_number' keys.
            preserve_page_boundaries: If True, don't chunk across page boundaries.

        Returns:
            List of TextChunk objects with page metadata.

        Example:
            >>> pages = [
            ...     {"text": "หน้าแรก...", "page_number": 1},
            ...     {"text": "หน้าสอง...", "page_number": 2}
            ... ]
            >>> chunks = chunker.chunk_pages(pages)
        """
        all_chunks = []
        chunk_id_offset = 0

        for page in pages:
            page_text = page.get("text", "")
            page_number = page.get("page_number", 1)

            if not page_text.strip():
                continue

            # Chunk this page
            page_metadata = {"page_number": page_number}
            chunks = self.chunk_text(page_text, metadata=page_metadata)

            # Update chunk IDs to be globally unique
            for chunk in chunks:
                chunk.chunk_id += chunk_id_offset

            all_chunks.extend(chunks)
            chunk_id_offset = len(all_chunks)

            logger.debug(
                f"Page {page_number}: created {len(chunks)} chunks "
                f"(total: {len(all_chunks)})"
            )

        return all_chunks


def chunk_document(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs
) -> List[TextChunk]:
    """Convenience function to chunk a document.

    Args:
        text: Text to chunk.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.
        **kwargs: Additional arguments for SemanticChunker.

    Returns:
        List of TextChunk objects.

    Example:
        >>> chunks = chunk_document("ฉันควรกินยา...", chunk_size=256)
    """
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
    return chunker.chunk_text(text)
