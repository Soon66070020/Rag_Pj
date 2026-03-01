"""Metadata extraction module for document chunks.

This module extracts metadata from document chunks including:
- Category classification based on Thai medical keywords
- Document properties (source, page number, etc.)
- Content statistics (length, sentence count, etc.)
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.utils.text_processing import (
    infer_category_from_thai_keywords,
    infer_categories_from_taxonomy,
    extract_thai_keywords
)


logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from document chunks.

    Provides methods to extract various metadata fields from text chunks,
    including automated category classification based on Thai medical
    keywords and content analysis.

    Example:
        >>> extractor = MetadataExtractor()
        >>> metadata = extractor.extract(
        ...     text="ฉันควรกินยาแก้ปวด",
        ...     source_file="medication_guide.pdf",
        ...     page_number=5
        ... )
        >>> print(metadata['category'])
        'Medication'
    """

    def __init__(
        self,
        auto_infer_category: bool = True,
        extract_keywords: bool = True,
        default_category: str = "Post-op Care",
        taxonomy: Optional[Dict[str, Any]] = None
    ):
        """Initialize metadata extractor.

        Args:
            auto_infer_category: Automatically infer category from content.
            extract_keywords: Extract Thai keywords from content.
            default_category: Default category if inference fails.
            taxonomy: Optional taxonomy dict for multi-category inference.
                If provided, uses taxonomy-based classification instead of
                simple keyword matching.
        """
        self.auto_infer_category = auto_infer_category
        self.extract_keywords = extract_keywords
        self.default_category = default_category
        self.taxonomy = taxonomy

    def extract(
        self,
        text: str,
        source_file: str,
        page_number: int,
        chunk_id: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract metadata from a text chunk.

        Args:
            text: Chunk text content.
            source_file: Source file path or name.
            page_number: Page number in source document.
            chunk_id: Optional chunk identifier.
            additional_metadata: Additional metadata to include.

        Returns:
            Dictionary containing extracted metadata.

        Example:
            >>> metadata = extractor.extract(
            ...     text="หลังผ่าตัดควรกินอาหารอ่อน",
            ...     source_file="nutrition.pdf",
            ...     page_number=3
            ... )
        """
        metadata = additional_metadata.copy() if additional_metadata else {}

        # Basic document properties
        metadata['source_file'] = Path(source_file).name
        metadata['page_number'] = page_number

        if chunk_id is not None:
            metadata['chunk_id'] = chunk_id

        # Content statistics
        metadata.update(self._extract_content_stats(text))

        # Category inference
        if self.auto_infer_category:
            category = self._infer_category(text, metadata)
            metadata['category'] = category

            # Multi-category inference using taxonomy
            if self.taxonomy:
                tax_result = infer_categories_from_taxonomy(text, self.taxonomy)
                metadata['categories'] = tax_result['categories']
                metadata['subcategories'] = tax_result['subcategories']
            else:
                metadata['categories'] = [category]
                metadata['subcategories'] = []
        else:
            metadata['category'] = metadata.get('category', self.default_category)
            metadata['categories'] = [metadata['category']]
            metadata['subcategories'] = []

        # Keyword extraction
        if self.extract_keywords:
            keywords = self._extract_keywords(text)
            metadata['keywords'] = keywords
            metadata['keyword_count'] = len(keywords)

        logger.debug(
            f"Extracted metadata for {metadata['source_file']} "
            f"(page {page_number}, category: {metadata['category']})"
        )

        return metadata

    def _extract_content_stats(self, text: str) -> Dict[str, Any]:
        """Extract statistical information about the content.

        Args:
            text: Text content.

        Returns:
            Dictionary with content statistics.
        """
        stats = {
            'char_count': len(text),
            'word_count': len(text.split()),
        }

        # Estimate sentence count (rough approximation)
        sentence_endings = text.count('.') + text.count('!') + text.count('?') + text.count('။')
        stats['sentence_count'] = max(1, sentence_endings)

        return stats

    def _infer_category(
        self,
        text: str,
        existing_metadata: Dict[str, Any]
    ) -> str:
        """Infer document category from content.

        Uses Thai medical keywords to classify content into categories:
        - Emergency: Urgent medical situations
        - Medication: Medicine and drug-related
        - Nutrition: Food and eating-related
        - Post-op Care: General post-operative care

        Args:
            text: Text content.
            existing_metadata: Existing metadata (may contain hints).

        Returns:
            Inferred category string.
        """
        # Check if category already specified in metadata
        if 'category' in existing_metadata:
            return existing_metadata['category']

        # Use Thai keyword-based inference
        category = infer_category_from_thai_keywords(text)

        logger.debug(f"Inferred category: {category}")
        return category

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important Thai keywords from text.

        Args:
            text: Text content.
            top_n: Number of top keywords to extract.

        Returns:
            List of Thai keywords.
        """
        keywords = extract_thai_keywords(text, top_n=top_n)
        return keywords

    def extract_from_source_filename(self, filename: str) -> Dict[str, Any]:
        """Extract metadata hints from source filename.

        Looks for category hints in filename (e.g., "medication_guide.pdf"
        suggests Medication category).

        Args:
            filename: Source filename.

        Returns:
            Dictionary with extracted metadata hints.
        """
        metadata = {}
        filename_lower = filename.lower()

        # Category hints from filename
        if any(word in filename_lower for word in ['medication', 'medicine', 'drug', 'ยา']):
            metadata['filename_category_hint'] = 'Medication'
        elif any(word in filename_lower for word in ['emergency', 'urgent', 'ฉุกเฉิน']):
            metadata['filename_category_hint'] = 'Emergency'
        elif any(word in filename_lower for word in ['nutrition', 'food', 'diet', 'อาหาร']):
            metadata['filename_category_hint'] = 'Nutrition'
        elif any(word in filename_lower for word in ['postop', 'post-op', 'care', 'ดูแล']):
            metadata['filename_category_hint'] = 'Post-op Care'

        return metadata

    def merge_metadata(
        self,
        chunk_metadata: Dict[str, Any],
        page_metadata: Dict[str, Any],
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge metadata from different levels.

        Merges chunk-level, page-level, and document-level metadata
        with proper precedence (chunk > page > document).

        Args:
            chunk_metadata: Chunk-specific metadata.
            page_metadata: Page-level metadata.
            document_metadata: Document-level metadata.

        Returns:
            Merged metadata dictionary.
        """
        # Start with document-level (lowest priority)
        merged = document_metadata.copy()

        # Add page-level (medium priority)
        merged.update(page_metadata)

        # Add chunk-level (highest priority)
        merged.update(chunk_metadata)

        return merged


def extract_metadata(
    text: str,
    source_file: str,
    page_number: int,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to extract metadata.

    Args:
        text: Text content.
        source_file: Source file path.
        page_number: Page number.
        **kwargs: Additional arguments for MetadataExtractor.

    Returns:
        Metadata dictionary.

    Example:
        >>> metadata = extract_metadata(
        ...     "ฉันควรกินยา",
        ...     "medication.pdf",
        ...     5
        ... )
    """
    extractor = MetadataExtractor(**kwargs)
    return extractor.extract(text, source_file, page_number)


def batch_extract_metadata(
    chunks: List[Dict[str, Any]],
    extractor: Optional[MetadataExtractor] = None
) -> List[Dict[str, Any]]:
    """Extract metadata for multiple chunks.

    Args:
        chunks: List of chunk dictionaries with 'text', 'source_file',
            'page_number' keys.
        extractor: MetadataExtractor instance. Creates new if None.

    Returns:
        List of metadata dictionaries.

    Example:
        >>> chunks = [
        ...     {"text": "chunk1", "source_file": "doc.pdf", "page_number": 1},
        ...     {"text": "chunk2", "source_file": "doc.pdf", "page_number": 2}
        ... ]
        >>> metadata_list = batch_extract_metadata(chunks)
    """
    if extractor is None:
        extractor = MetadataExtractor()

    metadata_list = []

    for i, chunk in enumerate(chunks):
        metadata = extractor.extract(
            text=chunk.get('text', ''),
            source_file=chunk.get('source_file', 'unknown'),
            page_number=chunk.get('page_number', 1),
            chunk_id=i,
            additional_metadata=chunk.get('metadata', {})
        )
        metadata_list.append(metadata)

    return metadata_list
