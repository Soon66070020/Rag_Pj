"""Document loader module for PDF and text files.

This module provides loaders for different file formats with proper
handling of Thai language text encoding and extraction.

Supports:
- PDF files (with Thai text preservation)
- Plain text files (.txt, .md)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core.exceptions import DocumentLoadError
from src.utils.text_processing import normalize_thai_text


logger = logging.getLogger(__name__)


@dataclass
class LoadedPage:
    """Represents a single page of loaded content.

    Attributes:
        text: Extracted text content.
        page_number: Page number (1-indexed).
        metadata: Additional page metadata.
    """
    text: str
    page_number: int
    metadata: Dict[str, Any]


class DocumentLoader:
    """Base class for document loaders.

    Provides common functionality for all document loaders including
    text normalization and validation.
    """

    def __init__(self, normalize_text: bool = True):
        """Initialize document loader.

        Args:
            normalize_text: Whether to normalize Thai text after extraction.
        """
        self.normalize_text = normalize_text

    def load(self, file_path: Path) -> List[LoadedPage]:
        """Load document and extract text.

        Args:
            file_path: Path to document file.

        Returns:
            List of LoadedPage objects, one per page.

        Raises:
            DocumentLoadError: If file cannot be loaded.
        """
        raise NotImplementedError("Subclasses must implement load()")

    def _normalize_if_enabled(self, text: str) -> str:
        """Normalize text if normalization is enabled.

        Args:
            text: Raw text content.

        Returns:
            Normalized text if enabled, otherwise original text.
        """
        if self.normalize_text:
            return normalize_thai_text(text)
        return text


class PDFLoader(DocumentLoader):
    """PDF document loader with Thai text support.

    Uses pdfplumber for robust text extraction that handles
    Thai characters and mixed Thai-English documents.

    Example:
        >>> loader = PDFLoader()
        >>> pages = loader.load(Path("document.pdf"))
        >>> print(f"Loaded {len(pages)} pages")
    """

    def __init__(self, normalize_text: bool = True, extract_tables: bool = False):
        """Initialize PDF loader.

        Args:
            normalize_text: Whether to normalize Thai text.
            extract_tables: Whether to extract table content (experimental).
        """
        super().__init__(normalize_text)
        self.extract_tables = extract_tables

    def load(self, file_path: Path) -> List[LoadedPage]:
        """Load PDF file and extract text page by page.

        Args:
            file_path: Path to PDF file.

        Returns:
            List of LoadedPage objects.

        Raises:
            DocumentLoadError: If PDF cannot be loaded or parsed.
        """
        if not file_path.exists():
            raise DocumentLoadError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != '.pdf':
            raise DocumentLoadError(f"Not a PDF file: {file_path}")

        try:
            import pdfplumber
        except ImportError:
            raise DocumentLoadError(
                "pdfplumber is required for PDF loading. "
                "Install with: pip install pdfplumber"
            )

        logger.info(f"Loading PDF: {file_path}")
        pages = []

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages")

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    text = page.extract_text()

                    if text is None or text.strip() == "":
                        logger.warning(
                            f"Page {page_num}/{total_pages} has no text content"
                        )
                        continue

                    # Normalize Thai text if enabled
                    text = self._normalize_if_enabled(text)

                    # Build metadata
                    metadata = {
                        "width": page.width,
                        "height": page.height,
                        "total_pages": total_pages,
                    }

                    # Add table count if extraction enabled
                    if self.extract_tables:
                        tables = page.extract_tables()
                        metadata["table_count"] = len(tables) if tables else 0

                    pages.append(LoadedPage(
                        text=text,
                        page_number=page_num,
                        metadata=metadata
                    ))

                    if page_num % 10 == 0:
                        logger.debug(f"Processed {page_num}/{total_pages} pages")

            logger.info(
                f"Successfully loaded {len(pages)} pages from {file_path.name}"
            )
            return pages

        except Exception as e:
            error_msg = f"Failed to load PDF {file_path}: {e}"
            logger.error(error_msg)
            raise DocumentLoadError(error_msg) from e


class TextLoader(DocumentLoader):
    """Plain text file loader with encoding detection.

    Supports .txt and .md files with automatic encoding detection
    for Thai text (UTF-8, TIS-620, etc.).

    Example:
        >>> loader = TextLoader()
        >>> pages = loader.load(Path("document.txt"))
        >>> print(pages[0].text[:100])
    """

    def __init__(
        self,
        normalize_text: bool = True,
        encoding: Optional[str] = None,
        split_by_headers: bool = False
    ):
        """Initialize text loader.

        Args:
            normalize_text: Whether to normalize Thai text.
            encoding: File encoding. If None, auto-detect.
            split_by_headers: Split into pages by markdown headers.
        """
        super().__init__(normalize_text)
        self.encoding = encoding
        self.split_by_headers = split_by_headers

    def load(self, file_path: Path) -> List[LoadedPage]:
        """Load text file.

        Args:
            file_path: Path to text file (.txt, .md).

        Returns:
            List of LoadedPage objects. For plain text files without
            headers, returns single page. For markdown with headers,
            can split into multiple logical pages.

        Raises:
            DocumentLoadError: If file cannot be loaded.
        """
        if not file_path.exists():
            raise DocumentLoadError(f"Text file not found: {file_path}")

        if file_path.suffix.lower() not in ['.txt', '.md', '.markdown']:
            raise DocumentLoadError(
                f"Unsupported text file format: {file_path.suffix}"
            )

        logger.info(f"Loading text file: {file_path}")

        try:
            # Try specified encoding or auto-detect
            encoding = self.encoding or self._detect_encoding(file_path)

            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # Normalize Thai text if enabled
            content = self._normalize_if_enabled(content)

            # Split into pages if requested
            if self.split_by_headers and file_path.suffix.lower() in ['.md', '.markdown']:
                pages = self._split_by_markdown_headers(content)
            else:
                # Single page for entire file
                pages = [LoadedPage(
                    text=content,
                    page_number=1,
                    metadata={"encoding": encoding}
                )]

            logger.info(
                f"Successfully loaded {len(pages)} page(s) from {file_path.name}"
            )
            return pages

        except UnicodeDecodeError as e:
            error_msg = (
                f"Failed to decode {file_path} with encoding {encoding}. "
                "Try specifying encoding explicitly."
            )
            logger.error(error_msg)
            raise DocumentLoadError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load text file {file_path}: {e}"
            logger.error(error_msg)
            raise DocumentLoadError(error_msg) from e

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding.

        Tries common Thai encodings: UTF-8, TIS-620, Windows-874.

        Args:
            file_path: Path to file.

        Returns:
            Detected encoding name.
        """
        # Try UTF-8 first (most common for modern files)
        encodings_to_try = ['utf-8', 'tis-620', 'windows-874', 'cp874']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read()
                logger.debug(f"Detected encoding: {encoding}")
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback to utf-8 with error handling
        logger.warning(
            f"Could not detect encoding for {file_path}, using utf-8 with errors='replace'"
        )
        return 'utf-8'

    def _split_by_markdown_headers(self, content: str) -> List[LoadedPage]:
        """Split markdown content by headers.

        Splits content at # headers to create logical pages.

        Args:
            content: Markdown content.

        Returns:
            List of LoadedPage objects, one per section.
        """
        import re

        # Split by markdown headers (# Header)
        sections = re.split(r'\n(?=#+\s)', content)

        pages = []
        for i, section in enumerate(sections, start=1):
            if section.strip():
                pages.append(LoadedPage(
                    text=section.strip(),
                    page_number=i,
                    metadata={"section_number": i}
                ))

        return pages if pages else [LoadedPage(
            text=content,
            page_number=1,
            metadata={}
        )]


def get_loader(file_path: Path, **kwargs) -> DocumentLoader:
    """Factory function to get appropriate loader for file type.

    Args:
        file_path: Path to document file.
        **kwargs: Additional arguments to pass to loader.

    Returns:
        Appropriate DocumentLoader instance.

    Raises:
        DocumentLoadError: If file type is not supported.

    Example:
        >>> loader = get_loader(Path("document.pdf"))
        >>> pages = loader.load(Path("document.pdf"))
    """
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        return PDFLoader(**kwargs)
    elif suffix in ['.txt', '.md', '.markdown']:
        return TextLoader(**kwargs)
    else:
        raise DocumentLoadError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: .pdf, .txt, .md, .markdown"
        )


def load_document(file_path: Path, **kwargs) -> List[LoadedPage]:
    """Convenience function to load a document.

    Automatically selects the appropriate loader based on file extension.

    Args:
        file_path: Path to document file.
        **kwargs: Additional arguments to pass to loader.

    Returns:
        List of LoadedPage objects.

    Raises:
        DocumentLoadError: If file cannot be loaded.

    Example:
        >>> pages = load_document(Path("document.pdf"))
        >>> for page in pages:
        ...     print(f"Page {page.page_number}: {len(page.text)} chars")
    """
    loader = get_loader(file_path, **kwargs)
    return loader.load(file_path)
