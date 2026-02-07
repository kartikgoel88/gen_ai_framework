"""Base document processor: shared interface for all document/OCR processors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from .types import ExtractResult


class BaseDocumentProcessor(ABC):
    """
    Abstract base for document and image processors that extract text from files.

    Implementations: DocumentProcessor (multi-format), OcrProcessor (PDF + images),
    DoclingProcessor (layout-aware). All return the
    canonical ExtractResult (text, metadata, error) so they can be used
    interchangeably where a single processor interface is expected.
    """

    @abstractmethod
    def extract(self, file_path: Union[str, Path]) -> ExtractResult:
        """
        Extract text from a file. Callers can use any processor via this interface.

        Args:
            file_path: Path to the document or image file.

        Returns:
            ExtractResult with .text, .metadata, and optional .error.
        """
        ...
