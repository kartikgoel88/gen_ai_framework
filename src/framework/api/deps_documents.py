"""Document and OCR processing dependencies for FastAPI.

This module provides dependency injection functions for document processors,
OCR processors, and related document handling components.
"""

from typing import Annotated

from fastapi import Depends

from ..documents.processor import DocumentProcessor
from ..documents.langchain_loader import LangChainDocProcessor
from ..documents.ocr_processor import OcrProcessor
from ..docling.processor import DoclingProcessor
from ..config import get_settings_dep, FrameworkSettings


def get_pdf_ocr_processor() -> OcrProcessor:
    """Dependency that returns the OCR processor configured for PDFs (and images).
    
    Uses PyMuPDF + pytesseract for PDFs, EasyOCR for images. Same OcrProcessor
    type is used for both PDF and image extraction.
    
    Returns:
        OcrProcessor instance with PDF-friendly defaults (dpi, min_text_len)
    """
    return OcrProcessor(pdf_dpi=300, pdf_min_text_len=10)


def get_document_processor(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
    pdf_ocr_processor: OcrProcessor = Depends(get_pdf_ocr_processor),
) -> DocumentProcessor:
    """Dependency that returns the document processor.
    
    Handles extraction from PDF, DOCX, TXT, and Excel files.
    PDFs use the injected OCR processor (PyMuPDF + pytesseract fallback) when provided.
    
    Args:
        settings: Framework settings (injected via FastAPI Depends)
        pdf_ocr_processor: OcrProcessor used for PDF extraction with OCR fallback (injected)
        
    Returns:
        DocumentProcessor instance
    """
    return DocumentProcessor(upload_dir=settings.UPLOAD_DIR, pdf_processor=pdf_ocr_processor)


def get_langchain_loader() -> LangChainDocProcessor:
    """Dependency that returns the LangChain document loader.
    
    Provides LangChain-compatible document loading for integration
    with LangChain workflows.
    
    Returns:
        LangChainDocProcessor instance
        
    Example:
        ```python
        @app.post("/documents/load")
        def load_docs(loader: LangChainDocProcessor = Depends(get_langchain_loader)):
            return loader.load("path/to/file.pdf")
        ```
    """
    return LangChainDocProcessor()


def get_ocr_processor() -> OcrProcessor:
    """Dependency that returns the OCR processor.
    
    Uses EasyOCR for image-based OCR processing (PNG, JPG, etc.).
    
    Returns:
        OcrProcessor instance
        
    Example:
        ```python
        @app.post("/ocr/extract")
        def extract_text(image_bytes: bytes, ocr: OcrProcessor = Depends(get_ocr_processor)):
            return ocr.extract_from_bytes(image_bytes)
        ```
    """
    return OcrProcessor()


def get_docling_processor() -> DoclingProcessor:
    """Dependency that returns the Docling processor.
    
    Provides layout-aware document parsing with OCR support
    for complex documents.
    
    Returns:
        DoclingProcessor instance
        
    Example:
        ```python
        @app.post("/docling/process")
        def process_doc(docling: DoclingProcessor = Depends(get_docling_processor)):
            return docling.process("path/to/file.pdf")
        ```
    """
    return DoclingProcessor()
