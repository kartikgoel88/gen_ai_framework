"""LangChain document loaders: load files into LangChain Document objects."""

from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader

from .base import BaseDocumentProcessor
from .types import ExtractResult


class LangChainDocProcessor(BaseDocumentProcessor):
    """Load documents using LangChain loaders; return unified text + metadata or Document list."""

    def __init__(self):
        pass

    def extract(self, file_path: Union[str, Path]) -> ExtractResult:
        """Extract text (base interface). Delegates to load_as_result."""
        return self.load_as_result(file_path)

    def load(self, file_path: str | Path) -> List[Document]:
        """Load file into list of LangChain Documents (one per page/sheet etc.)."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        loader = None
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in (".doc", ".docx"):
            loader = Docx2txtLoader(str(path))
        elif suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix == ".csv":
            loader = CSVLoader(str(path))
        else:
            # Fallback: try text
            try:
                loader = TextLoader(str(path), encoding="utf-8")
            except Exception:
                return []
        if loader is None:
            return []
        try:
            return loader.load()
        except Exception:
            return []

    def load_as_result(self, file_path: str | Path) -> ExtractResult:
        """Load file and return as ExtractResult (single text + metadata)."""
        docs = self.load(file_path)
        if not docs:
            return ExtractResult.error_result("Unsupported or empty file")
        texts = [d.page_content for d in docs]
        meta = docs[0].metadata if docs else {}
        return ExtractResult(
            text="\n\n".join(texts),
            metadata={"source": str(file_path), "num_chunks": len(docs), **meta},
        )
