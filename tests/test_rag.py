"""Tests for RAG: ChromaRAG add_documents, retrieve, query, export_corpus."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.framework.rag.base import RAGClient
from src.framework.rag.chroma_rag import ChromaRAG
from src.framework.rag.chunking import get_text_splitter


class MockEmbeddings:
    """Mock embeddings: deterministic 384-dim vectors (Chroma accepts any consistent size)."""

    _dim = 384

    def embed_documents(self, texts):
        return [[hash(t) % 1000 / 1000.0] * self._dim for t in texts]

    def embed_query(self, text: str):
        return [hash(text) % 1000 / 1000.0] * self._dim


@pytest.fixture
def mock_embeddings():
    return MockEmbeddings()


@pytest.fixture
def chroma_rag(mock_embeddings):
    with tempfile.TemporaryDirectory() as tmp:
        yield ChromaRAG(
            persist_directory=tmp,
            embeddings=mock_embeddings,
            chunk_size=100,
            chunk_overlap=20,
            use_hybrid=False,
        )


def test_rag_add_documents_retrieve(chroma_rag: RAGClient):
    chroma_rag.add_documents(
        ["First chunk of text about policy.", "Second chunk about leave."],
        metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    )
    chunks = chroma_rag.retrieve("policy", top_k=2)
    assert len(chunks) <= 2
    assert all("content" in c and "metadata" in c for c in chunks)
    contents = [c["content"] for c in chunks]
    assert any("policy" in c.lower() for c in contents) or len(contents) > 0


def test_rag_query_without_llm(chroma_rag: RAGClient):
    chroma_rag.add_documents(["Answer is 42."])
    out = chroma_rag.query("What is the answer?", llm_client=None)
    assert "42" in out


def test_rag_query_with_llm(chroma_rag: RAGClient):
    chroma_rag.add_documents(["The capital is Paris."])
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value="Paris")
    out = chroma_rag.query("What is the capital?", llm_client=mock_llm)
    assert out == "Paris"
    mock_llm.invoke.assert_called_once()


def test_rag_export_corpus(chroma_rag: RAGClient):
    chroma_rag.add_documents(["Doc A", "Doc B"], metadatas=[{"id": "a"}, {"id": "b"}])
    corpus = chroma_rag.export_corpus()
    assert isinstance(corpus, list)
    assert len(corpus) >= 2
    for item in corpus:
        assert "content" in item and "metadata" in item


def test_chunking_strategy_sentence():
    splitter = get_text_splitter("sentence", chunk_size=500, chunk_overlap=50)
    text = "First sentence. Second sentence. Third sentence."
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1
    assert "".join(chunks).replace(" ", "").replace("\n", "") == text.replace(" ", "")
