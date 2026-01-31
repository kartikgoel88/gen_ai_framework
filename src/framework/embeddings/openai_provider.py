"""OpenAI embeddings provider."""

from typing import List, Optional

from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings

from .base import EmbeddingsProvider


class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    """OpenAI embeddings (LangChain wrapper)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
    ):
        self._embeddings = LangChainOpenAIEmbeddings(
            model=model,
            openai_api_key=openai_api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)
