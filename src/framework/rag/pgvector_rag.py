"""pgvector RAG backend (Postgres). Optional: pip install gen-ai-framework[vectorstore-pgvector]."""

from typing import Any, Optional

from ..embeddings.base import EmbeddingsProvider
from .base import RAGClient
from .chunking import get_text_splitter


class PgvectorRAG(RAGClient):
    """RAG client using Postgres with pgvector extension."""

    def __init__(
        self,
        embeddings: EmbeddingsProvider,
        connection_string: str,
        table_name: str = "rag_embeddings",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            self._conn_str = connection_string
            self._table = table_name
            self._execute_values = execute_values
        except ImportError as e:
            raise ImportError("pgvector backend requires: pip install psycopg2-binary pgvector") from e
        self._embeddings = embeddings
        self._splitter = get_text_splitter("recursive_character", chunk_size, chunk_overlap)
        self._init_table()

    def _conn(self):
        import psycopg2
        return psycopg2.connect(self._conn_str)

    def _init_table(self) -> None:
        try:
            from pgvector.psycopg2 import register_vector
            conn = self._conn()
            try:
                register_vector(conn)
                with conn.cursor() as cur:
                    dim = len(self._embeddings.embed_query("x"))
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._table} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            content TEXT NOT NULL,
                            metadata JSONB DEFAULT '{{}}',
                            embedding vector({dim})
                        )
                    """)
                    try:
                        cur.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{self._table.replace('.', '_')}_embedding
                            ON {self._table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
                        """)
                    except Exception:
                        pass
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        from pgvector.psycopg2 import register_vector
        import json
        conn = self._conn()
        try:
            register_vector(conn)
            rows = []
            for i, text in enumerate(texts):
                chunks = self._splitter.split_text(text)
                meta = (metadatas or [{}])[i] if (metadatas and i < len(metadatas)) else {}
                for chunk in chunks:
                    vec = self._embeddings.embed_query(chunk)
                    rows.append((chunk, json.dumps(meta), vec))
            if rows:
                with conn.cursor() as cur:
                    self._execute_values(
                        cur,
                        f"INSERT INTO {self._table} (content, metadata, embedding) VALUES %s",
                        rows,
                    )
                conn.commit()
        finally:
            conn.close()

    def retrieve(self, query: str, top_k: int = 4, **kwargs: Any) -> list[dict[str, Any]]:
        from pgvector.psycopg2 import register_vector
        import json
        conn = self._conn()
        try:
            register_vector(conn)
            qvec = self._embeddings.embed_query(query)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT content, metadata FROM {self._table} ORDER BY embedding <=> %s LIMIT %s",
                    (qvec, top_k),
                )
                rows = cur.fetchall()
            return [
                {"content": r[0], "metadata": json.loads(r[1]) if isinstance(r[1], str) else (r[1] or {})}
                for r in rows
            ]
        finally:
            conn.close()

    def query(self, question: str, llm_client: Any = None, **kwargs: Any) -> str:
        contexts = self.retrieve(question, **{k: v for k, v in kwargs.items() if k != "llm_client"})
        context_text = "\n\n".join(c["content"] for c in contexts)
        if not llm_client:
            return context_text
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        return llm_client.invoke(prompt)

    def export_corpus(self, format: str = "jsonl", **kwargs: Any) -> list[dict[str, Any]]:
        import json
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT content, metadata FROM {self._table}")
                rows = cur.fetchall()
            return [
                {"content": r[0], "metadata": json.loads(r[1]) if isinstance(r[1], str) else (r[1] or {})}
                for r in rows
            ]
        finally:
            conn.close()
