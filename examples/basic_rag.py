"""Basic RAG Example.

This example demonstrates how to:
1. Create a RAG client
2. Ingest documents
3. Query the knowledge base
"""

from src.framework.config import get_settings
from src.framework.rag import ChromaRAG
from src.framework.embeddings import OpenAIEmbeddingsProvider


def main():
    # Get settings
    settings = get_settings()
    
    # Create embeddings provider
    embeddings = OpenAIEmbeddingsProvider(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    
    # Create RAG client
    rag = ChromaRAG(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embeddings=embeddings,
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    # Ingest documents
    documents = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "RAG combines retrieval and generation for better answers.",
    ]
    
    rag.add_documents(
        texts=documents,
        metadatas=[
            {"source": "doc1.txt"},
            {"source": "doc2.txt"},
            {"source": "doc3.txt"},
        ],
    )
    
    # Query
    question = "What is Python?"
    results = rag.retrieve(question, top_k=2)
    
    print(f"Question: {question}")
    print("\nRetrieved chunks:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('content', '')[:100]}...")
    
    # Query with LLM (requires LLM client)
    from src.framework.api.deps import get_llm
    llm = get_llm(settings)
    answer = rag.query(question, llm_client=llm)
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
