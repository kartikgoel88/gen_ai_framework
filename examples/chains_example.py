"""Chains Example - All Chain Types.

This example demonstrates how to use different chain types:
- Prompt Chain
- RAG Chain
- Structured Chain
- Summarization Chain
- Classification Chain
- Extraction Chain
- Pipeline (Multi-step chains)
- LangChain Integration
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.chains import (
    PromptChain,
    RAGChain,
    StructuredChain,
    SummarizationChain,
    ClassificationChain,
    ExtractionChain,
    Pipeline,
    PipelineStep,
    build_langchain_prompt_chain,
    build_langchain_rag_chain,
)


def main():
    # Get settings and components
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    
    print("=" * 60)
    print("Chains Examples")
    print("=" * 60 + "\n")
    
    # 1. Prompt Chain
    print("1. Prompt Chain:")
    print("-" * 60)
    prompt_chain = PromptChain(llm=llm, template="Translate to French: {text}")
    result = prompt_chain.invoke({"text": "Hello, how are you?"})
    print(f"Result: {result}\n")
    
    # 2. RAG Chain
    print("2. RAG Chain:")
    print("-" * 60)
    # First, add some documents
    rag.add_documents(
        texts=["Python is a programming language. It's easy to learn."],
        metadatas=[{"source": "doc1.txt"}]
    )
    rag_chain = RAGChain(llm=llm, rag=rag, top_k=2)
    result = rag_chain.invoke({"question": "What is Python?"})
    print(f"Result: {result}\n")
    
    # 3. Structured Chain
    print("3. Structured Chain:")
    print("-" * 60)
    structured_chain = StructuredChain(
        llm=llm,
        template="Extract information from: {text}"
    )
    result = structured_chain.invoke({
        "text": "John Doe, age 30, works as a software engineer at Google."
    })
    print(f"Result: {result}\n")
    
    # 4. Summarization Chain
    print("4. Summarization Chain:")
    print("-" * 60)
    summarization_chain = SummarizationChain(llm=llm)
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of achieving its goals.
    """
    result = summarization_chain.invoke({"text": long_text})
    print(f"Result: {result}\n")
    
    # 5. Classification Chain
    print("5. Classification Chain:")
    print("-" * 60)
    classification_chain = ClassificationChain(
        llm=llm,
        labels=["positive", "negative", "neutral"]
    )
    result = classification_chain.invoke({
        "text": "I love this product! It's amazing!"
    })
    print(f"Result: {result}\n")
    
    # 6. Extraction Chain
    print("6. Extraction Chain:")
    print("-" * 60)
    extraction_chain = ExtractionChain(
        llm=llm,
        schema="person names, locations, organizations"
    )
    result = extraction_chain.invoke({
        "text": "John works at Google in Mountain View, California."
    })
    print(f"Result: {result}\n")
    
    # 7. Pipeline (Multi-step Chain)
    print("7. Pipeline (Multi-step Chain):")
    print("-" * 60)
    # Create a pipeline: Extract → Summarize → Classify
    extract_step = PipelineStep(
        step_id="extract",
        chain=extraction_chain,
        output_key="extracted"
    )
    summarize_step = PipelineStep(
        step_id="summarize",
        chain=summarization_chain,
        output_key="summary"
    )
    classify_step = PipelineStep(
        step_id="classify",
        chain=classification_chain,
        output_key="classification"
    )
    
    pipeline = Pipeline(steps=[extract_step, summarize_step, classify_step])
    result = pipeline.invoke({
        "text": "John Doe, a software engineer at Google, loves working on AI projects. He finds the work exciting and rewarding."
    })
    print(f"Extracted: {result.get('extracted', 'N/A')}")
    print(f"Summary: {result.get('summary', 'N/A')}")
    print(f"Classification: {result.get('classification', 'N/A')}\n")
    
    # 8. LangChain Integration
    print("8. LangChain Integration:")
    print("-" * 60)
    # Build LangChain LCEL chain
    langchain_chain = build_langchain_prompt_chain(
        llm=llm,
        template="Translate '{text}' to Spanish",
        input_variables=["text"]
    )
    result = langchain_chain.invoke({"text": "Hello, world!"})
    print(f"Result: {result}\n")
    
    # LangChain RAG chain
    langchain_rag_chain = build_langchain_rag_chain(
        llm=llm,
        rag=rag,
        top_k=2
    )
    result = langchain_rag_chain.invoke({"question": "What is Python?"})
    print(f"RAG Result: {result}\n")


if __name__ == "__main__":
    main()
