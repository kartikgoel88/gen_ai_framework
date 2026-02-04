"""Pipeline Example - Multi-Step Chain Composition.

This example demonstrates how to compose multiple chains into a pipeline
for complex multi-step workflows.

Use cases:
- Extract → Summarize → Classify
- Retrieve → Extract → Structure
- Translate → Summarize → Extract
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.chains import (
    PromptChain,
    RAGChain,
    ExtractionChain,
    SummarizationChain,
    ClassificationChain,
    StructuredChain,
    Pipeline,
    PipelineStep,
)


def main():
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    
    print("=" * 60)
    print("Pipeline Example - Multi-Step Chains")
    print("=" * 60 + "\n")
    
    # Example text for processing
    sample_text = """
    John Smith, a senior software engineer at Microsoft, has been working on 
    artificial intelligence projects for the past 5 years. He recently published 
    a research paper on machine learning optimization techniques. The paper 
    received positive feedback from the AI research community and was cited 
    multiple times. John is excited about the future of AI and plans to continue 
    his research in this field.
    """
    
    # 1. Simple Pipeline: Extract → Summarize
    print("1. Simple Pipeline (Extract → Summarize):")
    print("-" * 60)
    
    extract_chain = ExtractionChain(
        llm=llm,
        schema="person names, organizations, topics"
    )
    
    summarize_chain = SummarizationChain(llm=llm)
    
    pipeline = Pipeline(steps=[
        PipelineStep(step_id="extract", chain=extract_chain, output_key="entities"),
        PipelineStep(step_id="summarize", chain=summarize_chain, output_key="summary")
    ])
    
    result = pipeline.invoke({"text": sample_text})
    print(f"Entities: {result.get('entities', 'N/A')}")
    print(f"Summary: {result.get('summary', 'N/A')}\n")
    
    # 2. Complex Pipeline: Extract → Summarize → Classify
    print("2. Complex Pipeline (Extract → Summarize → Classify):")
    print("-" * 60)
    
    classify_chain = ClassificationChain(
        llm=llm,
        labels=["technical", "biographical", "news", "research"]
    )
    
    complex_pipeline = Pipeline(steps=[
        PipelineStep(step_id="extract", chain=extract_chain, output_key="entities"),
        PipelineStep(step_id="summarize", chain=summarize_chain, output_key="summary"),
        PipelineStep(step_id="classify", chain=classify_chain, output_key="category")
    ])
    
    result = complex_pipeline.invoke({"text": sample_text})
    print(f"Entities: {result.get('entities', 'N/A')}")
    print(f"Summary: {result.get('summary', 'N/A')}")
    print(f"Category: {result.get('category', 'N/A')}\n")
    
    # 3. RAG Pipeline: Retrieve → Extract → Structure
    print("3. RAG Pipeline (Retrieve → Extract → Structure):")
    print("-" * 60)
    
    # Add documents to RAG
    rag.add_documents(
        texts=[sample_text],
        metadatas=[{"source": "example.txt"}]
    )
    
    rag_chain = RAGChain(llm=llm, rag=rag, top_k=2)
    
    structured_chain = StructuredChain(
        llm=llm,
        template="Extract structured information from: {text}"
    )
    
    # Create a custom chain that combines RAG and extraction
    class RAGExtractChain:
        def __init__(self, rag_chain, extract_chain):
            self.rag_chain = rag_chain
            self.extract_chain = extract_chain
        
        def invoke(self, inputs, **kwargs):
            # First get RAG answer
            rag_result = self.rag_chain.invoke(inputs, **kwargs)
            # Then extract from the answer
            extract_result = self.extract_chain.invoke({"text": rag_result}, **kwargs)
            return extract_result
    
    rag_extract = RAGExtractChain(rag_chain, extract_chain)
    
    result = rag_extract.invoke({"question": "Who is mentioned in the documents?"})
    print(f"Extracted from RAG: {result}\n")
    
    # 4. Conditional Pipeline (using different chains based on input)
    print("4. Conditional Processing:")
    print("-" * 60)
    
    def process_text(text: str):
        """Process text through different pipelines based on content."""
        # Simple heuristic: if text is long, summarize first
        if len(text) > 500:
            pipeline = Pipeline(steps=[
                PipelineStep(step_id="summarize", chain=summarize_chain, output_key="summary"),
                PipelineStep(step_id="extract", chain=extract_chain, output_key="entities")
            ])
        else:
            pipeline = Pipeline(steps=[
                PipelineStep(step_id="extract", chain=extract_chain, output_key="entities"),
                PipelineStep(step_id="classify", chain=classify_chain, output_key="category")
            ])
        
        return pipeline.invoke({"text": text})
    
    short_text = "John works at Google."
    long_text = sample_text
    
    print("Short text processing:")
    result = process_text(short_text)
    print(f"Result keys: {list(result.keys())}\n")
    
    print("Long text processing:")
    result = process_text(long_text)
    print(f"Result keys: {list(result.keys())}\n")
    
    # 5. Pipeline with Custom Output Key
    print("5. Pipeline with Custom Output Key:")
    print("-" * 60)
    
    custom_pipeline = Pipeline(
        steps=[
            PipelineStep(step_id="extract", chain=extract_chain, output_key="extracted_entities"),
            PipelineStep(step_id="summarize", chain=summarize_chain, output_key="text_summary"),
        ],
        final_output_key="text_summary"  # Return only the summary
    )
    
    result = custom_pipeline.invoke({"text": sample_text})
    print(f"Final output (summary only): {result}\n")


if __name__ == "__main__":
    main()
