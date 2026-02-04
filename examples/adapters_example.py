"""Adapters Example - LangChain Integration.

This example demonstrates how to use LangChain adapters to integrate
framework LLMClient with LangChain chains and agents.

Key concepts:
- LangChainLLMAdapter: Wraps framework LLMClient as LangChain BaseChatModel
- Provider-specific adapters: Optimized for different LLM providers
- Message serialization: Converting LangChain messages to provider format
- Tool call parsing: Parsing tool calls from LLM responses
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm
from src.framework.adapters import (
    LangChainLLMAdapter,
    AdapterFactory,
    MessageSerializerFactory,
    ToolCallParserFactory,
)


def main():
    settings = get_settings()
    llm = get_llm(settings)
    
    print("=" * 60)
    print("Adapters Example")
    print("=" * 60 + "\n")
    
    # 1. Basic Adapter Usage
    print("1. Basic LangChainLLMAdapter:")
    print("-" * 60)
    adapter = LangChainLLMAdapter(llm_client=llm)
    
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Say hello in French")
    ]
    
    result = adapter.invoke(messages)
    print(f"Response: {result.content}\n")
    
    # 2. Provider-Specific Adapters
    print("2. Provider-Specific Adapters:")
    print("-" * 60)
    # Automatically selects best adapter based on provider
    provider_adapter = AdapterFactory.create(
        provider=settings.LLM_PROVIDER,
        llm_client=llm
    )
    print(f"Created {type(provider_adapter).__name__} for {settings.LLM_PROVIDER}")
    
    result = provider_adapter.invoke(messages)
    print(f"Response: {result.content}\n")
    
    # 3. Message Serialization
    print("3. Message Serialization:")
    print("-" * 60)
    serializer = MessageSerializerFactory.create_serializer("string")
    serialized = serializer.serialize(messages)
    print(f"Serialized (first 100 chars): {serialized[:100]}...\n")
    
    # JSON serializer
    json_serializer = MessageSerializerFactory.create_serializer("json")
    json_serialized = json_serializer.serialize(messages)
    print(f"JSON Serialized (first 150 chars): {json_serialized[:150]}...\n")
    
    # 4. Tool Call Parsing
    print("4. Tool Call Parsing:")
    print("-" * 60)
    parser = ToolCallParserFactory.create_parser("json")
    
    # Simulate LLM response with tool call
    tool_call_response = '{"name": "rag_search", "arguments": {"query": "Python", "top_k": 4}}'
    parsed_calls = parser.parse(tool_call_response, available_tools=["rag_search", "web_search"])
    
    if parsed_calls:
        for call in parsed_calls:
            print(f"Tool: {call.name}")
            print(f"Arguments: {call.arguments}")
            print(f"Call ID: {call.call_id}\n")
    else:
        print("No tool calls parsed\n")
    
    # 5. Using Adapter in LangChain Chain
    print("5. Using Adapter in LangChain LCEL Chain:")
    print("-" * 60)
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful translator."),
        ("human", "Translate '{text}' to Spanish")
    ])
    
    chain = prompt | adapter | StrOutputParser()
    result = chain.invoke({"text": "Hello, how are you?"})
    print(f"Translation: {result}\n")
    
    # 6. Streaming with Adapter
    print("6. Streaming with Adapter:")
    print("-" * 60)
    if hasattr(llm, "stream_invoke"):
        print("Streaming response:")
        for chunk in adapter.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
        print("\n")
    else:
        print("Streaming not supported by this LLM provider\n")


if __name__ == "__main__":
    main()
