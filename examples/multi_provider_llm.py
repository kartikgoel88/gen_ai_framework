"""Multi-Provider LLM Example.

This example demonstrates how to:
1. Use the LLM provider registry
2. Switch between different LLM providers
3. Use the same interface for all providers
"""

from src.framework.llm.registry import LLMProviderRegistry


def main():
    # List available providers
    print("Available providers:", LLMProviderRegistry.list_providers())
    
    # Example: OpenAI (requires OPENAI_API_KEY)
    try:
        llm_openai = LLMProviderRegistry.create(
            provider="openai",
            api_key="your-openai-api-key",
            model="gpt-4",
            temperature=0.7,
        )
        response = llm_openai.invoke("Hello, how are you?")
        print(f"OpenAI: {response[:50]}...")
    except Exception as e:
        print(f"OpenAI error: {e}")
    
    # Example: Grok (requires XAI_API_KEY)
    try:
        llm_grok = LLMProviderRegistry.create(
            provider="grok",
            api_key="your-xai-api-key",
            model="grok-2",
            temperature=0.7,
        )
        response = llm_grok.invoke("Hello, how are you?")
        print(f"Grok: {response[:50]}...")
    except Exception as e:
        print(f"Grok error: {e}")
    
    # Example: Adding a custom provider
    from src.framework.llm.base import LLMClient
    
    class CustomLLM(LLMClient):
        def invoke(self, prompt: str, **kwargs):
            return f"Custom response to: {prompt}"
        
        def invoke_structured(self, prompt: str, **kwargs):
            return {"response": self.invoke(prompt, **kwargs)}
    
    @LLMProviderRegistry.register("custom")
    def create_custom(api_key: str, model: str, temperature: float, **kwargs):
        return CustomLLM()
    
    llm_custom = LLMProviderRegistry.create(
        provider="custom",
        api_key="dummy",
        model="custom-model",
        temperature=0.7,
    )
    response = llm_custom.invoke("Test")
    print(f"Custom: {response}")


if __name__ == "__main__":
    main()
