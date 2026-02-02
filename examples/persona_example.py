"""Agent Personas Example.

This example demonstrates different agent personas and their behaviors.
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import (
    LangChainReActAgent,
    PersonaAgent,
    PersonaType,
    create_persona_agent,
    create_custom_persona,
    build_framework_tools
)


def main():
    # Get settings and components
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    
    # Build tools
    tools = build_framework_tools(
        rag_client=rag,
        mcp_client=None,
        enable_web_search=True
    )
    
    # Create base agent
    base_agent = LangChainReActAgent(llm=llm, tools=tools)
    
    print("=" * 60)
    print("Agent Personas Example")
    print("=" * 60 + "\n")
    
    question = "Explain RAG systems in detail"
    
    # Test different personas
    personas_to_test = [
        PersonaType.RESEARCHER,
        PersonaType.WRITER,
        PersonaType.ANALYST,
    ]
    
    for persona_type in personas_to_test:
        print(f"{persona_type.value.upper()} Persona:")
        print("-" * 60)
        
        persona_agent = create_persona_agent(base_agent, persona_type)
        persona = persona_agent.get_persona()
        
        print(f"Name: {persona.name}")
        print(f"Description: {persona.description}")
        print(f"Capabilities: {', '.join(persona.capabilities)}")
        print(f"Tone: {persona.tone}")
        print(f"Allowed Tools: {', '.join(persona.allowed_tools)}\n")
        
        # Note: Actual invocation would require API keys
        print("(Skipping actual invocation - requires API keys)\n")
    
    # Custom persona example
    print("=" * 60)
    print("Custom Persona:")
    print("=" * 60)
    
    custom_persona = create_custom_persona(
        name="Technical Writer",
        description="Specialized in writing technical documentation",
        system_prompt="You are a technical writer. Write clear, concise technical documentation.",
        allowed_tools=["rag_search"],
        capabilities=["technical writing", "documentation"],
        tone="professional",
        expertise_level="expert"
    )
    
    custom_agent = PersonaAgent(base_agent, custom_persona)
    print(f"Created custom persona: {custom_persona.name}")
    print(f"Description: {custom_persona.description}\n")


if __name__ == "__main__":
    main()
