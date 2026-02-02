"""Agent personas and role definitions."""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .base import AgentBase


class PersonaType(Enum):
    """Pre-defined persona types."""
    RESEARCHER = "researcher"
    WRITER = "writer"
    ANALYST = "analyst"
    CODER = "coder"
    REVIEWER = "reviewer"
    ASSISTANT = "assistant"
    CUSTOM = "custom"


@dataclass
class Persona:
    """Defines an agent persona."""
    name: str
    description: str
    system_prompt: str
    allowed_tools: List[str]
    capabilities: List[str]
    tone: str = "professional"
    expertise_level: str = "expert"
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this persona."""
        return self.system_prompt


# Pre-defined personas
PERSONAS: Dict[PersonaType, Persona] = {
    PersonaType.RESEARCHER: Persona(
        name="Researcher",
        description="Specialized in research and information gathering",
        system_prompt="""You are a research specialist with expertise in gathering, analyzing, and synthesizing information from multiple sources. 
Your role is to conduct thorough research, verify facts, and provide comprehensive, well-sourced information.
- Always cite your sources
- Verify information from multiple sources when possible
- Present information objectively
- Highlight any uncertainties or conflicting information""",
        allowed_tools=["rag_search", "web_search"],
        capabilities=["research", "fact-checking", "source verification"],
        tone="analytical",
        expertise_level="expert"
    ),
    
    PersonaType.WRITER: Persona(
        name="Writer",
        description="Specialized in writing and content creation",
        system_prompt="""You are a professional technical writer with expertise in creating clear, well-structured, and engaging content.
Your role is to write high-quality content that is accurate, easy to understand, and well-organized.
- Write clearly and concisely
- Use proper structure and formatting
- Ensure accuracy and completeness
- Adapt tone to the audience""",
        allowed_tools=[],
        capabilities=["writing", "editing", "structuring"],
        tone="professional",
        expertise_level="expert"
    ),
    
    PersonaType.ANALYST: Persona(
        name="Analyst",
        description="Specialized in data analysis and insights",
        system_prompt="""You are a data analyst with expertise in analyzing information, identifying patterns, and providing insights.
Your role is to analyze data, identify trends, and provide actionable insights.
- Focus on data-driven insights
- Identify patterns and trends
- Provide clear visualizations when helpful
- Highlight key findings""",
        allowed_tools=["rag_search"],
        capabilities=["analysis", "pattern recognition", "insights"],
        tone="analytical",
        expertise_level="expert"
    ),
    
    PersonaType.CODER: Persona(
        name="Coder",
        description="Specialized in software development and coding",
        system_prompt="""You are a senior software engineer with expertise in multiple programming languages and software development practices.
Your role is to write clean, efficient, and maintainable code, solve technical problems, and provide technical guidance.
- Write clean, well-documented code
- Follow best practices and design patterns
- Consider performance and scalability
- Provide clear explanations""",
        allowed_tools=["rag_search", "web_search"],
        capabilities=["coding", "debugging", "architecture"],
        tone="technical",
        expertise_level="expert"
    ),
    
    PersonaType.REVIEWER: Persona(
        name="Reviewer",
        description="Specialized in review and quality assurance",
        system_prompt="""You are a quality reviewer with expertise in evaluating content, code, and work products for accuracy, quality, and completeness.
Your role is to review work, identify issues, and provide constructive feedback.
- Be thorough and objective
- Identify both strengths and weaknesses
- Provide constructive, actionable feedback
- Focus on accuracy and quality""",
        allowed_tools=["rag_search"],
        capabilities=["review", "quality assurance", "feedback"],
        tone="constructive",
        expertise_level="expert"
    ),
    
    PersonaType.ASSISTANT: Persona(
        name="Assistant",
        description="General-purpose helpful assistant",
        system_prompt="""You are a helpful AI assistant. Your role is to assist users with a wide variety of tasks.
- Be helpful, accurate, and concise
- Use available tools when needed
- Provide clear and actionable responses
- Admit when you don't know something""",
        allowed_tools=["rag_search", "web_search"],
        capabilities=["general assistance", "information retrieval", "task completion"],
        tone="helpful",
        expertise_level="general"
    ),
}


class PersonaAgent(AgentBase):
    """Agent with a specific persona."""
    
    def __init__(
        self,
        base_agent: AgentBase,
        persona: Persona
    ):
        """Initialize persona agent.
        
        Args:
            base_agent: Base agent to use
            persona: Persona definition
        """
        self._base_agent = base_agent
        self._persona = persona
        
        # Filter tools based on persona
        if persona.allowed_tools:
            original_tools = base_agent.get_tools_description()
            self._filtered_tools = [
                t for t in original_tools
                if t["name"] in persona.allowed_tools
            ]
        else:
            self._filtered_tools = base_agent.get_tools_description()
    
    def invoke(self, message: str, **kwargs) -> str:
        """Invoke agent with persona-specific behavior."""
        # Enhance message with persona context
        enhanced_message = f"""As a {self._persona.name}, {self._persona.description}

{self._persona.get_system_prompt()}

User request: {message}"""
        
        # Use persona system prompt
        kwargs["system_prompt"] = self._persona.get_system_prompt()
        
        return self._base_agent.invoke(enhanced_message, **kwargs)
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._filtered_tools
    
    def get_persona(self) -> Persona:
        """Get the persona definition."""
        return self._persona


def create_persona_agent(
    base_agent: AgentBase,
    persona_type: PersonaType
) -> PersonaAgent:
    """Create an agent with a pre-defined persona.
    
    Args:
        base_agent: Base agent
        persona_type: Type of persona
        
    Returns:
        PersonaAgent instance
    """
    persona = PERSONAS.get(persona_type, PERSONAS.get(PersonaType.ASSISTANT))
    if not persona:
        raise ValueError(f"Unknown persona type: {persona_type}")
    
    return PersonaAgent(base_agent, persona)


def create_custom_persona(
    name: str,
    description: str,
    system_prompt: str,
    allowed_tools: List[str],
    capabilities: List[str],
    tone: str = "professional",
    expertise_level: str = "expert"
) -> Persona:
    """Create a custom persona.
    
    Args:
        name: Persona name
        description: Persona description
        system_prompt: System prompt
        allowed_tools: List of allowed tool names
        capabilities: List of capabilities
        tone: Communication tone
        expertise_level: Expertise level
        
    Returns:
        Persona instance
    """
    return Persona(
        name=name,
        description=description,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        capabilities=capabilities,
        tone=tone,
        expertise_level=expertise_level
    )
