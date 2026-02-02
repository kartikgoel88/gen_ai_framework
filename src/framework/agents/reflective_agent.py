"""Agent reflection and self-correction capabilities."""

from typing import Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from .base import AgentBase


class ReflectionResult(Enum):
    """Result of reflection."""
    ACCEPTABLE = "acceptable"
    NEEDS_CORRECTION = "needs_correction"
    NEEDS_MORE_INFO = "needs_more_info"


@dataclass
class Reflection:
    """Represents agent reflection on its response."""
    result: ReflectionResult
    confidence: float  # 0.0 to 1.0
    reasoning: str
    suggestions: Optional[str] = None
    errors_found: Optional[List[str]] = None


class ReflectiveAgent(AgentBase):
    """Agent that reflects on its responses and corrects mistakes.
    
    This agent evaluates its own responses, detects errors, and
    automatically retries with corrections when needed.
    """
    
    def __init__(
        self,
        base_agent: AgentBase,
        enable_reflection: bool = True,
        min_confidence: float = 0.7,
        max_corrections: int = 2
    ):
        """Initialize reflective agent.
        
        Args:
            base_agent: Base agent to use
            enable_reflection: Enable reflection mechanism
            min_confidence: Minimum confidence threshold
            max_corrections: Maximum number of correction attempts
        """
        self._base_agent = base_agent
        self._enable_reflection = enable_reflection
        self._min_confidence = min_confidence
        self._max_corrections = max_corrections
    
    def invoke(self, message: str, **kwargs) -> str:
        """Invoke agent with reflection.
        
        Args:
            message: User message
            **kwargs: Additional arguments
            
        Returns:
            Final response (after reflection and correction if needed)
        """
        # Generate initial response
        response = self._base_agent.invoke(message, **kwargs)
        
        if not self._enable_reflection:
            return response
        
        # Reflect on response
        correction_count = 0
        
        while correction_count < self._max_corrections:
            reflection = self._reflect(response, message)
            
            if reflection.result == ReflectionResult.ACCEPTABLE:
                return response
            
            if reflection.result == ReflectionResult.NEEDS_CORRECTION:
                # Correct the response
                response = self._correct(response, reflection, message)
                correction_count += 1
            else:
                # Needs more info - try to get it
                enhanced_message = self._enhance_message(message, reflection)
                response = self._base_agent.invoke(enhanced_message, **kwargs)
                correction_count += 1
        
        return response
    
    def _reflect(self, response: str, original_message: str) -> Reflection:
        """Reflect on the agent's response."""
        reflection_prompt = f"""Evaluate the following response to the user's question.

User Question: {original_message}

Agent Response: {response}

Evaluate:
1. Is the response accurate and complete?
2. Does it address the user's question?
3. Are there any errors or inconsistencies?
4. What is your confidence level (0.0 to 1.0)?

Provide your evaluation in this format:
CONFIDENCE: [0.0-1.0]
RESULT: [ACCEPTABLE|NEEDS_CORRECTION|NEEDS_MORE_INFO]
REASONING: [Your reasoning]
SUGGESTIONS: [If correction needed, provide suggestions]
ERRORS: [List any errors found]"""
        
        reflection_text = self._base_agent.invoke(reflection_prompt)
        
        # Parse reflection
        return self._parse_reflection(reflection_text)
    
    def _parse_reflection(self, reflection_text: str) -> Reflection:
        """Parse reflection text into Reflection object."""
        lines = reflection_text.split("\n")
        
        confidence = 0.5
        result = ReflectionResult.ACCEPTABLE
        reasoning = ""
        suggestions = None
        errors = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("RESULT:"):
                result_str = line.split(":", 1)[1].strip().upper()
                if "NEEDS_CORRECTION" in result_str:
                    result = ReflectionResult.NEEDS_CORRECTION
                elif "NEEDS_MORE_INFO" in result_str:
                    result = ReflectionResult.NEEDS_MORE_INFO
                else:
                    result = ReflectionResult.ACCEPTABLE
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("SUGGESTIONS:"):
                suggestions = line.split(":", 1)[1].strip()
            elif line.startswith("ERRORS:"):
                errors_str = line.split(":", 1)[1].strip()
                errors = [e.strip() for e in errors_str.split(",") if e.strip()]
        
        return Reflection(
            result=result,
            confidence=confidence,
            reasoning=reasoning,
            suggestions=suggestions,
            errors_found=errors
        )
    
    def _correct(self, response: str, reflection: Reflection, original_message: str) -> str:
        """Correct the response based on reflection."""
        correction_prompt = f"""Correct the following response based on the evaluation.

Original Question: {original_message}

Original Response: {response}

Evaluation:
- Confidence: {reflection.confidence}
- Issues Found: {reflection.reasoning}
- Suggestions: {reflection.suggestions}
- Errors: {reflection.errors_found or 'None'}

Provide a corrected response that addresses all the issues identified."""
        
        return self._base_agent.invoke(correction_prompt)
    
    def _enhance_message(self, message: str, reflection: Reflection) -> str:
        """Enhance the message with additional context."""
        return f"""{message}

Note: The previous response was incomplete. Please provide more comprehensive information.
Additional context needed: {reflection.suggestions or reflection.reasoning}"""
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._base_agent.get_tools_description()
