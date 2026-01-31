"""Adapter: wrap framework LLMClient as a LangChain ChatModel for use in LCEL chains."""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..llm.base import LLMClient


def _messages_to_prompt(messages: List[BaseMessage]) -> str:
    """Convert a list of LangChain messages to a single prompt string."""
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(m.content if isinstance(m.content, str) else str(m.content))
        elif hasattr(m, "content"):
            parts.append(f"{getattr(m, 'type', 'message')}: {m.content}")
        else:
            parts.append(str(m))
    return "\n\n".join(parts)


class LangChainLLMAdapter(BaseChatModel):
    """
    Wraps the framework LLMClient as a LangChain BaseChatModel so it can be used
    in LCEL pipelines: prompt | adapter | StrOutputParser().
    """

    llm_client: Any  # LLMClient

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "framework_llm_adapter"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _messages_to_prompt(messages)
        response = self.llm_client.invoke(prompt, **kwargs)
        message = AIMessage(content=response)
        return ChatResult(generations=[ChatGeneration(message=message)])
