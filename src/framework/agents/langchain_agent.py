"""LangChain-based agent (OpenAI tools agent + executor)."""

from typing import Any, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from .base import AgentBase


class LangChainReActAgent(AgentBase):
    """Agent using LangChain create_openai_tools_agent + AgentExecutor."""

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: list[BaseTool],
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt or (
            "You are a helpful AI assistant. Use the available tools when needed to answer questions. "
            "Be concise and accurate."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(llm=self._llm, tools=self._tools, prompt=prompt)
        self._executor = AgentExecutor(
            agent=agent,
            tools=self._tools,
            verbose=verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )

    def invoke(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        chat_history: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        inputs = {
            "input": message,
            "chat_history": chat_history or [],
        }
        result = self._executor.invoke(inputs, **kwargs)
        return result.get("output", str(result))

    def get_tools_description(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools
        ]
