"""Refactored LangChain adapter - ChatModel layer only.

This adapter follows LangChain's mental model:
- ChatModel responsibility: "What did the LLM say?"
- Agent responsibility: "What do we do with that?"

IMPORTANT: Since we're using LangChain's create_agent, it expects AIMessage with tool_calls.
The adapter's job is to ensure tool_calls are in the response format LangChain expects.

The adapter:
1. Converts LangChain messages to provider format
2. Calls the LLM (with tools bound if native support exists)
3. Returns LangChain messages with tool_calls in LangChain format

Minimal parsing is acceptable IF:
- LLM doesn't natively support tool calling
- We need to convert text responses to tool_calls format for LangChain's agent
- This is just format conversion, not interpretation/validation/execution

The adapter does NOT:
- Validate tool calls (LangChain agent does this)
- Execute tools (LangChain agent does this)
- Handle tool errors (LangChain agent does this)
- Select tool strategies (agent's job)
"""

from typing import Any, List, Optional

from pydantic import ConfigDict
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool

from ..llm.base import LLMClient
from .message_serializer import MessageSerializerFactory


class LangChainLLMAdapter(BaseChatModel):
    """
    Thin adapter wrapping framework LLMClient as LangChain BaseChatModel.
    
    Responsibility: Convert messages <-> LLM calls. That's it.
    """
    
    llm_client: Any  # LLMClient
    bound_tools: Optional[List[BaseTool]] = None  # Tools bound via bind_tools()
    serializer_type: str = "string"  # "string", "json", "native"
    use_native_tool_calling: bool = False  # Try native tool calling if available

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(
        self,
        llm_client: Any,
        bound_tools: Optional[List[BaseTool]] = None,
        serializer_type: str = "string",
        use_native_tool_calling: bool = False,
        **kwargs
    ):
        # Pass all fields to super().__init__ for Pydantic validation
        super().__init__(
            llm_client=llm_client,
            bound_tools=bound_tools,
            serializer_type=serializer_type,
            use_native_tool_calling=use_native_tool_calling,
            **kwargs
        )

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
        """
        Generate response from LLM.
        
        ONLY responsibility: Call LLM and return what it said.
        No tool call parsing, validation, or error recovery here.
        """
        try:
            # Serialize messages to provider format
            serializer = MessageSerializerFactory.create_serializer(self.serializer_type)
            
            # If native tool calling is available and tools are bound, use it
            if (self.use_native_tool_calling and 
                self.bound_tools and 
                hasattr(self.llm_client, "invoke_with_tools")):
                try:
                    tool_definitions = _convert_tools_to_definitions(self.bound_tools)
                    prompt = serializer.serialize(messages, tool_definitions)
                    
                    # Call LLM with native tool calling
                    tool_call_response = self.llm_client.invoke_with_tools(
                        prompt,
                        tool_definitions,
                        **kwargs
                    )
                    
                    # Convert ToolCallResponse to LangChain AIMessage
                    # If LLM natively returned tool calls, include them
                    from .tool_utils import convert_tool_call_to_langchain_format
                    tool_calls = []
                    if tool_call_response.has_tool_calls():
                        tool_calls = [
                            convert_tool_call_to_langchain_format(tc)
                            for tc in tool_call_response.tool_calls
                        ]
                    
                    # Only include tool_calls if we have any (Pydantic v2 requires list, not None)
                    if tool_calls:
                        message = AIMessage(
                            content=tool_call_response.content,
                            tool_calls=tool_calls
                        )
                    else:
                        message = AIMessage(content=tool_call_response.content)
                    
                    return ChatResult(generations=[ChatGeneration(message=message)])
                except NotImplementedError:
                    # Fall back to regular invoke if native tool calling not supported
                    pass
            
            # Standard flow: serialize and invoke
            # Include tools in prompt so LLM knows about them (for text-based tool calling)
            tool_definitions = _convert_tools_to_definitions(self.bound_tools) if self.bound_tools else []
            prompt = serializer.serialize(messages, tool_definitions)
            
            response = self.llm_client.invoke(prompt, **kwargs)
            
            if not response:
                response = ""
            if not isinstance(response, str):
                response = str(response)
            
            # Minimal parsing: Convert text response to tool_calls format if LLM returned JSON tool calls
            # This is format conversion for LangChain's agent, not interpretation
            tool_calls = []
            message_content = response
            
            if self.bound_tools and response.strip().startswith("{"):
                # LLM might have returned a JSON tool call
                tool_call = self._try_parse_json_tool_call(response)
                if tool_call:
                    tool_calls.append(tool_call)
                    message_content = ""
            
            # Return message with tool_calls if found (LangChain agent will validate/execute)
            # Only include tool_calls if we have any (Pydantic v2 requires list, not None)
            if tool_calls:
                message = AIMessage(
                    content=message_content,
                    tool_calls=tool_calls
                )
            else:
                message = AIMessage(content=message_content)
            return ChatResult(generations=[ChatGeneration(message=message)])
            
        except NotImplementedError:
            raise NotImplementedError(
                f"LLM client {type(self.llm_client).__name__} does not support invoke(). "
                "Make sure your LLM provider implements the invoke() method."
            )
        except Exception as e:
            raise RuntimeError(f"Error calling LLM client: {e}") from e
    
    def _try_parse_json_tool_call(self, response: str) -> Optional[dict]:
        """Try to parse a JSON tool call from response."""
        try:
            import json
            tool_call_data = json.loads(response.strip())
            if isinstance(tool_call_data, dict) and "name" in tool_call_data:
                from .tool_utils import convert_tool_call_to_langchain_format
                return convert_tool_call_to_langchain_format(
                    tool_call_data,
                    tool_name=tool_call_data.get("name"),
                    tool_args=tool_call_data.get("arguments") or tool_call_data.get("args", {})
                )
        except json.JSONDecodeError:
            pass
        return None
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Stream response tokens. Falls back to non-streaming if not supported."""
        try:
            serializer = MessageSerializerFactory.create_serializer(self.serializer_type)
            prompt = serializer.serialize(messages, tools=None)
            
            if hasattr(self.llm_client, 'stream_invoke'):
                try:
                    for chunk in self.llm_client.stream_invoke(prompt, **kwargs):
                        yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
                    return
                except NotImplementedError:
                    pass  # Fall back to non-streaming
            
            # Fallback to non-streaming
            result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            if result.generations:
                content = result.generations[0].message.content if hasattr(result.generations[0].message, 'content') else str(result.generations[0].message)
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))
        except NotImplementedError as e:
            raise NotImplementedError(
                f"Streaming not supported by LLM client {type(self.llm_client).__name__}. "
                "The client's stream_invoke() method is not implemented."
            ) from e
    
    def bind_tools(
        self,
        tools: Any,
        **kwargs: Any,
    ) -> "LangChainLLMAdapter":
        """
        Bind tools to the model. Returns a new instance with tools bound.
        
        This only stores tools for native tool calling. The agent handles
        tool call parsing and execution.
        """
        # Normalize tools to list
        if isinstance(tools, (BaseTool, dict)):
            tools_list = [tools]
        else:
            tools_list = list(tools)
        
        # Create a copy with tools bound
        new_instance = self.model_copy(update={"bound_tools": tools_list})
        return new_instance


# Import shared utility
from .tool_utils import convert_tools_to_definitions as _convert_tools_to_definitions
from .tool_utils import convert_tool_call_to_langchain_format
