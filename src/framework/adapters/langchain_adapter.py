"""Adapter: wrap framework LLMClient as a LangChain ChatModel for use in LCEL chains."""

from typing import Any, List, Optional, Union

from pydantic import ConfigDict
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool

from ..llm.base import LLMClient
from ..llm.tool_calling import ToolCall
from .tool_call_parser import ToolCallParserFactory
from .message_serializer import MessageSerializerFactory
from .errors import ToolCallParseError
from .error_recovery import ErrorRecoveryManager
from .tool_utils import convert_tools_to_definitions, convert_tool_call_to_langchain_format


class LangChainLLMAdapter(BaseChatModel):
    """
    Wraps the framework LLMClient as a LangChain BaseChatModel so it can be used
    in LCEL pipelines: prompt | adapter | StrOutputParser().
    
    Supports multiple serialization and parsing strategies for better compatibility
    with different LLM providers.
    """

    llm_client: Any  # LLMClient
    bound_tools: Optional[List[BaseTool]] = None  # Tools bound via bind_tools()
    serializer_type: str = "string"  # "string", "json", "native"
    parser_type: str = "json"  # "json", "structured", "native"
    use_native_tool_calling: bool = False  # Try native tool calling if available

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(
        self,
        llm_client: Any,
        bound_tools: Optional[List[BaseTool]] = None,
        serializer_type: str = "string",
        parser_type: str = "json",
        use_native_tool_calling: bool = False,
        **kwargs
    ):
        # Pass all fields to super().__init__ for Pydantic validation
        super().__init__(
            llm_client=llm_client,
            bound_tools=bound_tools,
            serializer_type=serializer_type,
            parser_type=parser_type,
            use_native_tool_calling=use_native_tool_calling,
            **kwargs
        )
        self._recovery_manager = ErrorRecoveryManager()

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
        try:
            # Try native tool calling first if available
            result = self._try_native_tool_calling(messages, **kwargs)
            if result:
                return result
            
            # Standard flow: serialize and invoke
            result = self._invoke_standard_flow(messages, **kwargs)
            return result
            
        except NotImplementedError:
            raise NotImplementedError(
                f"LLM client {type(self.llm_client).__name__} does not support invoke(). "
                "Make sure your LLM provider implements the invoke() method."
            )
        except Exception as e:
            raise RuntimeError(f"Error calling LLM client: {e}") from e
    
    def _try_native_tool_calling(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> Optional[ChatResult]:
        """Try to use native tool calling if available and enabled."""
        if not (self.use_native_tool_calling and self.bound_tools and 
                hasattr(self.llm_client, "invoke_with_tools")):
            return None
        
        try:
            tool_definitions = convert_tools_to_definitions(self.bound_tools)
            serializer = MessageSerializerFactory.create_serializer(self.serializer_type)
            prompt = serializer.serialize(messages, tool_definitions)
            
            tool_call_response = self.llm_client.invoke_with_tools(
                prompt,
                tool_definitions,
                **kwargs
            )
            
            # Convert ToolCall objects to LangChain format
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
            return None
    
    def _invoke_standard_flow(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> ChatResult:
        """Standard invocation flow with optional tool call parsing."""
        tool_definitions = convert_tools_to_definitions(self.bound_tools) if self.bound_tools else []
        serializer = MessageSerializerFactory.create_serializer(self.serializer_type)
        prompt = serializer.serialize(messages, tool_definitions)
        
        response = self.llm_client.invoke(prompt, **kwargs)
        response = self._normalize_response(response)
        
        # Parse tool calls if tools are bound
        tool_calls, message_content = self._parse_tool_calls(response, tool_definitions)
        
        # Create AIMessage - only include tool_calls if we have any (Pydantic v2 requires list, not None)
        if tool_calls:
            message = AIMessage(
                content=message_content,
                tool_calls=tool_calls
            )
        else:
            message = AIMessage(content=message_content)
        
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _normalize_response(self, response: Any) -> str:
        """Normalize LLM response to string."""
        if not response:
            return ""
        if not isinstance(response, str):
            return str(response)
        return response
    
    def _parse_tool_calls(
        self,
        response: str,
        tool_definitions: List
    ) -> tuple[List[dict], str]:
        """Parse tool calls from response, return (tool_calls, remaining_content)."""
        if not tool_definitions:
            return [], response
        
        tool_calls = []
        message_content = response
        
        try:
            parser = ToolCallParserFactory.create_parser(self.parser_type)
            available_tool_names = [td.name for td in tool_definitions]
            parsed_tool_calls = parser.parse(response, available_tool_names)
            
            # Convert to LangChain format
            for tc in parsed_tool_calls:
                tool_calls.append(convert_tool_call_to_langchain_format(tc))
            
            # If we found tool calls, clear content
            if tool_calls:
                message_content = ""
                
        except Exception as parse_error:
            # Try error recovery
            tool_calls, message_content = self._recover_tool_calls(parse_error, response)
        
        return tool_calls, message_content
    
    def _recover_tool_calls(
        self,
        parse_error: Exception,
        response: str
    ) -> tuple[List[dict], str]:
        """Attempt to recover tool calls using error recovery manager."""
        try:
            recovery_result = self._recovery_manager.handle_error(
                ToolCallParseError(str(parse_error), response),
                context={"retry_count": 0}
            )
            if recovery_result and isinstance(recovery_result, dict) and "name" in recovery_result:
                return [
                    convert_tool_call_to_langchain_format(recovery_result)
                ], ""
        except Exception:
            pass
        
        return [], response
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Stream response tokens. Falls back to non-streaming if not supported."""
        try:
            # Try streaming if available
            if hasattr(self.llm_client, 'stream_invoke'):
                try:
                    yield from self._stream_with_tool_parsing(messages, **kwargs)
                    return
                except NotImplementedError:
                    pass  # Fall back to non-streaming
            
            # Fallback to non-streaming
            result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            if result.generations:
                content = self._extract_content_from_result(result.generations[0])
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))
        except NotImplementedError as e:
            raise NotImplementedError(
                f"Streaming not supported by LLM client {type(self.llm_client).__name__}. "
                "The client's stream_invoke() method is not implemented."
            ) from e
    
    def _stream_with_tool_parsing(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ):
        """Stream response and parse tool calls at the end."""
        tool_definitions = convert_tools_to_definitions(self.bound_tools) if self.bound_tools else []
        serializer = MessageSerializerFactory.create_serializer(self.serializer_type)
        prompt = serializer.serialize(messages, tool_definitions)
        
        accumulated_content = ""
        for chunk in self.llm_client.stream_invoke(prompt, **kwargs):
            accumulated_content += chunk
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
        
        # After streaming completes, try to parse tool calls if tools are bound
        if tool_definitions and accumulated_content:
            tool_calls = self._parse_tool_calls_from_stream(accumulated_content, tool_definitions)
            if tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", tool_calls=tool_calls)
                )
            # If no tool calls, final chunk was already yielded above
    
    def _parse_tool_calls_from_stream(
        self,
        content: str,
        tool_definitions: List
    ) -> List[dict]:
        """Parse tool calls from accumulated stream content."""
        try:
            parser = ToolCallParserFactory.create_parser(self.parser_type)
            available_tool_names = [td.name for td in tool_definitions]
            parsed_tool_calls = parser.parse(content, available_tool_names)
            
            return [
                convert_tool_call_to_langchain_format(tc)
                for tc in parsed_tool_calls
            ]
        except Exception:
            return []
    
    def _extract_content_from_result(self, generation: ChatGeneration) -> str:
        """Extract content string from ChatGeneration."""
        if hasattr(generation.message, 'content') and generation.message.content:
            content = generation.message.content
            return content if isinstance(content, str) else str(content)
        return str(generation.message)
    
    def bind_tools(
        self,
        tools: Union[List[BaseTool], List[dict], BaseTool, dict],
        **kwargs: Any,
    ) -> "LangChainLLMAdapter":
        """
        Bind tools to the model. Returns a new instance with tools bound.
        
        Since our underlying LLMClient doesn't natively support tool binding,
        we store the tools and they'll be handled by the agent framework.
        """
        # Normalize tools to list
        if isinstance(tools, (BaseTool, dict)):
            tools_list = [tools]
        else:
            tools_list = list(tools)
        
        # Create a copy of self with tools bound using Pydantic's model_copy
        new_instance = self.model_copy(update={"bound_tools": tools_list})
        return new_instance
