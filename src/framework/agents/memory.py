"""Agent memory and persistence for long-term conversation context."""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from datetime import datetime
from dataclasses import dataclass, asdict

from ..rag.base import RAGClient


@dataclass
class ConversationMemory:
    """Represents a stored conversation memory."""
    user_id: str
    message: str
    response: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


class MemoryStore(ABC):
    """Abstract interface for memory storage."""
    
    @abstractmethod
    def store(
        self,
        user_id: str,
        message: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a conversation memory."""
        ...
    
    @abstractmethod
    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 5
    ) -> List[ConversationMemory]:
        """Retrieve relevant past conversations."""
        ...
    
    @abstractmethod
    def clear(self, user_id: Optional[str] = None) -> None:
        """Clear memories for a user or all users."""
        ...


class RAGMemoryStore(MemoryStore):
    """Memory store using RAG for semantic search over past conversations.
    
    Stores conversations in a RAG store and retrieves relevant memories
    based on semantic similarity.
    """
    
    def __init__(
        self,
        rag_client: RAGClient,
        user_id_field: str = "user_id",
        collection_prefix: str = "agent_memory"
    ):
        """Initialize RAG-based memory store.
        
        Args:
            rag_client: RAG client for storage and retrieval
            user_id_field: Field name for user ID in metadata
            collection_prefix: Prefix for memory collection
        """
        self._rag = rag_client
        self._user_id_field = user_id_field
        self._collection_prefix = collection_prefix
    
    def store(
        self,
        user_id: str,
        message: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a conversation memory."""
        memory = ConversationMemory(
            user_id=user_id,
            message=message,
            response=response,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Create text representation for storage
        text = f"User: {message}\nAssistant: {response}"
        
        # Store in RAG with metadata
        memory_metadata = {
            self._user_id_field: user_id,
            "timestamp": memory.timestamp.isoformat(),
            "type": "conversation_memory",
            **(metadata or {})
        }
        
        self._rag.add_documents(
            texts=[text],
            metadatas=[memory_metadata]
        )
    
    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 5
    ) -> List[ConversationMemory]:
        """Retrieve relevant past conversations."""
        # Search with user_id filter in query
        search_query = f"{query} user:{user_id}"
        
        chunks = self._rag.retrieve(search_query, top_k=top_k)
        
        memories = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if metadata.get(self._user_id_field) == user_id:
                # Parse the stored text
                text = chunk.get("content", "")
                parts = text.split("\nAssistant: ", 1)
                if len(parts) == 2:
                    message = parts[0].replace("User: ", "")
                    response = parts[1]
                    
                    memory = ConversationMemory(
                        user_id=user_id,
                        message=message,
                        response=response,
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                        metadata={k: v for k, v in metadata.items() if k not in [self._user_id_field, "timestamp", "type"]}
                    )
                    memories.append(memory)
        
        return memories
    
    def clear(self, user_id: Optional[str] = None) -> None:
        """Clear memories for a user or all users."""
        # Note: RAG clear() clears everything
        # For user-specific clearing, would need custom implementation
        if user_id is None:
            self._rag.clear()
        else:
            # Would need to implement user-specific clearing in RAG client
            # For now, clear all (limitation)
            self._rag.clear()


class AgentWithMemory(AgentBase):
    """Agent wrapper that adds memory capabilities."""
    
    def __init__(self, base_agent: AgentBase, memory: Optional[MemoryStore] = None):
        """Initialize agent with optional memory.
        
        Args:
            base_agent: Base agent to wrap
            memory: Optional memory store
        """
        self._base_agent = base_agent
        self._memory = memory
    
    def invoke_with_memory(
        self,
        message: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Invoke agent with memory context.
        
        Args:
            message: User message
            user_id: Optional user ID for memory context
            **kwargs: Additional arguments
            
        Returns:
            Agent response
        """
        # Retrieve relevant memories
        context = ""
        if self._memory and user_id:
            memories = self._memory.retrieve(user_id, message, top_k=3)
            if memories:
                context_parts = []
                for mem in memories:
                    context_parts.append(f"Previous conversation:\nUser: {mem.message}\nAssistant: {mem.response}")
                context = "\n\n".join(context_parts) + "\n\n"
        
        # Enhance message with context
        enhanced_message = context + f"Current question: {message}"
        
        # Invoke agent
        response = self._base_agent.invoke(enhanced_message, **kwargs)
        
        # Store conversation
        if self._memory and user_id:
            self._memory.store(user_id, message, response, kwargs.get("metadata"))
        
        return response
    
    def invoke(self, message: str, **kwargs) -> str:
        """Invoke agent (without memory context)."""
        return self._base_agent.invoke(message, **kwargs)
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._base_agent.get_tools_description()
