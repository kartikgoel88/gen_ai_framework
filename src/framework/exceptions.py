"""Framework exception hierarchy.

This module defines custom exceptions for the framework to provide
clearer error messages and better error handling.
"""


class FrameworkError(Exception):
    """Base exception for all framework errors.
    
    All framework-specific exceptions should inherit from this class.
    """
    pass


class ProviderNotFoundError(FrameworkError):
    """Raised when a provider (LLM, RAG, embeddings) is not found or not registered.
    
    Attributes:
        provider: Name of the provider that was not found
        available: List of available provider names
    """
    def __init__(self, provider: str, available: list[str] | None = None):
        self.provider = provider
        self.available = available or []
        message = f"Provider '{provider}' not found"
        if self.available:
            message += f". Available providers: {', '.join(self.available)}"
        super().__init__(message)


class ConfigurationError(FrameworkError):
    """Raised when configuration is invalid or missing required settings.
    
    Attributes:
        setting: Name of the setting that is invalid/missing
        message: Detailed error message
    """
    def __init__(self, setting: str, message: str | None = None):
        self.setting = setting
        self.message = message or f"Invalid or missing configuration for '{setting}'"
        super().__init__(self.message)


class APIKeyError(ConfigurationError):
    """Raised when a required API key is missing.
    
    Attributes:
        provider: Provider name requiring the API key
        env_var: Environment variable name for the API key
    """
    def __init__(self, provider: str, env_var: str):
        self.provider = provider
        self.env_var = env_var
        message = f"{env_var} is required for {provider}. Set {env_var}."
        super().__init__(env_var, message)


class VectorStoreError(FrameworkError):
    """Raised when vector store operations fail.
    
    Attributes:
        store: Vector store name
        operation: Operation that failed
        message: Detailed error message
    """
    def __init__(self, store: str, operation: str, message: str | None = None):
        self.store = store
        self.operation = operation
        self.message = message or f"Vector store '{store}' operation '{operation}' failed"
        super().__init__(self.message)
