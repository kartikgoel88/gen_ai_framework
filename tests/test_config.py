"""Tests for framework config."""

from src.framework.config import FrameworkSettings, get_settings, get_settings_dep


def test_framework_settings_defaults():
    """FrameworkSettings has expected default values."""
    # Use env override to avoid loading .env in tests
    settings = FrameworkSettings(OPENAI_API_KEY="")
    assert settings.LLM_PROVIDER == "openai"
    assert settings.LLM_MODEL == "gpt-4-turbo-preview"
    assert settings.TEMPERATURE == 0.7
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 200
    assert settings.CHUNKING_STRATEGY == "recursive_character"
    assert settings.PROMPTS_BASE_PATH == "./data/prompts"
    assert settings.VECTOR_STORE == "chroma"
    assert settings.UPLOAD_DIR == "./uploads"
    assert settings.DEBUG is True


def test_get_settings_returns_framework_settings():
    """get_settings returns FrameworkSettings instance."""
    # Clear cache to avoid cross-test pollution
    get_settings.cache_clear()
    settings = get_settings()
    assert isinstance(settings, FrameworkSettings)
    get_settings.cache_clear()


def test_get_settings_dep_returns_settings():
    """get_settings_dep returns same type as get_settings."""
    s = get_settings_dep()
    assert isinstance(s, FrameworkSettings)
