"""Tests for configuration."""

from multimodal_agent.config import Settings, get_settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "Multimodal Agent"
    assert settings.max_graph_steps == 20


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
