"""LLM invocation service."""

from __future__ import annotations

from typing import Any

from multimodal_agent.services.llm_provider import (
    LLMInvocationError,
    invoke_text,
    is_llm_configured,
)


class LLMService:
    """Interface for LLM-backed semantic operations."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt."""
        return self.generate_sync(prompt, **kwargs)

    def generate_sync(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt synchronously."""
        raise NotImplementedError


class GoogleLLMService(LLMService):
    """Google Gemini-compatible LLM service using LangChain."""

    def generate_sync(self, prompt: str, **kwargs: Any) -> str:
        system = kwargs.get("system")
        if not is_llm_configured():
            raise LLMInvocationError("LLM is not configured. Set GEMINI_API_KEY.")
        return invoke_text(prompt, system=system if isinstance(system, str) else None)


def get_llm_service() -> LLMService:
    """Return the default LLM service implementation."""
    return GoogleLLMService()
