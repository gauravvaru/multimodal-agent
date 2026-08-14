from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar
from pydantic import BaseModel
from multimodal_agent.config.settings import get_settings
from multimodal_agent.services.intent_service import IntentService

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI

    from multimodal_agent.services.planner_service import PlannerContext, PlannerLLMClient
    from multimodal_agent.services.synthesis_service import (
        SynthesisContext,
        SynthesisLLMClient,
    )

T = TypeVar("T", bound=BaseModel)
StructuredInvokeFn = Callable[..., BaseModel]
_structured_invoke_override: StructuredInvokeFn | None = None


class LLMNotConfiguredError(RuntimeError):
    """Raised when an LLM operation is requested without provider configuration."""

class LLMInvocationError(RuntimeError):
    """Raised when the configured LLM provider fails."""

def is_llm_configured() -> bool:
    settings = get_settings()
    return bool(settings.llm_api_key and settings.llm_api_key.strip())

def get_chat_model() -> ChatGoogleGenerativeAI | None:
    settings = get_settings()
    if not is_llm_configured():
        return None

    provider = settings.llm_provider.strip().lower()
    if provider != "google":
        raise LLMNotConfiguredError(
            f"Unsupported LLM provider '{settings.llm_provider}'. Supported: google"
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.llm_api_key,
        timeout=float(settings.llm_timeout_seconds),
        temperature=0,
    )

def require_chat_model() -> ChatGoogleGenerativeAI:
    model = get_chat_model()
    if model is None:
        raise LLMNotConfiguredError(
            "LLM is not configured. Set GEMINI_API_KEY in the environment."
        )
    return model

def set_structured_invoke_for_tests(handler: StructuredInvokeFn | None) -> None:
    global _structured_invoke_override
    _structured_invoke_override = handler

def reset_structured_invoke_for_tests() -> None:
    set_structured_invoke_for_tests(None)

def invoke_structured(
    prompt: str,
    output_model: type[T],
    *,
    system: str | None = None,
) -> T:
    import time
    if _structured_invoke_override is not None:
        try:
            result = _structured_invoke_override(prompt, output_model, system=system)
        except LLMInvocationError:
            raise
        except Exception as exc:
            raise LLMInvocationError(f"LLM structured invocation failed: {exc}") from exc
        if isinstance(result, output_model):
            return result
        return output_model.model_validate(result)

    model = require_chat_model()
    structured = model.with_structured_output(output_model)
    messages = _build_messages(prompt, system=system)
    
    last_exc = None
    for attempt in range(3):
        try:
            result = structured.invoke(messages)
            if not isinstance(result, output_model):
                return output_model.model_validate(result)
            return result
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                if "Quota exceeded" in str(exc):
                    raise LLMInvocationError("LLM quota exceeded. Please try again later or use a different key.") from exc
                import logging
                logger = logging.getLogger(__name__)
                delay = 2 ** (attempt + 1)
                logger.warning(f"Rate limited. Retrying structured in {delay}s...")
                time.sleep(delay)
                continue
            raise LLMInvocationError(f"LLM structured invocation failed: {exc}") from exc
            
    raise LLMInvocationError("LLM quota exceeded. Please try again later or use a different key.") from last_exc

def invoke_text(prompt: str, *, system: str | None = None) -> str:
    import time
    model = require_chat_model()
    messages = _build_messages(prompt, system=system)
    
    last_exc = None
    for attempt in range(3):
        try:
            response = model.invoke(messages)
            content = getattr(response, "content", response)
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                return "\n".join(parts).strip()
            return str(content).strip()
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                if "Quota exceeded" in str(exc):
                    raise LLMInvocationError("LLM quota exceeded. Please try again later or use a different key.") from exc
                import logging
                logger = logging.getLogger(__name__)
                delay = 2 ** (attempt + 1)
                logger.warning(f"Rate limited. Retrying text in {delay}s...")
                time.sleep(delay)
                continue
            raise LLMInvocationError(f"LLM invocation failed: {exc}") from exc

    raise LLMInvocationError("LLM quota exceeded. Please try again later or use a different key.") from last_exc

def build_synthesis_llm_client() -> SynthesisLLMClient | None:
    if not is_llm_configured():
        return None
    return LangChainSynthesisLLMClient()

class LangChainSynthesisLLMClient:
    def synthesize_structured(self, context: SynthesisContext, output_model: type[T]) -> T:
        from multimodal_agent.agent.nodes.synthesis import build_synthesis_prompt

        prompt = build_synthesis_prompt(context, output_model)
        system = (
            "You are a grounded assistant. Use only the supplied tool results and evidence. "
            "Do not invent facts, citations, or content."
        )
        return invoke_structured(prompt, output_model, system=system)

def build_intent_service() -> IntentService | None:
    if not is_llm_configured():
        return None
    return LangChainIntentLLMClient()

class LangChainIntentLLMClient(IntentService):
    def detect_semantic(
        self,
        query: str,
        normalized_contents: list[Any],
    ) -> Any:
        from multimodal_agent.models.intent import Intent

        prompt = (
            f"Determine the intent of the following user query based on the supplied context.\n\n"
            f"Query: {query}\n"
            f"Provided Artifacts:\n"
            f"{[c.model_dump() if hasattr(c, 'model_dump') else c for c in normalized_contents]}\n"
        )
        system = (
            "You are a routing agent. Determine the correct intent based on the user's query and provided files.\n"
            "Intents:\n"
            "- 'summarize': request to summarize documents or audio.\n"
            "- 'code_explanation': request to explain code from an image or file.\n"
            "- 'comparison': request to compare information across multiple files.\n"
            "- 'sentiment': request for sentiment analysis.\n"
            "- 'rag': specific retrieval-augmented generation questions against the documents.\n"
            "- 'conversational': general chat not tied to the documents.\n"
            "Provide the intent name and confidence score (0.0 to 1.0)."
        )
        return invoke_structured(prompt, Intent, system=system)

def build_planner_llm_client() -> PlannerLLMClient | None:
    if not is_llm_configured():
        return None
    return LangChainPlannerLLMClient()

class LangChainPlannerLLMClient:
    def generate_plan(self, context: PlannerContext) -> Any:
        from multimodal_agent.services.planner_service import PlanDraft, build_planner_prompt

        prompt = build_planner_prompt(context)
        system = (
            "You are an AI planner. Your job is to select the correct sequence of tools to call to resolve the user's request.\n"
            "Valid tool_name choices are in the available_tools list. Return a step_id, tool_name, inputs, and depends_on list of step_ids for each step.\n"
            "Ensure the plan uses the minimal steps necessary."
        )
        return invoke_structured(prompt, PlanDraft, system=system)

def _build_messages(prompt: str, *, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages
