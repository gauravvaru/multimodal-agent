from __future__ import annotations
import re
from multimodal_agent.models.intent import Intent
from multimodal_agent.models.state import AgentState
from multimodal_agent.utilities.urls import is_youtube_url

_SUMMARIZE_PATTERN = re.compile(r"\b(summarize|summarise|summary|action items?)\b", re.IGNORECASE)
_SENTIMENT_PATTERN = re.compile(r"\b(sentiment|tone|polarity)\b", re.IGNORECASE)
_TRANSCRIBE_PATTERN = re.compile(r"\b(transcribe|transcription|transcript)\b", re.IGNORECASE)
_OCR_PATTERN = re.compile(r"\b(ocr|extract text)\b", re.IGNORECASE)
_CODE_EXPLAIN_PATTERN = re.compile(
    r"\b(explain (this )?code|what does (this )?code)\b",
    re.IGNORECASE,
)
_COMPARE_PATTERN = re.compile(r"\b(compare|difference between|vs\.?|same topic)\b", re.IGNORECASE)


def requires_semantic_intent(state: AgentState) -> bool:
    return detect_intent_deterministic(state) is None


def detect_intent_deterministic(state: AgentState) -> Intent | None:
    query = state.user_query.strip()
    if not query:
        return None

    if is_youtube_url(query) or _query_contains_youtube_url(query):
        return Intent(name="youtube", confidence=1.0, rationale="YouTube URL detected")

    artifact_types = {item.artifact_type for item in state.normalized_contents}
    
    if _SUMMARIZE_PATTERN.search(query) and artifact_types.intersection({"pdf", "text", "document", "audio"}):
        return Intent(name="summarize", confidence=0.95, rationale="Summarize keyword with document input")

    if _SENTIMENT_PATTERN.search(query) and artifact_types.intersection({"text", "document", "pdf"}):
        return Intent(name="sentiment", confidence=0.9, rationale="Sentiment keyword with text input")

    if _TRANSCRIBE_PATTERN.search(query) and "audio" in artifact_types:
        return Intent(name="transcription", confidence=0.95, rationale="Transcription keyword with audio input")

    if _OCR_PATTERN.search(query) and artifact_types.intersection({"image", "pdf"}):
        return Intent(name="ocr", confidence=0.9, rationale="OCR keyword with visual input")

    if _CODE_EXPLAIN_PATTERN.search(query) and artifact_types.intersection({"code", "text", "image"}):
        return Intent(name="code_explanation", confidence=0.9, rationale="Code explanation keyword detected")

    if _COMPARE_PATTERN.search(query) and len(state.normalized_contents) >= 2:
        return Intent(name="comparison", confidence=0.85, rationale="Comparison keyword with multiple inputs")

    if not state.normalized_contents and not state.input_artifacts:
        return None

    return None


def _query_contains_youtube_url(query: str) -> bool:
    for token in query.split():
        if is_youtube_url(token):
            return True
    return False
