"""Helpers for extracting text from tool result payloads."""

from __future__ import annotations

from typing import Any


def extract_text_from_tool_data(data: Any) -> str:
    """Extract the primary text content from a tool result data payload."""
    if isinstance(data, str) and data.strip():
        return data.strip()
    if not isinstance(data, dict):
        return ""

    for key in ("text", "content", "transcript", "transcription", "summary"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    pages = data.get("pages")
    if isinstance(pages, list):
        parts: list[str] = []
        for page in pages:
            if isinstance(page, dict):
                page_text = page.get("text")
                if isinstance(page_text, str) and page_text.strip():
                    parts.append(page_text.strip())
        if parts:
            return "\n\n".join(parts)

    return ""
