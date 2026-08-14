from __future__ import annotations
import re
import time
from multimodal_agent.config.settings import get_settings
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.utilities.urls import validate_youtube_url

_TOOL_NAME = "youtube_transcript"
_VIDEO_ID_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{6,})",
    re.IGNORECASE,
)


def fetch_youtube_transcript(url: str) -> ToolResult:
    started = time.perf_counter()
    settings = get_settings()
    errors = validate_youtube_url(url, block_private_hosts=settings.block_private_urls)
    if errors:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error=errors[0],
            latency_ms=_elapsed_ms(started),
        )

    video_id = _extract_video_id(url)
    if not video_id:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="Invalid YouTube URL format",
            latency_ms=_elapsed_ms(started),
        )

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="youtube-transcript-api is not installed",
            latency_ms=_elapsed_ms(started),
        )

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as exc: 
        message = str(exc) or "YouTube transcript is unavailable"
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="partial",
            data={"transcript_unavailable": True, "message": message},
            error=message,
            latency_ms=_elapsed_ms(started),
        )

    text = " ".join(item.get("text", "") for item in transcript if isinstance(item, dict)).strip()
    if not text:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="partial",
            data={"transcript_unavailable": True, "message": "Transcript is empty"},
            error="Transcript is empty",
            latency_ms=_elapsed_ms(started),
        )

    return ToolResult(
        tool_name=_TOOL_NAME,
        status="success",
        data={"transcript": text, "video_id": video_id},
        confidence=0.95,
        latency_ms=_elapsed_ms(started),
    )


def _extract_video_id(url: str) -> str | None:
    match = _VIDEO_ID_PATTERN.search(url)
    return match.group(1) if match else None


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
