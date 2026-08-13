"""Audio transcription tool."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.storage_service import load_artifact_bytes
from multimodal_agent.tools._tool_utils import elapsed_ms

_TOOL_NAME = "audio_transcribe"


def transcribe_audio(source: str) -> ToolResult:
    """Transcribe audio input using faster-whisper when available."""
    started = time.perf_counter()

    if not source or not source.strip():
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="Audio source is required",
            latency_ms=elapsed_ms(started),
        )

    try:
        content = load_artifact_bytes(source)
    except (ValueError, FileNotFoundError) as exc:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error=str(exc),
            latency_ms=elapsed_ms(started),
        )
    except OSError as exc:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error=f"Unable to read audio source: {exc}",
            latency_ms=elapsed_ms(started),
        )

    if not content:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="Audio file is empty",
            latency_ms=elapsed_ms(started),
        )

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="Audio transcription is not installed. Install with: pip install multimodal-agent[media]",
            latency_ms=elapsed_ms(started),
        )

    suffix = _audio_suffix(source)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(temp_path), beam_size=1)

        segment_payload: list[dict[str, object]] = []
        transcript_parts: list[str] = []
        for index, segment in enumerate(segments):
            text = segment.text.strip()
            if not text:
                continue
            transcript_parts.append(text)
            segment_payload.append(
                {
                    "chunk_id": str(index),
                    "text": text,
                    "timestamp_start": float(segment.start),
                    "timestamp_end": float(segment.end),
                }
            )

        transcript = " ".join(transcript_parts).strip()
        if not transcript:
            return ToolResult(
                tool_name=_TOOL_NAME,
                status="failed",
                error="Audio transcription returned no text",
                latency_ms=elapsed_ms(started),
            )

        language = getattr(info, "language", None)
        data: dict[str, object] = {
            "transcript": transcript,
            "segments": segment_payload,
        }
        if isinstance(language, str):
            data["language"] = language

        return ToolResult(
            tool_name=_TOOL_NAME,
            status="success",
            data=data,
            confidence=0.85,
            latency_ms=elapsed_ms(started),
        )
    except Exception as exc:  # noqa: BLE001 - transcription failures must degrade gracefully
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error=f"Audio transcription failed: {exc}",
            latency_ms=elapsed_ms(started),
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _audio_suffix(source: str) -> str:
    lowered = source.lower().split("?")[0]
    for ext in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"):
        if lowered.endswith(ext):
            return ext
    return ".wav"
