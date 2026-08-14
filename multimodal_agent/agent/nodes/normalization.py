from __future__ import annotations
import mimetypes

from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.config.settings import get_settings
from multimodal_agent.models.artifacts import NormalizedArtifact
from multimodal_agent.models.requests import InputArtifact
from multimodal_agent.models.state import AgentState
from multimodal_agent.utilities.tracing import TraceEvent
from multimodal_agent.utilities.urls import is_youtube_url, validate_youtube_url


def normalize_inputs(state: AgentState) -> StateUpdate:
    if state.errors:
        return {
            "trace": [
                TraceEvent(
                    step="normalize_inputs",
                    detail={"status": "skipped", "reason": "validation_errors"},
                )
            ],
        }

    normalized = [_normalize_artifact(artifact, index=index) for index, artifact in enumerate(state.input_artifacts)]
    return {
        "normalized_contents": normalized,
        "trace": [
            TraceEvent(
                step="normalize_inputs",
                detail={"artifact_count": len(normalized), "status": "ok"},
            )
        ],
    }


def _normalize_artifact(artifact: InputArtifact, *, index: int) -> NormalizedArtifact:
    source = artifact.source or artifact.filename
    artifact_type = _detect_artifact_type(artifact, source=source or artifact.filename)
    metadata: dict[str, object] = {}
    if artifact.filename:
        metadata["filename"] = artifact.filename
    if source and is_youtube_url(source):
        settings = get_settings()
        if not validate_youtube_url(source, block_private_hosts=settings.block_private_urls):
            metadata["contains_youtube_url"] = False
        else:
            metadata["contains_youtube_url"] = True
            metadata["youtube_url"] = source

    return NormalizedArtifact(
        artifact_id=f"artifact-{index + 1}",
        artifact_type=artifact_type,
        source=source,
        content_type=artifact.content_type or (_guess_content_type(source) if source else None),
        metadata=metadata,
    )

def _detect_artifact_type(artifact: InputArtifact, *, source: str | None) -> str:
    filename = (artifact.filename or source or "").lower()
    if filename.endswith(".pdf"):
        return "pdf"
    if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".tiff")):
        return "image"
    if filename.endswith((".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")):
        return "audio"
    if filename.endswith((".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c")):
        return "code"
    if filename.endswith((".txt", ".md", ".csv")):
        return "text"
    if source and is_youtube_url(source):
        return "youtube"
    if artifact.content_type:
        if "pdf" in artifact.content_type:
            return "pdf"
        if artifact.content_type.startswith("audio/"):
            return "audio"
        if artifact.content_type.startswith("image/"):
            return "image"
    return "document"


def _guess_content_type(source: str | None) -> str | None:
    if not source:
        return None
    return mimetypes.guess_type(source)[0]
