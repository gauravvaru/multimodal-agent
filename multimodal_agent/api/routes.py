from __future__ import annotations
from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from multimodal_agent.agent.streaming import format_sse
from multimodal_agent.api.schemas import (
    AgentRunResponse,
    ErrorResponse,
    HealthResponse,
    agent_response_to_run_response,
    is_input_validation_error,
)
from multimodal_agent.config.settings import Settings, get_settings
from multimodal_agent.ingestion.validators import (
    validate_upload_bytes,
    validate_upload_count,
)
from multimodal_agent.models.requests import InputArtifact
from multimodal_agent.services.agent_service import AgentService
from multimodal_agent.services.storage_service import get_upload_storage
from multimodal_agent.utilities.security import sanitize_filename

router = APIRouter()


def get_agent_service() -> AgentService:
    return AgentService()

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/api/v1/agent/run",
    response_model=AgentRunResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
async def run_agent(
    query: Annotated[str, Form()],
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> AgentRunResponse:
    normalized_query = query.strip()

    if not normalized_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required",
        )

    artifacts, upload_errors = await _build_artifacts_from_uploads(
        files or [],
        max_size_mb=settings.max_upload_size_mb,
        max_files=settings.max_upload_files,
    )
    if upload_errors:
        status_code = (
            status.HTTP_413_CONTENT_TOO_LARGE
            if _contains_oversized_error(upload_errors)
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=upload_errors)

    try:
        agent_response = agent_service.run(normalized_query, artifacts)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request.",
        ) from exc

    api_response = agent_response_to_run_response(agent_response)

    if api_response.status == "error" and is_input_validation_error(agent_response.errors):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=agent_response.errors,
        )

    return api_response


@router.post("/api/v1/agent/run/stream")
async def run_agent_stream(
    query: Annotated[str, Form()],
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> StreamingResponse:
    """Stream agent execution progress using Server-Sent Events."""
    normalized_query = query.strip()

    if not normalized_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required",
        )

    artifacts, upload_errors = await _build_artifacts_from_uploads(
        files or [],
        max_size_mb=settings.max_upload_size_mb,
        max_files=settings.max_upload_files,
    )
    if upload_errors:
        status_code = (
            status.HTTP_413_CONTENT_TOO_LARGE
            if _contains_oversized_error(upload_errors)
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=upload_errors)

    def event_generator():
        try:
            for event in agent_service.run_stream(normalized_query, artifacts):
                yield format_sse(event)
        except Exception as exc:
            import logging
            logger = logging.getLogger(__name__)
            logger.exception("Unhandled exception during agent streaming")
            from multimodal_agent.models.stream_events import AgentStreamEvent

            error_msg = f"An unexpected error occurred while streaming agent progress: {exc}"
            failure = AgentStreamEvent(
                type="error",
                message=error_msg,
                errors=[error_msg],
            )
            yield format_sse(failure)
            yield format_sse(AgentStreamEvent(type="complete", errors=failure.errors))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

async def _build_artifacts_from_uploads(
    files: list[UploadFile],
    *,
    max_size_mb: int,
    max_files: int,
) -> tuple[list[InputArtifact], list[str]]:
    artifacts: list[InputArtifact] = []
    errors: list[str] = list(validate_upload_count(len(files), max_files=max_files))
    if errors:
        return artifacts, errors

    storage = get_upload_storage()

    for upload in files:
        filename = upload.filename or ""
        content = await upload.read()
        file_errors = validate_upload_bytes(
            filename=filename,
            content=content,
            content_type=upload.content_type,
            max_size_mb=max_size_mb,
        )
        if file_errors:
            errors.extend(file_errors)
            continue

        safe_name = sanitize_filename(filename)
        reference = storage.store_bytes(safe_name, content)
        artifacts.append(
            InputArtifact(
                filename=safe_name,
                content_type=upload.content_type,
                source=reference,
            )
        )

    return artifacts, errors

def _contains_oversized_error(errors: list[str]) -> bool:
    return any("file exceeds maximum size" in error for error in errors)
