"""Inbound artifact validation."""

from __future__ import annotations

from pathlib import PurePath

from multimodal_agent.models.requests import InputArtifact
from multimodal_agent.utilities.mime import detect_mime_type
from multimodal_agent.utilities.security import (
    sanitize_filename,
    validate_file_signature,
    validate_file_size,
    validate_filename_security,
)

ALLOWED_EXTENSIONS = frozenset({".pdf", ".jpg", ".jpeg", ".png", ".mp3", ".wav", ".m4a"})

ALLOWED_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
    }
)


def validate_artifact(
    artifact: InputArtifact,
    *,
    max_size_mb: int,
    size_bytes: int | None = None,
) -> list[str]:
    """Validate a single artifact reference and return validation errors."""
    errors: list[str] = []
    filename = artifact.filename or ""
    label = sanitize_filename(filename) if filename else (artifact.source or "artifact")

    if filename:
        errors.extend(validate_filename_security(filename, label=label))
        errors.extend(_validate_extension(filename, label=label))

    if size_bytes is not None and not validate_file_size(size_bytes, max_size_mb):
        errors.append(f"{label}: file exceeds maximum size of {max_size_mb} MB")

    if artifact.content_type and filename:
        detected = detect_mime_type(filename)
        if (
            artifact.content_type not in ALLOWED_MIME_TYPES
            and detected not in ALLOWED_MIME_TYPES
            and PurePath(filename).suffix.lower() not in ALLOWED_EXTENSIONS
        ):
            errors.append(f"{label}: unsupported content type '{artifact.content_type}'")

    return errors


def validate_upload(
    *,
    filename: str,
    content_type: str | None,
    size_bytes: int,
    max_size_mb: int,
) -> list[str]:
    """Validate an uploaded file before ingestion."""
    errors: list[str] = []

    if not filename or not filename.strip():
        errors.append("uploaded file must have a filename")
        return errors

    safe_name = sanitize_filename(filename)
    label = safe_name

    errors.extend(validate_filename_security(filename, label=label))
    errors.extend(_validate_extension(safe_name, label=label))

    if size_bytes == 0:
        errors.append(f"{label}: file is empty")

    if not validate_file_size(size_bytes, max_size_mb):
        errors.append(f"{label}: file exceeds maximum size of {max_size_mb} MB")

    detected = content_type or detect_mime_type(safe_name)
    if detected and detected not in ALLOWED_MIME_TYPES:
        extension = PurePath(safe_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            errors.append(f"{label}: unsupported file type")

    return errors


def validate_upload_bytes(
    *,
    filename: str,
    content: bytes,
    content_type: str | None,
    max_size_mb: int,
) -> list[str]:
    """Validate uploaded bytes including empty and corrupt file checks."""
    errors = validate_upload(
        filename=filename,
        content_type=content_type,
        size_bytes=len(content),
        max_size_mb=max_size_mb,
    )
    if errors:
        return errors

    safe_name = sanitize_filename(filename)
    return validate_file_signature(content, filename=safe_name, label=safe_name)


def validate_upload_count(file_count: int, *, max_files: int) -> list[str]:
    """Validate the number of uploaded files in a single request."""
    if file_count > max_files:
        return [f"too many files uploaded: maximum allowed is {max_files}"]
    return []


def _validate_extension(filename: str, *, label: str | None = None) -> list[str]:
    extension = PurePath(filename).suffix.lower()
    display = label or filename
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".") for ext in ALLOWED_EXTENSIONS))
        return [f"{display}: unsupported file type '{extension or 'unknown'}'. Allowed: {allowed.upper()}"]
    return []
