"""MIME type detection utilities."""

import mimetypes


def detect_mime_type(filename: str) -> str | None:
    """Detect MIME type from a filename using deterministic lookup."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type
