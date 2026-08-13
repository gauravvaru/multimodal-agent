"""Security and upload constraint helpers."""

from __future__ import annotations

import re
from pathlib import PurePath

_FILENAME_UNSAFE_PATTERN = re.compile(r"[\x00<>:\"|?*]")
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"lsv2_[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*\S+"),
)

_FILE_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".wav": (b"RIFF",),
    ".mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"),
    ".m4a": (b"\x00\x00\x00", b"ftyp"),
}


def validate_file_size(size_bytes: int, max_size_mb: int) -> bool:
    """Return True when file size is within the configured limit."""
    return size_bytes <= max_size_mb * 1024 * 1024


def sanitize_filename(filename: str) -> str:
    """Return a safe basename without path traversal components."""
    cleaned = filename.replace("\\", "/").split("/")[-1].strip()
    cleaned = _FILENAME_UNSAFE_PATTERN.sub("_", cleaned)
    cleaned = cleaned.lstrip(".")
    return cleaned or "upload"


def is_safe_filename(filename: str) -> bool:
    """Return True when a filename does not contain traversal or unsafe parts."""
    if not filename or not filename.strip():
        return False
    if "\x00" in filename:
        return False
    if ".." in filename or filename.startswith(("/", "\\")):
        return False
    basename = PurePath(filename.replace("\\", "/")).name
    return basename == filename.replace("\\", "/").split("/")[-1]


def validate_filename_security(filename: str, *, label: str | None = None) -> list[str]:
    """Validate filename safety and return errors."""
    display = label or filename or "upload"
    if not is_safe_filename(filename):
        return [f"{display}: filename contains unsafe path characters"]
    return []


def validate_non_empty_file(content: bytes, *, label: str) -> list[str]:
    """Reject empty uploads."""
    if not content:
        return [f"{label}: file is empty"]
    return []


def validate_file_signature(content: bytes, *, filename: str, label: str | None = None) -> list[str]:
    """Validate file content using deterministic magic-byte checks."""
    display = label or filename
    extension = PurePath(filename).suffix.lower()
    signatures = _FILE_SIGNATURES.get(extension)
    if signatures is None:
        return []

    if extension == ".m4a":
        if len(content) >= 8 and b"ftyp" in content[4:12]:
            return []
        return [f"{display}: file content does not match expected M4A format"]

    if extension == ".wav":
        if content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WAVE":
            return []
        return [f"{display}: file content does not match expected WAV format"]

    if any(content.startswith(signature) for signature in signatures):
        return []
    return [f"{display}: file content does not match expected {extension.lstrip('.').upper()} format"]


def redact_secrets(text: str) -> str:
    """Redact secret-like substrings from user-visible text."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted


def safe_client_error_message(*, context: str = "operation") -> str:
    """Return a generic client-safe error message."""
    return f"Unable to complete the {context} due to an unexpected error."


def ensure_no_shell_metacharacters(value: str) -> bool:
    """Return True when a value is safe to pass without shell interpretation."""
    forbidden = {";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\x00"}
    return not any(char in value for char in forbidden)
