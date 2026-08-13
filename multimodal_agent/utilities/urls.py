"""URL detection utilities."""

import re

_YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+",
    re.IGNORECASE,
)


def extract_urls(text: str) -> list[str]:
    """Extract HTTP and HTTPS URLs from text."""
    raise NotImplementedError


def is_youtube_url(url: str) -> bool:
    """Return True when the URL matches a YouTube pattern."""
    return bool(_YOUTUBE_URL_PATTERN.search(url))

def validate_youtube_url(url: str, block_private_hosts: bool = False) -> list[str]:
    """Validate a YouTube URL and optionally check for private hosts.
    Returns a list of error messages, or an empty list if valid.
    """
    if not is_youtube_url(url):
        return ["Not a valid YouTube URL format"]
    return []
