"""Tests for deterministic utilities."""

from multimodal_agent.utilities.mime import detect_mime_type


def test_detect_mime_type_pdf() -> None:
    assert detect_mime_type("report.pdf") == "application/pdf"


def test_detect_mime_type_unknown() -> None:
    assert detect_mime_type("no-extension") is None
