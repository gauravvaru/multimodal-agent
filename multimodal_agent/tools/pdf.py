from __future__ import annotations
import io
import logging
import time
from typing import Any
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.storage_service import load_artifact_bytes

_TOOL_NAME = "pdf_extract"
_logger = logging.getLogger(__name__)


def extract_pdf(source: str, *, artifact_id: str | None = None) -> ToolResult:
    started = time.perf_counter()

    if not source or not source.strip():
        return _failed(
            error="PDF source is required",
            latency_ms=_elapsed_ms(started),
        )

    try:
        pdf_bytes = load_artifact_bytes(source)
    except ValueError as exc:
        return _failed(error=str(exc), latency_ms=_elapsed_ms(started))
    except FileNotFoundError as exc:
        return _failed(error=str(exc), latency_ms=_elapsed_ms(started))
    except OSError as exc:
        return _failed(
            error=f"Unable to read PDF source: {exc}",
            latency_ms=_elapsed_ms(started),
        )

    if not pdf_bytes:
        return _failed(
            error="PDF file is empty",
            latency_ms=_elapsed_ms(started),
        )

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    except PdfReadError as exc:
        return _failed(
            error=f"Invalid or corrupt PDF: {exc}",
            latency_ms=_elapsed_ms(started),
        )

    pages: list[dict[str, Any]] = []
    text_parts: list[str] = []
    pages_with_text = 0

    for page_number, page in enumerate(reader.pages, start=1):
        extracted = (page.extract_text() or "").strip()
        if extracted:
            pages_with_text += 1
            text_parts.append(extracted)
        pages.append(
            {
                "page": page_number,
                "chunk_id": f"page-{page_number}",
                "text": extracted,
            }
        )

    page_count = len(reader.pages)
    full_text = "\n\n".join(text_parts)

    from multimodal_agent.config.settings import get_settings
    settings = get_settings()

    if len(full_text.strip()) < settings.min_pdf_text_chars and page_count > 0:
        try:
            import pdf2image
            import pytesseract

            images = pdf2image.convert_from_bytes(pdf_bytes)
            pages.clear()
            text_parts.clear()
            pages_with_text = 0

            for page_number, image in enumerate(images, start=1):
                extracted = (pytesseract.image_to_string(image) or "").strip()
                if extracted:
                    pages_with_text += 1
                    text_parts.append(extracted)
                pages.append(
                    {
                        "page": page_number,
                        "chunk_id": f"page-{page_number}",
                        "text": extracted,
                    }
                )
            
            page_count = len(images)
            full_text = "\n\n".join(text_parts)
        except Exception as exc: 
            _logger.warning(f"OCR fallback failed: {exc}")
    latency_ms = _elapsed_ms(started)
    filename = _display_filename(source)

    data: dict[str, Any] = {
        "artifact_id": artifact_id,
        "source_id": artifact_id,
        "filename": filename,
        "source": source,
        "pages": pages,
        "text": full_text,
        "page_count": page_count,
    }

    if page_count == 0:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            data=data,
            confidence=0.0,
            latency_ms=latency_ms,
            error="PDF contains no pages",
        )

    if not full_text:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="partial",
            data=data,
            confidence=0.0,
            latency_ms=latency_ms,
            error="PDF contains no extractable text",
        )

    confidence = round(pages_with_text / page_count, 4)
    return ToolResult(
        tool_name=_TOOL_NAME,
        status="success",
        data=data,
        confidence=confidence,
        latency_ms=latency_ms,
    )


def _display_filename(source: str) -> str:
    if source.startswith("upload://"):
        return source.removeprefix("upload://")
    return source.rsplit("/", 1)[-1]


def _failed(*, error: str, latency_ms: float) -> ToolResult:
    return ToolResult(
        tool_name=_TOOL_NAME,
        status="failed",
        error=error,
        latency_ms=latency_ms,
    )

def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
