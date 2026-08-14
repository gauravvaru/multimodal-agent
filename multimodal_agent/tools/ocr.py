from __future__ import annotations
import io
import time

from multimodal_agent.config.settings import get_settings
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.storage_service import load_artifact_bytes
from multimodal_agent.tools._tool_utils import elapsed_ms

_TOOL_NAME = "ocr"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tiff", ".bmp"}


def run_ocr(source: str) -> ToolResult:
    started = time.perf_counter()

    if not source or not source.strip():
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="OCR source is required",
            latency_ms=elapsed_ms(started),
        )

    if not _is_image_source(source):
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="OCR currently supports image inputs only. Use pdf_extract for text-based PDFs.",
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
            error=f"Unable to read OCR source: {exc}",
            latency_ms=elapsed_ms(started),
        )

    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="OCR dependencies are not installed. Install with: pip install multimodal-agent[media]",
            latency_ms=elapsed_ms(started),
        )

    try:
        image = Image.open(io.BytesIO(content))
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(image).strip()
        confidence = _average_confidence(ocr_data)
    except Exception as exc: 
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error=f"OCR failed: {exc}",
            latency_ms=elapsed_ms(started),
        )

    if not text:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="partial",
            data={"text": "", "confidence": confidence},
            error="OCR returned no extracted text",
            confidence=confidence,
            latency_ms=elapsed_ms(started),
        )

    settings = get_settings()
    status = "success"
    error: str | None = None
    if confidence < settings.min_ocr_confidence:
        status = "partial"
        error = "OCR confidence is below threshold"

    return ToolResult(
        tool_name=_TOOL_NAME,
        status=status,
        data={"text": text, "confidence": confidence},
        confidence=confidence,
        error=error,
        latency_ms=elapsed_ms(started),
    )


def _is_image_source(source: str) -> bool:
    lowered = source.lower().split("?")[0]
    return any(lowered.endswith(ext) for ext in _IMAGE_EXTENSIONS)


def _average_confidence(ocr_data: dict[str, list[object]]) -> float:
    confidences = ocr_data.get("conf", [])
    values: list[float] = []
    for value in confidences:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score >= 0:
            values.append(score / 100.0)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
