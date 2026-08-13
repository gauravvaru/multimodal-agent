from __future__ import annotations
from multimodal_agent.utilities.security import redact_secrets
_SENSITIVE_FIELD_MARKERS = ('password', 'secret', 'token', 'api_key', 'authorization', 'cookie', 'extracted_text', 'file_content', 'raw_content', 'user_query')

def safe_log_message(message: str) -> str:
    return redact_secrets(message)

def safe_log_fields(fields: dict[str, object]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in fields.items():
        lowered = key.lower()
        if any((marker in lowered for marker in _SENSITIVE_FIELD_MARKERS)):
            sanitized[key] = '[omitted]'
            continue
        sanitized[key] = redact_secrets(str(value))
    return sanitized