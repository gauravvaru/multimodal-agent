from enum import StrEnum
from pydantic import BaseModel

class ValidationStatus(StrEnum):
    SUCCESS = 'success'
    PARTIAL = 'partial'
    RETRYABLE_FAILURE = 'retryable_failure'
    FATAL_FAILURE = 'fatal_failure'

class ResultValidationOutcome(BaseModel):
    validation_status: ValidationStatus
    retry_required: bool = False
    validation_error: str | None = None
    no_evidence: bool = False