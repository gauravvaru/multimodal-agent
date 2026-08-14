from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Multimodal Agent"
    app_version: str = "0.1.0"
    max_upload_size_mb: int = Field(default=25, ge=1)
    max_upload_files: int = Field(default=10, ge=1)
    request_timeout_seconds: int = Field(default=120, ge=1)
    llm_timeout_seconds: int = Field(default=120, ge=1)
    llm_provider: str = Field(default="google")
    llm_model: str = Field(default="gemini-flash-latest")
    llm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("llm_api_key", "gemini_api_key"))
    llm_max_input_chars: int = Field(default=12000, ge=1000)
    block_private_urls: bool = True
    max_graph_steps: int = Field(default=20, ge=1)
    max_tool_retries: int = Field(default=2, ge=0)
    min_ocr_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_pdf_text_chars: int = Field(default=1, ge=0)
    min_rag_evidence_items: int = Field(default=1, ge=0)

@lru_cache
def get_settings() -> Settings:
    return Settings()

def clear_settings_cache() -> None:
    get_settings.cache_clear()
