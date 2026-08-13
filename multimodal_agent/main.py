"""FastAPI application entrypoint."""

from fastapi import FastAPI

from multimodal_agent.api.routes import router
from multimodal_agent.config import get_settings
from multimodal_agent.config.langsmith import configure_langsmith_tracing

settings = get_settings()
configure_langsmith_tracing()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(router)
