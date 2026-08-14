from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from multimodal_agent.api.routes import router
from multimodal_agent.config import get_settings
from multimodal_agent.config.langsmith import configure_langsmith_tracing

settings = get_settings()
configure_langsmith_tracing()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
