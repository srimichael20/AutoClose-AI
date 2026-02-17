"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autoclose.api.routes import router
from autoclose.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure data directories exist on startup."""
    settings = get_settings()
    Path(settings.upload_directory).mkdir(parents=True, exist_ok=True)
    Path(settings.processed_directory).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="AutoClose AI",
        description="Autonomous Accounting Agent for SMB Book Closing - Multi-Agent System",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1", tags=["workflow"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "autoclose-ai"}

    return app
