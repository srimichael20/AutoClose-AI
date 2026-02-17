"""
AutoClose AI - FastAPI entry point.

Multi-agent workflow: Intake → Vision → Classification → MCP → Orchestrator
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from utils.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    Path(s.upload_directory).mkdir(parents=True, exist_ok=True)
    Path(s.processed_directory).mkdir(parents=True, exist_ok=True)
    Path(s.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
    Path(s.database_path).parent.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="AutoClose AI",
    description="Autonomous Accounting Agent for SMB Book Closing",
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


@app.get("/")
async def root():
    return {
        "service": "AutoClose AI",
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "autoclose-ai"}


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run(app, host=s.api_host, port=s.api_port)
