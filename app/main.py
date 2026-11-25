from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes import router as api_router
from app.core.db import close_pool

WEB_INDEX_FILE = Path(__file__).resolve().parent / "web" / "index.html"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    close_pool()


app = FastAPI(
    title="Ai_Validator API",
    description=(
        "Multi-model AI Fact-Checking & Hallucination Guardrail Engine "
        "powered by LangGraph, pgvector, and deterministic Python verification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", include_in_schema=False)
def serve_demo_ui() -> FileResponse:
    """Serves the single-page Ai_Validator interactive demo UI."""
    return FileResponse(WEB_INDEX_FILE)