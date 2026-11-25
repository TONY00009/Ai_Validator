from collections.abc import Iterator
import hmac
import json
from typing import Any, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.db import get_conn
from app.db import repository
from app.graph.workflow import run_validation_pipeline, stream_validation_pipeline

router = APIRouter()

MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB safety limit

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _format_sse(event_name: str, payload: dict[str, Any]) -> str:
    """Formats a dict as a standard Server-Sent Events (SSE) message frame."""
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"


def _sse_stream(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for item in events:
        event_name = str(item.get("event", "status"))
        yield _format_sse(event_name, item)


def require_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """Validates the X-API-Key header against VALIDATOR_API_KEY in .env."""
    expected = get_settings().validator_api_key
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
    return api_key


class VerifyTextRequest(BaseModel):
    input_text: str = Field(
        ...,
        min_length=10,
        max_length=15000,
        description="The AI-generated answer, paragraph, or claim text to fact-check.",
    )
    mode: Literal["web_check", "api_guardrail"] = Field(
        default="web_check",
        description="Use 'web_check' for general fact-checking or 'api_guardrail' for app gating.",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Maximum adaptive web-search retry rounds for unresolved claims.",
    )


def _read_and_validate_pdf(file: UploadFile) -> tuple[bytes, str]:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pdf files are supported.",
        )

    pdf_bytes = file.file.read()
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF file is empty.",
        )
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF file exceeds the 10 MB size limit.",
        )
    return pdf_bytes, filename


@router.get("/health", tags=["System"])
def health_check() -> dict[str, Any]:
    """Verifies that both the FastAPI server and PostgreSQL database are healthy."""
    settings = get_settings()
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                cur.fetchone()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unreachable: {exc}",
        )

    return {
        "status": "ok",
        "database": "connected",
        "env": settings.app_env,
    }


@router.post("/v1/verify", tags=["Verification"])
def verify_text(
    payload: VerifyTextRequest,
    _api_key: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Blocking JSON endpoint: fact-checks text against live web sources."""
    return run_validation_pipeline(
        mode=payload.mode,
        input_text=payload.input_text,
        max_retries=payload.max_retries,
    )


@router.post("/v1/verify/stream", tags=["Verification"])
def verify_text_stream(
    payload: VerifyTextRequest,
    _api_key: str = Depends(require_api_key),
) -> StreamingResponse:
    """SSE streaming endpoint: pushes real-time LangGraph progress events + final report."""
    events = stream_validation_pipeline(
        mode=payload.mode,
        input_text=payload.input_text,
        max_retries=payload.max_retries,
    )
    return StreamingResponse(
        _sse_stream(events),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/v1/verify-pdf", tags=["Verification"])
def verify_against_pdf(
    input_text: str = Form(
        ...,
        min_length=10,
        max_length=15000,
        description="Text or AI summary to audit strictly against the uploaded PDF.",
    ),
    file: UploadFile = File(..., description="Source PDF document (max 10 MB)."),
    _api_key: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Blocking JSON endpoint: audits text strictly against an uploaded PDF."""
    pdf_bytes, filename = _read_and_validate_pdf(file)
    return run_validation_pipeline(
        mode="pdf_audit",
        input_text=input_text,
        pdf_bytes=pdf_bytes,
        pdf_filename=filename,
        max_retries=0,
    )


@router.post("/v1/verify-pdf/stream", tags=["Verification"])
def verify_against_pdf_stream(
    input_text: str = Form(
        ...,
        min_length=10,
        max_length=15000,
        description="Text or AI summary to audit strictly against the uploaded PDF.",
    ),
    file: UploadFile = File(..., description="Source PDF document (max 10 MB)."),
    _api_key: str = Depends(require_api_key),
) -> StreamingResponse:
    """SSE streaming endpoint: streams real-time PDF audit progress + final report."""
    pdf_bytes, filename = _read_and_validate_pdf(file)
    events = stream_validation_pipeline(
        mode="pdf_audit",
        input_text=input_text,
        pdf_bytes=pdf_bytes,
        pdf_filename=filename,
        max_retries=0,
    )
    return StreamingResponse(
        _sse_stream(events),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/v1/runs", tags=["History"])
def get_recent_runs(
    limit: int = 10,
    _api_key: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Lists recent verification runs (newest first)."""
    safe_limit = max(1, min(limit, 50))
    return {"runs": repository.list_recent_runs(limit=safe_limit)}


@router.get("/v1/runs/{run_id}", tags=["History"])
def get_run_by_id(
    run_id: str,
    _api_key: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Fetches the full verification report, verdicts, and audit trail for a run_id."""
    try:
        UUID(run_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid run_id format (expected UUID).",
        )

    try:
        return repository.get_run_report(run_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )