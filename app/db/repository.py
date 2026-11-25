import json
from typing import Any

from app.core.db import get_conn
from app.graph.schemas import RunMode, VerdictAction, VerdictLabel


def create_run(mode: RunMode, input_text: str) -> str:
    """Creates a new run row in 'running' status and returns its UUID string."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (mode, input_text, status)
                VALUES (%s, %s, 'running')
                RETURNING id
                """,
                (mode, input_text),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to create run record.")
            run_id = str(row["id"])
        conn.commit()
    return run_id


def finalize_run_record(
    run_id: str,
    trust_score: float,
    verdict_action: VerdictAction,
) -> None:
    """Marks a run as completed with its final Trust Score and Guardrail Action."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                SET status = 'completed',
                    trust_score = %s,
                    verdict_action = %s,
                    completed_at = NOW()
                WHERE id = %s
                """,
                (trust_score, verdict_action, run_id),
            )
        conn.commit()


def mark_run_failed(run_id: str, error_message: str) -> None:
    """Marks a run as failed and stores the error for debugging."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                SET status = 'failed',
                    error_message = %s,
                    completed_at = NOW()
                WHERE id = %s
                """,
                (error_message[:1000], run_id),
            )
        conn.commit()


def log_audit_event(
    run_id: str,
    stage: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Appends a timestamped step to audit_events for full pipeline traceability."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_events (run_id, stage, message, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (run_id, stage, message, json.dumps(payload or {})),
            )
        conn.commit()


def save_claims(run_id: str, claim_texts: list[str]) -> list[dict[str, Any]]:
    """Inserts extracted claims and returns their DB records with UUIDs."""
    saved: list[dict[str, Any]] = []
    if not claim_texts:
        return saved

    with get_conn() as conn:
        with conn.cursor() as cur:
            for idx, text in enumerate(claim_texts, start=1):
                cur.execute(
                    """
                    INSERT INTO claims (run_id, claim_index, claim_text, status)
                    VALUES (%s, %s, %s, 'pending')
                    RETURNING id
                    """,
                    (run_id, idx, text),
                )
                row = cur.fetchone()
                if row is None:
                    continue
                saved.append(
                    {
                        "claim_id": str(row["id"]),
                        "claim_index": idx,
                        "claim_text": text,
                        "retry_count": 0,
                    }
                )
        conn.commit()
    return saved


def save_verdict(
    claim_id: str,
    verdict: VerdictLabel,
    reason: str,
    verbatim_quote: str,
    cited_chunk_id: str | None,
    quote_matched: bool,
    names_matched: bool,
    numbers_matched: bool,
    model_used: str,
    retry_count: int = 0,
) -> str:
    """Saves a verdict row and updates the parent claim's status and retry_count."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO verdicts (
                    claim_id, verdict, reason, verbatim_quote, cited_chunk_id,
                    quote_matched, names_matched, numbers_matched, model_used
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    claim_id,
                    verdict,
                    reason,
                    verbatim_quote or None,
                    cited_chunk_id,
                    quote_matched,
                    names_matched,
                    numbers_matched,
                    model_used,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert verdict row.")
            verdict_id = str(row["id"])

            cur.execute(
                """
                UPDATE claims
                SET status = %s, retry_count = %s
                WHERE id = %s
                """,
                (verdict, retry_count, claim_id),
            )
        conn.commit()
    return verdict_id


def save_contradictions(
    run_id: str,
    contradictions: list[dict[str, Any]],
) -> None:
    """Persists detected claim contradictions into PostgreSQL."""
    if not contradictions:
        return

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO contradictions (run_id, claim_a_id, claim_b_id, summary)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (run_id, item["claim_a_id"], item["claim_b_id"], item["summary"])
                    for item in contradictions
                ],
            )
        conn.commit()


def get_run_report(run_id: str) -> dict[str, Any]:
    """
    Fetches the complete verification report for a run (used by both smoke tests
    and the upcoming FastAPI /v1/verify response).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, mode, input_text, status, verdict_action,
                       trust_score, error_message, created_at, completed_at
                FROM runs
                WHERE id = %s
                """,
                (run_id,),
            )
            run_row = cur.fetchone()
            if run_row is None:
                raise ValueError(f"Run not found: {run_id}")

            # Latest verdict per claim along with cited source metadata
            cur.execute(
                """
                SELECT DISTINCT ON (c.claim_index)
                    c.id AS claim_id,
                    c.claim_index,
                    c.claim_text,
                    c.status,
                    c.retry_count,
                    v.verdict,
                    v.reason,
                    v.verbatim_quote,
                    v.cited_chunk_id,
                    v.quote_matched,
                    v.names_matched,
                    v.numbers_matched,
                    v.model_used,
                    s.title AS source_title,
                    s.url AS source_url,
                    s.domain AS source_domain,
                    s.quality_tier AS source_tier
                FROM claims c
                LEFT JOIN verdicts v ON v.claim_id = c.id
                LEFT JOIN chunks ch ON ch.id = v.cited_chunk_id
                LEFT JOIN sources s ON s.id = ch.source_id
                WHERE c.run_id = %s
                ORDER BY c.claim_index ASC, v.created_at DESC
                """,
                (run_id,),
            )
            claim_rows = cur.fetchall()

            cur.execute(
                """
                SELECT id, claim_a_id, claim_b_id, summary
                FROM contradictions
                WHERE run_id = %s
                ORDER BY created_at ASC
                """,
                (run_id,),
            )
            contradiction_rows = cur.fetchall()

            cur.execute(
                """
                SELECT stage, message, payload, created_at
                FROM audit_events
                WHERE run_id = %s
                ORDER BY id ASC
                """,
                (run_id,),
            )
            audit_rows = cur.fetchall()

    return {
        "run_id": str(run_row["id"]),
        "mode": run_row["mode"],
        "status": run_row["status"],
        "trust_score": run_row["trust_score"],
        "verdict_action": run_row["verdict_action"],
        "error_message": run_row["error_message"],
        "claims": [
            {
                "claim_id": str(r["claim_id"]),
                "claim_index": r["claim_index"],
                "claim_text": r["claim_text"],
                "status": r["status"],
                "retry_count": r["retry_count"],
                "verdict": r["verdict"],
                "reason": r["reason"],
                "verbatim_quote": r["verbatim_quote"],
                "cited_chunk_id": str(r["cited_chunk_id"]) if r["cited_chunk_id"] else None,
                "guardrails": {
                    "quote_matched": bool(r["quote_matched"]),
                    "names_matched": bool(r["names_matched"]),
                    "numbers_matched": bool(r["numbers_matched"]),
                },
                "model_used": r["model_used"],
                "source": {
                    "title": r["source_title"],
                    "url": r["source_url"],
                    "domain": r["source_domain"],
                    "quality_tier": r["source_tier"],
                }
                if r["source_title"]
                else None,
            }
            for r in claim_rows
        ],
        "contradictions": [
            {
                "id": str(c["id"]),
                "claim_a_id": str(c["claim_a_id"]),
                "claim_b_id": str(c["claim_b_id"]),
                "summary": c["summary"],
            }
            for c in contradiction_rows
        ],
        "audit_trail": [
            {
                "stage": a["stage"],
                "message": a["message"],
                "payload": a["payload"],
            }
            for a in audit_rows
        ],
    }


def delete_run(run_id: str) -> None:
    """Deletes a run and all child records via ON DELETE CASCADE."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM runs WHERE id = %s", (run_id,))
        conn.commit()

def list_recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    """Returns a lightweight summary list of the most recent verification runs."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, mode, input_text, status, verdict_action,
                       trust_score, created_at
                FROM runs
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "run_id": str(r["id"]),
            "mode": r["mode"],
            "input_preview": (r["input_text"] or "")[:120],
            "status": r["status"],
            "verdict_action": r["verdict_action"],
            "trust_score": r["trust_score"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]