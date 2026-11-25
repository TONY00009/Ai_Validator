from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.db import repository
from app.graph.nodes import (
    detect_contradictions_node,
    extract_claims_node,
    finalize_run_node,
    gather_web_evidence_node,
    prepare_evidence_node,
    should_retry_web,
    verify_claims_node,
)
from app.graph.schemas import RunMode, ValidatorState

_compiled_graph = None


def get_validator_graph():
    """Builds and caches the compiled LangGraph verification state machine."""
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    builder = StateGraph(ValidatorState)

    builder.add_node("prepare_evidence", prepare_evidence_node)
    builder.add_node("extract_claims", extract_claims_node)
    builder.add_node("gather_web_evidence", gather_web_evidence_node)
    builder.add_node("verify_claims", verify_claims_node)
    builder.add_node("detect_contradictions", detect_contradictions_node)
    builder.add_node("finalize_run", finalize_run_node)

    builder.add_edge(START, "prepare_evidence")
    builder.add_edge("prepare_evidence", "extract_claims")
    builder.add_edge("extract_claims", "gather_web_evidence")
    builder.add_edge("gather_web_evidence", "verify_claims")

    builder.add_conditional_edges(
        "verify_claims",
        should_retry_web,
        {
            "retry_web": "gather_web_evidence",
            "detect_contradictions": "detect_contradictions",
        },
    )

    builder.add_edge("detect_contradictions", "finalize_run")
    builder.add_edge("finalize_run", END)

    _compiled_graph = builder.compile()
    return _compiled_graph


def run_validation_pipeline(
    mode: RunMode,
    input_text: str,
    pdf_bytes: bytes | None = None,
    pdf_filename: str | None = None,
    existing_run_id: str | None = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """
    Standard blocking entrypoint (for REST API clients, Swagger, and evals).
    Returns the final structured report fetched from PostgreSQL.
    """
    run_id = existing_run_id or repository.create_run(mode=mode, input_text=input_text)
    graph = get_validator_graph()

    initial_state: ValidatorState = {
        "run_id": run_id,
        "mode": mode,
        "input_text": input_text,
        "pdf_bytes": pdf_bytes,
        "pdf_filename": pdf_filename,
        "max_retries": max_retries,
    }

    try:
        graph.invoke(initial_state)
    except Exception as exc:
        repository.mark_run_failed(run_id, str(exc))
        raise

    return repository.get_run_report(run_id)


def stream_validation_pipeline(
    mode: RunMode,
    input_text: str,
    pdf_bytes: bytes | None = None,
    pdf_filename: str | None = None,
    existing_run_id: str | None = None,
    max_retries: int = 1,
) -> Iterator[dict[str, Any]]:
    """
    Streaming entrypoint using LangGraph's graph.stream(stream_mode='updates').
    Yields real-time status events as each agent node completes, ending with
    the full verification report.
    """
    run_id = existing_run_id or repository.create_run(mode=mode, input_text=input_text)
    graph = get_validator_graph()

    initial_state: ValidatorState = {
        "run_id": run_id,
        "mode": mode,
        "input_text": input_text,
        "pdf_bytes": pdf_bytes,
        "pdf_filename": pdf_filename,
        "max_retries": max_retries,
    }

    # Immediate first update so user gets feedback at t=0s
    if mode == "pdf_audit":
        yield {
            "event": "status",
            "stage": "prepare_evidence",
            "state": "running",
            "message": f"Parsing & embedding PDF '{pdf_filename or 'document.pdf'}' into pgvector...",
        }
    else:
        yield {
            "event": "status",
            "stage": "extract_claims",
            "state": "running",
            "message": "Extracting atomic factual claims using OpenAI...",
        }

    try:
        for update in graph.stream(initial_state, stream_mode="updates"):
            for node_name, raw_output in update.items():
                output = raw_output or {}

                if node_name == "prepare_evidence":
                    if mode == "pdf_audit":
                        yield {
                            "event": "status",
                            "stage": "prepare_evidence",
                            "state": "done",
                            "message": "PDF parsed and indexed into pgvector.",
                        }
                        yield {
                            "event": "status",
                            "stage": "extract_claims",
                            "state": "running",
                            "message": "Extracting atomic factual claims using OpenAI...",
                        }

                elif node_name == "extract_claims":
                    count = len(output.get("claims", []))
                    yield {
                        "event": "status",
                        "stage": "extract_claims",
                        "state": "done",
                        "message": f"Extracted {count} verifiable factual claim(s).",
                    }
                    if mode == "pdf_audit":
                        yield {
                            "event": "status",
                            "stage": "verify_claims",
                            "state": "running",
                            "message": f"Checking {count} claim(s) against PDF chunks & applying guardrails...",
                        }
                    else:
                        yield {
                            "event": "status",
                            "stage": "gather_web_evidence",
                            "state": "running",
                            "message": f"Collecting live web evidence for {count} claim(s)...",
                        }

                elif node_name == "gather_web_evidence":
                    if mode != "pdf_audit":
                        yield {
                            "event": "status",
                            "stage": "gather_web_evidence",
                            "state": "done",
                            "message": "Collected & embedded live web evidence into pgvector.",
                        }
                        yield {
                            "event": "status",
                            "stage": "verify_claims",
                            "state": "running",
                            "message": "Verifying facts with Gemini & running deterministic Python guardrails...",
                        }

                elif node_name == "verify_claims":
                    checked = len(output.get("verdicts", {}))
                    retries = output.get("claims_to_retry", [])
                    yield {
                        "event": "status",
                        "stage": "verify_claims",
                        "state": "done",
                        "message": f"Verified {checked} claim(s) & applied guardrails (Quote, Names, Numbers).",
                    }
                    if mode != "pdf_audit" and retries:
                        yield {
                            "event": "status",
                            "stage": "gather_web_evidence",
                            "state": "running",
                            "message": f"Adaptive retry: searching deeper sources for {len(retries)} claim(s)...",
                        }
                    else:
                        yield {
                            "event": "status",
                            "stage": "detect_contradictions",
                            "state": "running",
                            "message": "Scanning claims for internal contradictions...",
                        }

                elif node_name == "detect_contradictions":
                    c_count = len(output.get("contradictions", []))
                    yield {
                        "event": "status",
                        "stage": "detect_contradictions",
                        "state": "done",
                        "message": f"Contradiction scan complete ({c_count} conflict(s) found).",
                    }
                    yield {
                        "event": "status",
                        "stage": "finalize_run",
                        "state": "running",
                        "message": "Computing final Trust Score & Guardrail Decision...",
                    }

                elif node_name == "finalize_run":
                    score = output.get("trust_score", 0)
                    action = output.get("verdict_action", "FLAG")
                    yield {
                        "event": "status",
                        "stage": "finalize_run",
                        "state": "done",
                        "message": f"Verification complete: {action} (Trust Score: {score}/100).",
                    }

        report = repository.get_run_report(run_id)
        yield {"event": "result", "report": report}

    except Exception as exc:
        repository.mark_run_failed(run_id, str(exc))
        yield {"event": "error", "detail": str(exc)}