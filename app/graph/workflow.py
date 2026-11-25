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
    Main entrypoint for executing a full fact-checking run.
    Returns the structured report fetched from PostgreSQL.
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