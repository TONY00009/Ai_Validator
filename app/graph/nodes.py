from typing import Any

from app.db import repository
from app.graph.guardrails import (
    apply_deterministic_guardrails,
    compute_trust_score_and_action,
)
from app.graph.schemas import (
    ClaimVerificationOutput,
    ContradictionsOutput,
    ExtractedClaimsOutput,
    ValidatorState,
)
from app.llm.router import get_llm_router
from app.tools.evidence_store import (
    ingest_pdf_source,
    ingest_web_results,
    search_evidence,
)
from app.tools.web_search import search_web


def prepare_evidence_node(state: ValidatorState) -> dict[str, Any]:
    """Node 1: If running PDF audit with uploaded bytes, ingest PDF into pgvector."""
    run_id = state["run_id"]
    mode = state["mode"]
    pdf_bytes = state.get("pdf_bytes")
    pdf_filename = state.get("pdf_filename") or "uploaded_document.pdf"

    if mode == "pdf_audit" and pdf_bytes:
        src_id, num_chunks = ingest_pdf_source(
            run_id=run_id,
            title=pdf_filename,
            pdf_source=pdf_bytes,
        )
        repository.log_audit_event(
            run_id,
            stage="prepare_evidence",
            message=f"Ingested PDF '{pdf_filename}' into {num_chunks} vector chunks.",
            payload={"source_id": src_id, "chunks": num_chunks},
        )
    else:
        repository.log_audit_event(
            run_id,
            stage="prepare_evidence",
            message=f"Prepared evidence pipeline for mode '{mode}'.",
            payload={"mode": mode},
        )

    return {"retry_round": 0, "verdicts": {}, "claims_to_retry": []}


def extract_claims_node(state: ValidatorState) -> dict[str, Any]:
    """Node 2: Extracts atomic, verifiable factual claims using OpenAI (extractor role)."""
    run_id = state["run_id"]
    input_text = state["input_text"]
    router = get_llm_router()

    system_prompt = (
        "You are a precision fact-extraction engine. Break the user's text into "
        "short, standalone, verifiable factual claims (each containing concrete "
        "entities, numbers, dates, or events). Resolve pronouns ('it', 'they', "
        "'the company') to explicit names. Ignore opinions, greetings, and fluff. "
        "Return at most 10 claims."
    )
    extracted, model_used = router.generate_structured(
        role="extractor",
        system_prompt=system_prompt,
        user_prompt=f"Extract all verifiable factual claims from this text:\n\n{input_text}",
        response_model=ExtractedClaimsOutput,
    )

    clean_claims = [c.strip() for c in extracted.claims if c.strip()][:10]
    saved_claims = repository.save_claims(run_id, clean_claims)

    repository.log_audit_event(
        run_id,
        stage="extract_claims",
        message=f"Extracted {len(saved_claims)} atomic claims using {model_used}.",
        payload={"model_used": model_used, "claim_count": len(saved_claims)},
    )
    return {"claims": saved_claims}


def gather_web_evidence_node(state: ValidatorState) -> dict[str, Any]:
    """
    Node 3: Runs live web search (Tavily -> DuckDuckGo) for web_check / api_guardrail.
    On retry_round > 0, searches ONLY for the claims flagged in claims_to_retry.
    """
    run_id = state["run_id"]
    mode = state["mode"]
    retry_round = state.get("retry_round", 0)

    if mode == "pdf_audit":
        return {}

    claims = state.get("claims", [])
    retry_ids = set(state.get("claims_to_retry", []))

    target_claims = (
        [c for c in claims if c["claim_id"] in retry_ids]
        if retry_round > 0
        else claims
    )

    total_sources = 0
    total_chunks = 0

    for claim in target_claims:
        query = claim["claim_text"]
        if retry_round > 0:
            query = f"{query} official report data statistics"

        results = search_web(query, max_results=3)
        stats = ingest_web_results(run_id, results)
        total_sources += stats["sources_added"]
        total_chunks += stats["chunks_added"]

    repository.log_audit_event(
        run_id,
        stage="gather_web_evidence",
        message=(
            f"Web search round {retry_round}: searched {len(target_claims)} claim(s), "
            f"added {total_sources} new sources ({total_chunks} chunks)."
        ),
        payload={
            "retry_round": retry_round,
            "claims_searched": len(target_claims),
            "sources_added": total_sources,
            "chunks_added": total_chunks,
        },
    )
    return {}


def verify_claims_node(state: ValidatorState) -> dict[str, Any]:
    """
    Node 4: Verifies each pending/retry claim against pgvector chunks using Gemini
    (cross-model verifier), then enforces all 3 deterministic Python guardrails.
    """
    run_id = state["run_id"]
    mode = state["mode"]
    retry_round = state.get("retry_round", 0)
    max_retries = state.get("max_retries", 1)
    claims = state.get("claims", [])
    verdicts = dict(state.get("verdicts", {}))
    retry_ids = set(state.get("claims_to_retry", []))

    router = get_llm_router()
    next_retries: list[str] = []

    claims_to_check = (
        [c for c in claims if c["claim_id"] in retry_ids]
        if retry_round > 0
        else claims
    )

    system_prompt = (
        "You are a strict, skeptical fact-checking verifier. Evaluate the claim "
        "ONLY against the numbered evidence chunks provided below. Never use outside "
        "memory to mark a claim 'supported'.\n"
        "Rules:\n"
        "1. Mark 'supported' ONLY if an evidence chunk directly confirms the entities, "
        "numbers, and meaning of the claim.\n"
        "2. Mark 'rejected' if the evidence directly contradicts the claim (e.g., wrong "
        "number, wrong year, wrong company) or contains no supporting proof.\n"
        "3. Copy verbatim_quote EXACTLY word-for-word from the cited chunk.\n"
        "4. Set cited_chunk_number to the 1-based chunk number you quoted (or 0 if none).\n"
        "5. Set needs_more_evidence=true ONLY if the chunks do not address the claim "
        "and another web search might find the missing fact."
    )

    for claim in claims_to_check:
        claim_id = claim["claim_id"]
        claim_text = claim["claim_text"]

        chunks = search_evidence(run_id=run_id, query=claim_text, top_k=5)

        if not chunks:
            verdict_record = {
                "claim_id": claim_id,
                "verdict": "rejected",
                "reason": "No relevant evidence chunks were found for this claim.",
                "verbatim_quote": "",
                "cited_chunk_id": None,
                "quote_matched": False,
                "names_matched": False,
                "numbers_matched": False,
                "model_used": "none/no_evidence",
            }
            repository.save_verdict(
                **verdict_record,
                retry_count=retry_round,
            )
            verdicts[claim_id] = verdict_record
            if mode != "pdf_audit" and retry_round < max_retries:
                next_retries.append(claim_id)
            continue

        evidence_block = "\n\n".join(
            f"[Chunk {idx} | Tier {ch.quality_tier} | {ch.title}]\n{ch.content}"
            for idx, ch in enumerate(chunks, start=1)
        )
        user_prompt = f"CLAIM:\n{claim_text}\n\nEVIDENCE CHUNKS:\n{evidence_block}"

        llm_out, model_used = router.generate_structured(
            role="verifier",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ClaimVerificationOutput,
        )

        # Safely map 1-based chunk number to actual UUID and text
        cited_chunk = (
            chunks[llm_out.cited_chunk_number - 1]
            if 1 <= llm_out.cited_chunk_number <= len(chunks)
            else None
        )
        chunk_content = cited_chunk.content if cited_chunk else ""
        cited_chunk_id = cited_chunk.chunk_id if cited_chunk else None

        # Enforce all 3 deterministic Python guardrails
        guarded = apply_deterministic_guardrails(
            claim_text=claim_text,
            llm_verdict=llm_out.verdict,
            llm_reason=llm_out.reason,
            verbatim_quote=llm_out.verbatim_quote,
            chunk_content=chunk_content,
        )

        verdict_record = {
            "claim_id": claim_id,
            "verdict": guarded["verdict"],
            "reason": guarded["reason"],
            "verbatim_quote": llm_out.verbatim_quote,
            "cited_chunk_id": cited_chunk_id,
            "quote_matched": guarded["quote_matched"],
            "names_matched": guarded["names_matched"],
            "numbers_matched": guarded["numbers_matched"],
            "model_used": model_used,
        }
        repository.save_verdict(**verdict_record, retry_count=retry_round)
        verdicts[claim_id] = verdict_record

        if (
            mode != "pdf_audit"
            and retry_round < max_retries
            and llm_out.needs_more_evidence
        ):
            next_retries.append(claim_id)

    repository.log_audit_event(
        run_id,
        stage="verify_claims",
        message=(
            f"Verification round {retry_round}: checked {len(claims_to_check)} claim(s); "
            f"{len(next_retries)} queued for adaptive web retry."
        ),
        payload={
            "retry_round": retry_round,
            "checked": len(claims_to_check),
            "queued_for_retry": len(next_retries),
        },
    )

    return {
        "verdicts": verdicts,
        "claims_to_retry": next_retries,
        "retry_round": retry_round + 1 if next_retries else retry_round,
    }


def should_retry_web(state: ValidatorState) -> str:
    """Conditional edge: loops back to gather_web_evidence if claims need deeper search."""
    if state.get("mode") != "pdf_audit" and state.get("claims_to_retry"):
        return "retry_web"
    return "detect_contradictions"


def detect_contradictions_node(state: ValidatorState) -> dict[str, Any]:
    """Node 5: Detects mutually conflicting claims across the input text."""
    run_id = state["run_id"]
    claims = state.get("claims", [])
    verdicts = state.get("verdicts", {})

    if len(claims) < 2:
        return {"contradictions": []}

    router = get_llm_router()
    numbered_summary = "\n".join(
        f"{idx}. {c['claim_text']}"
        for idx, c in enumerate(claims, start=1)
    )

    system_prompt = (
        "You are a logical consistency auditor. Identify pairs of claims in the list "
        "that directly contradict each other (e.g., conflicting revenue figures, dates, "
        "or mutually exclusive statements about the same entity). Do not invent minor "
        "differences; return an empty list if all claims are mutually consistent."
    )
    detected, model_used = router.generate_structured(
        role="extractor",
        system_prompt=system_prompt,
        user_prompt=f"Analyze these claims for direct contradictions:\n\n{numbered_summary}",
        response_model=ContradictionsOutput,
    )

    seen_pairs: set[tuple[int, int]] = set()
    valid_contradictions: list[dict[str, Any]] = []

    for item in detected.contradictions:
        a_idx, b_idx = item.claim_a_number, item.claim_b_number
        if (
            1 <= a_idx <= len(claims)
            and 1 <= b_idx <= len(claims)
            and a_idx != b_idx
        ):
            pair = (min(a_idx, b_idx), max(a_idx, b_idx))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            valid_contradictions.append(
                {
                    "claim_a_id": claims[pair[0] - 1]["claim_id"],
                    "claim_b_id": claims[pair[1] - 1]["claim_id"],
                    "summary": item.summary.strip(),
                }
            )

    repository.save_contradictions(run_id, valid_contradictions)
    repository.log_audit_event(
        run_id,
        stage="detect_contradictions",
        message=f"Found {len(valid_contradictions)} contradiction(s) using {model_used}.",
        payload={"count": len(valid_contradictions), "model_used": model_used},
    )
    return {"contradictions": valid_contradictions}


def finalize_run_node(state: ValidatorState) -> dict[str, Any]:
    """Node 6: Computes deterministic Trust Score + PASS/FLAG/BLOCK and updates runs table."""
    run_id = state["run_id"]
    verdicts_list = list(state.get("verdicts", {}).values())
    contradictions = state.get("contradictions", [])

    trust_score, action = compute_trust_score_and_action(
        verdicts=verdicts_list,
        contradiction_count=len(contradictions),
    )

    repository.finalize_run_record(
        run_id=run_id,
        trust_score=trust_score,
        verdict_action=action,
    )
    repository.log_audit_event(
        run_id,
        stage="finalize_run",
        message=f"Run completed: Trust Score = {trust_score}/100 -> Action = {action}.",
        payload={"trust_score": trust_score, "verdict_action": action},
    )
    return {"trust_score": trust_score, "verdict_action": action}