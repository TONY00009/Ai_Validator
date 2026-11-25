from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

VerdictLabel = Literal["supported", "partial", "rejected"]
VerdictAction = Literal["PASS", "FLAG", "BLOCK"]
RunMode = Literal["pdf_audit", "web_check", "api_guardrail"]


# --- 1. LLM Output Schemas (Strict JSON Contracts) ---
class ExtractedClaimsOutput(BaseModel):
    claims: list[str] = Field(
        description=(
            "List of atomic, standalone, verifiable factual claims extracted "
            "from the input text. Ignore opinions or greetings. Max 10 claims."
        )
    )


class ClaimVerificationOutput(BaseModel):
    verdict: VerdictLabel = Field(
        description=(
            "'supported' if evidence directly proves the claim, "
            "'partial' if only partly supported or ambiguous, "
            "'rejected' if evidence directly contradicts the claim or no proof exists."
        )
    )
    reason: str = Field(
        description="Concise 1-2 sentence explanation grounded strictly in the evidence."
    )
    verbatim_quote: str = Field(
        default="",
        description=(
            "Exact word-for-word substring copied verbatim from the cited evidence chunk "
            "that proves or refutes the claim. Empty string if no relevant quote exists."
        ),
    )
    cited_chunk_number: int = Field(
        default=0,
        description=(
            "The 1-based number of the evidence chunk (e.g., 1, 2, 3) from which "
            "verbatim_quote was copied. Use 0 if no chunk supports or refutes the claim."
        ),
    )
    needs_more_evidence: bool = Field(
        default=False,
        description=(
            "Set to true ONLY if the provided evidence chunks do not contain enough "
            "information to confirm or refute the claim and an additional web search could help."
        ),
    )


class ContradictionItem(BaseModel):
    claim_a_number: int = Field(description="1-based index of the first conflicting claim.")
    claim_b_number: int = Field(description="1-based index of the second conflicting claim.")
    summary: str = Field(
        description="Clear 1-sentence explanation of how these two claims contradict each other."
    )


class ContradictionsOutput(BaseModel):
    contradictions: list[ContradictionItem] = Field(
        default_factory=list,
        description="Pairs of claims that conflict with each other or with verified evidence.",
    )


# --- 2. LangGraph Shared State ---
class ValidatorState(TypedDict, total=False):
    run_id: str
    mode: RunMode
    input_text: str
    pdf_bytes: bytes | None
    pdf_filename: str | None

    # Populated step-by-step by graph nodes
    claims: list[dict[str, Any]]          # [{"claim_id", "claim_index", "claim_text", "retry_count"}]
    verdicts: dict[str, dict[str, Any]]   # keyed by claim_id
    claims_to_retry: list[str]            # claim_ids needing deeper web search
    retry_round: int
    max_retries: int
    contradictions: list[dict[str, Any]]
    trust_score: float
    verdict_action: VerdictAction