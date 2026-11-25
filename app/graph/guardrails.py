import re
from typing import Any

from app.graph.schemas import VerdictAction, VerdictLabel

# Common capitalized words at the start of sentences that are NOT named entities
_STOP_ENTITIES = {
    "The", "This", "That", "These", "Those", "There", "Their", "They", "Then",
    "When", "Where", "Which", "While", "Who", "What", "Why", "How",
    "In", "On", "At", "By", "For", "From", "With", "Without", "Under", "Over",
    "After", "Before", "During", "Between", "Across", "About", "Into", "Through",
    "According", "However", "Although", "Because", "Since", "Also", "Moreover",
    "Furthermore", "Therefore", "Thus", "Meanwhile", "Additionally", "Overall",
    "Some", "Many", "Most", "Several", "Both", "Each", "Every", "All", "Any",
    "An", "As", "It", "Its", "He", "She", "We", "Our", "New", "Total",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
}


def _normalize_for_match(text: str) -> str:
    """Normalizes quotes, dashes, case, and whitespace for fair substring checks."""
    cleaned = text.lower()
    cleaned = (
        cleaned.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def check_quote_matched(verbatim_quote: str, chunk_content: str) -> bool:
    """
    Guardrail 1: Verifies that the LLM's verbatim_quote genuinely exists
    inside the cited source chunk (catches fabricated/paraphrased quotes).
    """
    norm_quote = _normalize_for_match(verbatim_quote).strip(" .,:;\"'")
    norm_chunk = _normalize_for_match(chunk_content)

    # Ignore empty or trivially short quotes (e.g., "yes" or "2025")
    if len(norm_quote) < 10 or not norm_chunk:
        return False

    return norm_quote in norm_chunk


def _extract_numbers(text: str) -> set[str]:
    """
    Extracts normalized numeric tokens (e.g., '348.5', '24', '2025', '1800')
    so '$1,800.0' and '1800' match cleanly.
    """
    raw_matches = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)
    normalized: set[str] = set()
    for token in raw_matches:
        no_commas = token.replace(",", "")
        try:
            num = float(no_commas)
            # Represent whole floats like 24.0 as '24' for exact token comparison
            normalized.add(str(int(num)) if num.is_integer() else f"{num:g}")
        except ValueError:
            normalized.add(no_commas)
    return normalized


def check_numbers_matched(claim_text: str, evidence_text: str) -> bool:
    """
    Guardrail 2: Verifies that every concrete number/year/percentage in the claim
    appears in the cited evidence (catches silent numeric hallucinations like 24% vs 42%).
    Returns True if the claim contains no numbers at all.
    """
    claim_nums = _extract_numbers(claim_text)
    if not claim_nums:
        return True

    evidence_nums = _extract_numbers(evidence_text)
    return claim_nums.issubset(evidence_nums)


def _extract_entities(text: str) -> set[str]:
    """
    Extracts capitalized proper nouns and acronyms (e.g., 'Acme', 'RBI', 'FY2025', 'Gujarat')
    using lightweight regex without requiring heavy NLP models.
    """
    # Matches capitalized words (len >= 2) or acronyms/codes like RBI, EV, FY2025
    tokens = re.findall(r"\b(?:[A-Z][a-zA-Z0-9]{1,}|[A-Z]{2,}[0-9]*)\b", text)
    return {tok for tok in tokens if tok not in _STOP_ENTITIES}


def check_names_matched(claim_text: str, evidence_text: str) -> bool:
    """
    Guardrail 3: Verifies that key entities (companies, people, places, codes)
    mentioned in the claim actually appear in the cited evidence.
    """
    claim_entities = _extract_entities(claim_text)
    if not claim_entities:
        return True

    evidence_lower = evidence_text.lower()
    matched = sum(1 for ent in claim_entities if ent.lower() in evidence_lower)
    # Require all entities to match for 1-2 entities, or at least 70% for longer lists
    if len(claim_entities) <= 2:
        return matched == len(claim_entities)
    return (matched / len(claim_entities)) >= 0.7


def apply_deterministic_guardrails(
    claim_text: str,
    llm_verdict: VerdictLabel,
    llm_reason: str,
    verbatim_quote: str,
    chunk_content: str,
) -> dict[str, Any]:
    """
    Runs all 3 deterministic Python checks on the Verifier LLM's output.
    If the LLM marked a claim 'supported' but failed quote or number verification,
    automatically downgrades the verdict to 'partial' so hallucinations never pass silently.
    """
    quote_ok = check_quote_matched(verbatim_quote, chunk_content)
    names_ok = check_names_matched(claim_text, chunk_content)
    numbers_ok = check_numbers_matched(claim_text, chunk_content)

    final_verdict: VerdictLabel = llm_verdict
    final_reason = llm_reason.strip()
    downgrade_reasons: list[str] = []

    if llm_verdict == "supported":
        if not quote_ok:
            downgrade_reasons.append("verbatim quote did not match cited source text")
        if not numbers_ok:
            downgrade_reasons.append("numeric figures in claim were not found in cited source")
        if not names_ok:
            downgrade_reasons.append("key named entities in claim were missing from cited source")

        if downgrade_reasons:
            final_verdict = "partial"
            final_reason += (
                " [Guardrail downgrade: " + "; ".join(downgrade_reasons) + ".]"
            )

    return {
        "verdict": final_verdict,
        "reason": final_reason,
        "quote_matched": quote_ok,
        "names_matched": names_ok,
        "numbers_matched": numbers_ok,
    }


def compute_trust_score_and_action(
    verdicts: list[dict[str, Any]],
    contradiction_count: int = 0,
) -> tuple[float, VerdictAction]:
    """
    Computes an auditable 0-100 Trust Score and maps it to a Guardrail Action:
      - PASS  : score >= 80, zero rejected claims, zero contradictions
      - BLOCK : score < 50 OR >= 40% of claims rejected
      - FLAG  : everything in between (needs human review)
    """
    if not verdicts:
        return 0.0, "FLAG"

    score_map = {"supported": 100.0, "partial": 50.0, "rejected": 0.0}
    total_points = sum(score_map.get(v["verdict"], 0.0) for v in verdicts)
    base_score = total_points / len(verdicts)

    # Penalize internal/source contradictions (-10 points each, max -30)
    penalty = min(contradiction_count * 10.0, 30.0)
    trust_score = round(max(0.0, base_score - penalty), 2)

    rejected_count = sum(1 for v in verdicts if v["verdict"] == "rejected")
    rejected_ratio = rejected_count / len(verdicts)

    if trust_score < 50.0 or rejected_ratio >= 0.4:
        action: VerdictAction = "BLOCK"
    elif trust_score >= 80.0 and rejected_count == 0 and contradiction_count == 0:
        action = "PASS"
    else:
        action = "FLAG"

    return trust_score, action