import argparse
import json
from pathlib import Path
import time
from typing import Any

from pydantic import BaseModel, Field

from app.core.db import close_pool
from app.db import repository
from app.graph.workflow import run_validation_pipeline
from app.llm.router import get_llm_router
from app.tools.evidence_store import ingest_text_source
from eval.dataset import EVAL_CASES, EvalCase

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


class JudgeEvaluation(BaseModel):
    reasoning_quality: int = Field(
        ge=1,
        le=5,
        description="1 (poor) to 5 (excellent) score for clarity and factual grounding of claim reasons.",
    )
    decision_justified: bool = Field(
        description="True if the system's final PASS/FLAG/BLOCK action is fair given the ground truth.",
    )
    critique: str = Field(
        description="Concise 1-sentence independent audit note from the judge.",
    )


def _judge_with_groq(case: EvalCase, report: dict[str, Any]) -> tuple[JudgeEvaluation, str]:
    """Uses the 3rd independent model family (Groq/Qwen) to audit the pipeline's output."""
    router = get_llm_router()

    claims_summary = "\n".join(
        f"- Claim: {c['claim_text']} | Verdict: {c['verdict']} | "
        f"Guardrails: {c['guardrails']} | Reason: {c['reason']}"
        for c in report.get("claims", [])
    )

    system_prompt = (
        "You are an independent MLOps evaluation judge auditing an AI fact-checking system. "
        "Compare the system's verdict, trust score, and claim-level reasons against the "
        "ground truth and expected behavior. Be strict and objective."
    )
    user_prompt = (
        f"CASE CATEGORY: {case.category}\n"
        f"EXPECTED ACTION: {case.expected_action}\n\n"
        f"GROUND TRUTH:\n{case.ground_truth}\n\n"
        f"INPUT SUMMARY TESTED:\n{case.input_text}\n\n"
        f"SYSTEM OUTPUT:\n"
        f"Final Action: {report.get('verdict_action')} | Trust Score: {report.get('trust_score')}/100\n"
        f"Contradictions Found: {len(report.get('contradictions', []))}\n"
        f"Claims:\n{claims_summary}"
    )

    return router.generate_structured(
        role="eval_judge",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=JudgeEvaluation,
    )


def evaluate_case(case: EvalCase, use_judge: bool = True) -> dict[str, Any]:
    run_id: str | None = None
    start_time = time.perf_counter()

    try:
        run_id = repository.create_run(mode="pdf_audit", input_text=case.input_text)
        ingest_text_source(
            run_id=run_id,
            source_type="pdf",
            title=f"{case.case_id}_ground_truth.pdf",
            text=case.ground_truth,
            domain="uploaded_pdf",
            quality_tier=1,
        )

        report = run_validation_pipeline(
            mode="pdf_audit",
            input_text=case.input_text,
            existing_run_id=run_id,
            max_retries=0,
        )
        latency_sec = round(time.perf_counter() - start_time, 2)

        actual_action = report.get("verdict_action")
        exact_match = actual_action == case.expected_action

        # Safety metric: did a hallucinated/mixed input get caught (not allowed to PASS)?
        if case.expected_action == "PASS":
            hallucination_safe = actual_action == "PASS"
        else:
            hallucination_safe = actual_action in ("FLAG", "BLOCK")

        contradictions_found = len(report.get("contradictions", []))
        contradiction_hit = (
            contradictions_found > 0 if case.expect_contradiction else contradictions_found == 0
        )

        # Track whether deterministic Python guardrails flagged any mismatch on claims
        claims = report.get("claims", [])
        number_flags = sum(1 for c in claims if not c["guardrails"]["numbers_matched"])
        name_flags = sum(1 for c in claims if not c["guardrails"]["names_matched"])
        quote_flags = sum(1 for c in claims if not c["guardrails"]["quote_matched"])

        judge_data: dict[str, Any] | None = None
        if use_judge:
            judge_out, judge_model = _judge_with_groq(case, report)
            judge_data = {
                "judge_model": judge_model,
                "reasoning_quality": judge_out.reasoning_quality,
                "decision_justified": judge_out.decision_justified,
                "critique": judge_out.critique,
            }

        return {
            "case_id": case.case_id,
            "category": case.category,
            "expected_action": case.expected_action,
            "actual_action": actual_action,
            "trust_score": report.get("trust_score"),
            "exact_match": exact_match,
            "hallucination_safe": hallucination_safe,
            "expect_contradiction": case.expect_contradiction,
            "contradictions_found": contradictions_found,
            "contradiction_hit": contradiction_hit,
            "claim_count": len(claims),
            "guardrail_flags": {
                "numbers_mismatched": number_flags,
                "names_mismatched": name_flags,
                "quotes_mismatched": quote_flags,
            },
            "latency_sec": latency_sec,
            "judge": judge_data,
            "error": None,
        }
    except Exception as exc:
        return {
            "case_id": case.case_id,
            "category": case.category,
            "expected_action": case.expected_action,
            "actual_action": "ERROR",
            "trust_score": None,
            "exact_match": False,
            "hallucination_safe": False,
            "expect_contradiction": case.expect_contradiction,
            "contradictions_found": 0,
            "contradiction_hit": False,
            "claim_count": 0,
            "guardrail_flags": {},
            "latency_sec": round(time.perf_counter() - start_time, 2),
            "judge": None,
            "error": str(exc),
        }
    finally:
        if run_id is not None:
            repository.delete_run(run_id)


def run_eval_suite(limit: int | None = None, skip_judge: bool = False) -> None:
    cases = EVAL_CASES[:limit] if limit else EVAL_CASES
    print(f"🧪 Running Ai_Validator Evaluation Suite ({len(cases)} test cases)...\n")

    results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        res = evaluate_case(case, use_judge=not skip_judge)
        results.append(res)

        icon = "✅" if res["exact_match"] else ("⚠️ " if res["hallucination_safe"] else "❌")
        judge_str = (
            f" | Groq Judge: {res['judge']['reasoning_quality']}/5"
            if res["judge"]
            else ""
        )
        print(
            f"[{idx:02d}/{len(cases):02d}] {icon} {case.case_id} ({case.category:21}) | "
            f"Expected: {case.expected_action:5} | Actual: {str(res['actual_action']):5} | "
            f"Score: {str(res['trust_score']):>5} | {res['latency_sec']}s{judge_str}"
        )
        if res["error"]:
            print(f"        ERROR: {res['error']}")

    # Aggregate Metrics
    total = len(results)
    exact_matches = sum(1 for r in results if r["exact_match"])

    non_pass_cases = [r for r in results if r["expected_action"] != "PASS"]
    hallucinations_caught = sum(1 for r in non_pass_cases if r["hallucination_safe"])

    pass_cases = [r for r in results if r["expected_action"] == "PASS"]
    clean_pass_ok = sum(1 for r in pass_cases if r["exact_match"])

    contradiction_cases = [r for r in results if r["expect_contradiction"]]
    contradictions_caught = sum(1 for r in contradiction_cases if r["contradiction_hit"])

    judged = [r["judge"] for r in results if r["judge"]]
    avg_judge_score = (
        round(sum(j["reasoning_quality"] for j in judged) / len(judged), 2)
        if judged
        else None
    )
    judge_justified_pct = (
        round(100.0 * sum(1 for j in judged if j["decision_justified"]) / len(judged), 1)
        if judged
        else None
    )
    avg_latency = round(sum(r["latency_sec"] for r in results) / total, 2) if total else 0.0

    summary = {
        "total_cases": total,
        "exact_action_accuracy_pct": round(100.0 * exact_matches / total, 1) if total else 0.0,
        "hallucination_catch_rate_pct": (
            round(100.0 * hallucinations_caught / len(non_pass_cases), 1)
            if non_pass_cases
            else None
        ),
        "clean_pass_accuracy_pct": (
            round(100.0 * clean_pass_ok / len(pass_cases), 1) if pass_cases else None
        ),
        "contradiction_catch_rate_pct": (
            round(100.0 * contradictions_caught / len(contradiction_cases), 1)
            if contradiction_cases
            else None
        ),
        "avg_groq_judge_score_out_of_5": avg_judge_score,
        "groq_judge_justified_pct": judge_justified_pct,
        "avg_latency_sec": avg_latency,
    }

    output_payload = {"summary": summary, "cases": results}
    RESULTS_PATH.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print("📊 EVALUATION BENCHMARK SUMMARY")
    print("=" * 68)
    print(f"   Total Test Cases             : {summary['total_cases']}")
    print(f"   Exact Action Accuracy        : {summary['exact_action_accuracy_pct']}% ({exact_matches}/{total})")
    if summary["hallucination_catch_rate_pct"] is not None:
        print(
            f"   Hallucination Catch Rate     : {summary['hallucination_catch_rate_pct']}% "
            f"({hallucinations_caught}/{len(non_pass_cases)})"
        )
    if summary["clean_pass_accuracy_pct"] is not None:
        print(
            f"   Clean Grounded PASS Accuracy : {summary['clean_pass_accuracy_pct']}% "
            f"({clean_pass_ok}/{len(pass_cases)})"
        )
    if summary["contradiction_catch_rate_pct"] is not None:
        print(
            f"   Contradiction Catch Rate     : {summary['contradiction_catch_rate_pct']}% "
            f"({contradictions_caught}/{len(contradiction_cases)})"
        )
    if avg_judge_score is not None:
        print(
            f"   3rd-Party Groq Judge Rating  : {avg_judge_score} / 5.0 "
            f"({judge_justified_pct}% decisions justified)"
        )
    print(f"   Average End-to-End Latency   : {avg_latency}s / case")
    print(f"\n💾 Full report saved to: {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ai_Validator automated evaluation suite.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases (e.g., --limit 3 for a quick sanity check).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the Groq LLM-as-a-Judge step to save time/tokens.",
    )
    args = parser.parse_args()

    try:
        run_eval_suite(limit=args.limit, skip_judge=args.skip_judge)
    finally:
        close_pool()


if __name__ == "__main__":
    main()