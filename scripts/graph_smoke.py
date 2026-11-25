from app.core.db import close_pool
from app.db import repository
from app.graph.workflow import run_validation_pipeline
from app.tools.evidence_store import ingest_text_source


def run_graph_smoke_test() -> None:
    print("🚀 Running Milestone 3 LangGraph End-to-End Verification Smoke Test...\n")

    ground_truth_doc = (
        "[Page 1]\n"
        "Acme Solar reported FY2025 revenue of $348.5 million, representing a 24% "
        "year-over-year increase driven by utility-scale battery deployments in Gujarat.\n\n"
        "[Page 2]\n"
        "The company installed 1.8 GW of new capacity across India and reduced "
        "manufacturing costs by 11.4% compared to FY2024. CEO Meera Nair announced "
        "no new debt was issued during the fiscal year."
    )

    # Deliberately mix 1 true claim, 1 hallucinated figure, and 1 direct contradiction
    ai_generated_summary = (
        "Acme Solar reported FY2025 revenue of $348.5 million, driven by battery "
        "deployments in Gujarat. However, later in the year Acme Solar's total FY2025 "
        "revenue was reported as $512 million, and the company cut manufacturing "
        "costs by 45% under CEO Rajesh Verma."
    )

    run_id: str | None = None
    try:
        run_id = repository.create_run(mode="pdf_audit", input_text=ai_generated_summary)
        ingest_text_source(
            run_id=run_id,
            source_type="pdf",
            title="Acme_Solar_FY2025_Annual_Report.pdf",
            text=ground_truth_doc,
            domain="uploaded_pdf",
            quality_tier=1,
        )

        report = run_validation_pipeline(
            mode="pdf_audit",
            input_text=ai_generated_summary,
            existing_run_id=run_id,
        )

        print(f"📊 Status         : {report['status']}")
        print(f"🛡️  Verdict Action : {report['verdict_action']}")
        print(f"🎯 Trust Score    : {report['trust_score']} / 100\n")

        print("🔎 Claims & Deterministic Guardrails:")
        for c in report["claims"]:
            g = c["guardrails"]
            print(
                f"   {c['claim_index']}. [{c['verdict'].upper():9}] {c['claim_text']}\n"
                f"      Model      : {c['model_used']}\n"
                f"      Guardrails : quote={g['quote_matched']} | "
                f"names={g['names_matched']} | numbers={g['numbers_matched']}\n"
                f"      Quote      : {(c['verbatim_quote'] or 'None')[:90]}\n"
                f"      Reason     : {c['reason']}\n"
            )

        print(f"⚠️  Contradictions Detected: {len(report['contradictions'])}")
        for item in report["contradictions"]:
            print(f"   - {item['summary']}")

        print(f"\n📜 Audit Trail ({len(report['audit_trail'])} events):")
        for ev in report["audit_trail"]:
            print(f"   - [{ev['stage']}] {ev['message']}")
    finally:
        if run_id is not None:
            repository.delete_run(run_id)
            print(f"\n🧹 Cleaned up temporary run ({run_id[:8]}...).")
        close_pool()


if __name__ == "__main__":
    run_graph_smoke_test()