from app.core.db import close_pool, get_conn
from app.tools.chunking import chunk_text
from app.tools.evidence_store import (
    ingest_text_source,
    ingest_web_results,
    search_evidence,
)
from app.tools.source_tiers import classify_source_tier, get_tier_label
from app.tools.web_search import search_web


def run_tools_smoke_test() -> None:
    print("🚀 Running Milestone 2 End-to-End Evidence Tools Smoke Test...\n")

    # 1. Test Source Quality Tiers
    sample_urls = [
        "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
        "https://www.reuters.com/world/india/economy-update",
        "https://random-finance-blog.example.com/post-1",
    ]
    print("1️⃣  Testing Source Quality Tiers:")
    for url in sample_urls:
        tier = classify_source_tier(url, source_type="web")
        print(f"   ✅ {url[:50]:50} -> {get_tier_label(tier)}")

    # 2. Test Smart Chunker
    sample_doc = (
        "[Page 1]\n"
        "Acme Solar reported FY2025 revenue of $348.5 million, representing a 24% "
        "year-over-year increase driven by utility-scale battery deployments in Gujarat.\n\n"
        "[Page 2]\n"
        "The company installed 1.8 GW of new capacity across India and reduced manufacturing "
        "costs by 11.4% compared to FY2024."
    )
    pieces = chunk_text(sample_doc, chunk_size=180, chunk_overlap=30)
    print(f"\n2️⃣  Testing Smart Chunker:\n   ✅ Split sample doc into {len(pieces)} clean chunks.")

    # 3. Test Live Web Search (Tavily primary -> DuckDuckGo fallback)
    print("\n3️⃣  Testing Live Web Search:")
    web_results = search_web("India GDP growth rate official World Bank RBI", max_results=3)
    if web_results:
        print(
            f"   ✅ Retrieved {len(web_results)} web results via '{web_results[0].provider}':"
        )
        for r in web_results:
            print(f"      - [Tier {r.quality_tier}] {r.domain}: {r.title[:60]}")
    else:
        print("   ⚠️  No web results returned (check TAVILY_API_KEY or internet).")

    # 4. Test PostgreSQL + pgvector Ingestion & Semantic Search
    print("\n4️⃣  Testing PostgreSQL + pgvector Ingestion & Vector Search:")
    run_id: str | None = None
    try:
        # Create a temporary test run
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runs (mode, input_text, status)
                    VALUES ('pdf_audit', 'Milestone 2 smoke test run', 'running')
                    RETURNING id
                    """
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("Could not create test run.")
                run_id = str(row["id"])
            conn.commit()
        print(f"   ✅ Created temporary run_id: {run_id}")

        # Ingest sample PDF-style text
        src_id, pdf_chunks = ingest_text_source(
            run_id=run_id,
            source_type="pdf",
            title="Acme_Solar_FY2025.pdf",
            text=sample_doc,
            domain="uploaded_pdf",
            quality_tier=1,
        )
        print(f"   ✅ Ingested PDF source ({src_id[:8]}...) -> {pdf_chunks} embedded chunks.")

        # Ingest live web results (plus test deduplication by calling twice!)
        first_stats = ingest_web_results(run_id, web_results)
        dup_stats = ingest_web_results(run_id, web_results)
        print(
            f"   ✅ Ingested Web sources -> added {first_stats['sources_added']} sources / "
            f"{first_stats['chunks_added']} chunks (duplicate retry added: {dup_stats['sources_added']})."
        )

        # Semantic Vector Search for a specific claim
        query = "What was Acme Solar's revenue in FY2025?"
        matches = search_evidence(run_id=run_id, query=query, top_k=3)
        print(f"\n   🔍 Vector Search Query: '{query}'")
        for idx, m in enumerate(matches, start=1):
            preview = m.content.replace("\n", " ")[:95]
            print(
                f"      {idx}. [sim={m.similarity:.4f} | Tier {m.quality_tier} | {m.source_type}] "
                f"{preview}..."
            )
    finally:
        # Clean up test run (ON DELETE CASCADE removes all test sources & chunks automatically)
        if run_id is not None:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM runs WHERE id = %s", (run_id,))
                conn.commit()
            print(f"\n🧹 Cleaned up temporary test run ({run_id[:8]}...) via ON DELETE CASCADE.")
        close_pool()


if __name__ == "__main__":
    run_tools_smoke_test()