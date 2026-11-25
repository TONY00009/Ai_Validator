from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.core.db import get_conn
from app.llm.router import get_llm_router
from app.tools.chunking import chunk_text, extract_text_from_pdf
from app.tools.source_tiers import classify_source_tier, extract_domain
from app.tools.web_search import WebSearchResult


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    chunk_index: int
    content: str
    similarity: float
    source_type: str
    title: str
    url: str | None = None
    domain: str | None = None
    quality_tier: int


def _format_vector(embedding: list[float]) -> str:
    """Formats a Python float list into pgvector literal syntax: '[0.12,-0.04,...]'."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


def ingest_text_source(
    run_id: str,
    source_type: Literal["pdf", "web"],
    title: str,
    text: str,
    url: str | None = None,
    domain: str | None = None,
    quality_tier: int | None = None,
) -> tuple[str | None, int]:
    """
    Chunks raw text, generates OpenAI embeddings, and saves the source + chunks
    into PostgreSQL (pgvector) in a single atomic transaction.
    Returns (source_id, number_of_chunks_saved).
    """
    chunks = chunk_text(text)
    if not chunks:
        return None, 0

    router = get_llm_router()
    embeddings = router.embed_texts(chunks)

    clean_domain = domain if domain is not None else (extract_domain(url) or None)
    tier = (
        quality_tier
        if quality_tier is not None
        else classify_source_tier(url, source_type=source_type)
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sources (run_id, source_type, title, url, domain, quality_tier)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (run_id, source_type, title, url, clean_domain, tier),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert source row into database.")
            source_id = str(row["id"])

            chunk_rows = [
                (run_id, source_id, idx, chunk_content, _format_vector(vec))
                for idx, (chunk_content, vec) in enumerate(zip(chunks, embeddings))
            ]
            cur.executemany(
                """
                INSERT INTO chunks (run_id, source_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                chunk_rows,
            )
        conn.commit()

    return source_id, len(chunks)


def ingest_pdf_source(
    run_id: str,
    title: str,
    pdf_source: str | Path | bytes,
) -> tuple[str | None, int]:
    """Reads a PDF (from path or upload bytes) and stores all embedded chunks."""
    text = extract_text_from_pdf(pdf_source)
    return ingest_text_source(
        run_id=run_id,
        source_type="pdf",
        title=title,
        text=text,
        url=None,
        domain="uploaded_pdf",
        quality_tier=1,
    )


def ingest_web_results(
    run_id: str,
    results: list[WebSearchResult],
) -> dict[str, int]:
    """
    Saves a batch of WebSearchResults into PostgreSQL, skipping any URLs
    that were already ingested for this run_id (prevents duplicates on retries).
    """
    if not results:
        return {"sources_added": 0, "chunks_added": 0}

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT url FROM sources WHERE run_id = %s AND url IS NOT NULL",
                (run_id,),
            )
            existing_urls = {row["url"] for row in cur.fetchall()}

    sources_added = 0
    chunks_added = 0

    for item in results:
        if item.url in existing_urls:
            continue

        source_id, num_chunks = ingest_text_source(
            run_id=run_id,
            source_type="web",
            title=item.title,
            text=item.content,
            url=item.url,
            domain=item.domain,
            quality_tier=item.quality_tier,
        )
        if source_id:
            existing_urls.add(item.url)
            sources_added += 1
            chunks_added += num_chunks

    return {"sources_added": sources_added, "chunks_added": chunks_added}


def search_evidence(
    run_id: str,
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.25,
) -> list[RetrievedChunk]:
    """
    Converts a claim/query into an embedding and retrieves the top_k most
    relevant chunks for this run_id using pgvector cosine similarity.
    """
    if not query.strip():
        return []

    router = get_llm_router()
    query_vec = router.embed_texts([query])[0]
    vec_literal = _format_vector(query_vec)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    c.source_id,
                    c.chunk_index,
                    c.content,
                    1 - (c.embedding <=> %s::vector) AS similarity,
                    s.source_type,
                    s.title,
                    s.url,
                    s.domain,
                    s.quality_tier
                FROM chunks c
                JOIN sources s ON c.source_id = s.id
                WHERE c.run_id = %s
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector ASC
                LIMIT %s
                """,
                (vec_literal, run_id, vec_literal, top_k),
            )
            rows = cur.fetchall()

    matches: list[RetrievedChunk] = []
    for row in rows:
        score = round(float(row["similarity"]), 4)
        if score < min_similarity:
            continue
        matches.append(
            RetrievedChunk(
                chunk_id=str(row["chunk_id"]),
                source_id=str(row["source_id"]),
                chunk_index=row["chunk_index"],
                content=row["content"],
                similarity=score,
                source_type=row["source_type"],
                title=row["title"] or "",
                url=row["url"],
                domain=row["domain"],
                quality_tier=row["quality_tier"],
            )
        )

    return matches