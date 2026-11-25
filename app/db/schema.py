from app.core.db import close_pool, get_conn

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT UNIQUE NOT NULL,
    name            TEXT,
    google_sub      TEXT UNIQUE,
    daily_quota     INTEGER NOT NULL DEFAULT 5,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    mode            TEXT NOT NULL CHECK (mode IN ('pdf_audit', 'web_check', 'api_guardrail')),
    input_text      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    verdict_action  TEXT CHECK (verdict_action IN ('PASS', 'FLAG', 'BLOCK')),
    trust_score     DOUBLE PRECISION,
    is_public       BOOLEAN NOT NULL DEFAULT FALSE,
    share_token     TEXT UNIQUE,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS sources (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    source_type     TEXT NOT NULL CHECK (source_type IN ('pdf', 'web')),
    title           TEXT,
    url             TEXT,
    domain          TEXT,
    quality_tier    SMALLINT NOT NULL DEFAULT 3 CHECK (quality_tier IN (1, 2, 3)),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    source_id       UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    embedding       vector(1536),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_run_id ON chunks(run_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS claims (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    claim_index     INTEGER NOT NULL,
    claim_text      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'supported', 'partial', 'rejected')),
    retry_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS verdicts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id        UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    verdict         TEXT NOT NULL CHECK (verdict IN ('supported', 'partial', 'rejected')),
    reason          TEXT NOT NULL,
    verbatim_quote  TEXT,
    cited_chunk_id  UUID REFERENCES chunks(id) ON DELETE SET NULL,
    quote_matched   BOOLEAN NOT NULL DEFAULT FALSE,
    names_matched   BOOLEAN NOT NULL DEFAULT FALSE,
    numbers_matched BOOLEAN NOT NULL DEFAULT FALSE,
    model_used      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS contradictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    claim_a_id      UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    claim_b_id      UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    summary         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_events (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage           TEXT NOT NULL,
    message         TEXT NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_events_run_id ON audit_events(run_id, id);
"""


def init_db() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
    print("✅ Ai_Validator database schema and pgvector extension initialized successfully!")


if __name__ == "__main__":
    try:
        init_db()
    finally:
        close_pool()