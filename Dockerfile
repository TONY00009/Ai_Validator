FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# 1. Copy dependency files first so Docker caches installed packages
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Copy application code
COPY app ./app
COPY eval ./eval
COPY scripts ./scripts

EXPOSE 8000

# 3. Auto-initialize PostgreSQL schema + pgvector, then start FastAPI server
CMD ["sh", "-c", "python -m app.db.schema && uvicorn app.main:app --host 0.0.0.0 --port 8000"]