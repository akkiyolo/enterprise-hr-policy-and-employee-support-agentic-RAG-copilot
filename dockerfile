# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage — resolve and install dependencies from pyproject.toml/uv.lock
# (the lockfile is the accurate dependency set; requirements.txt is stale)
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Build tools for any dependency that needs to compile from source
# (e.g. tokenizers/sentencepiece if no prebuilt wheel matches the platform)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first, separately from app code, for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Now copy the rest of the source and install the project itself
COPY . .
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Runtime stage — slim image, non-root user, only the built venv + source
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /app app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/api/health', timeout=3)" || exit 1

# Render sets $PORT at runtime; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]