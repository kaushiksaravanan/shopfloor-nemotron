# =============================================================================
# gym/Dockerfile — ShopFloor-Nemotron Gym (NeMo Gym FastAPI)
# Multi-stage build using uv for fast deterministic installs, with a slim
# python:3.11-slim runtime stage.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder — install Python deps into /opt/venv using uv
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# uv (astral.sh) — fast resolver/installer. Falls back to pip if uv install fails.
RUN pip install --no-cache-dir uv==0.4.* \
    && uv --version

# Create a venv we can copy verbatim into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install runtime deps. Pin tight; bump deliberately.
RUN uv pip install --python /opt/venv/bin/python \
    "fastapi==0.115.*" \
    "uvicorn[standard]==0.30.*" \
    "pydantic==2.9.*" \
    "python-multipart==0.0.*"

# -----------------------------------------------------------------------------
# Stage 2: runtime — slim image with just the venv + app code
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:${PATH}"

# Non-root user for the runtime.
RUN groupadd --system --gid 10001 gym \
    && useradd  --system --uid 10001 --gid gym --home /app gym \
    && mkdir -p /app \
    && chown -R gym:gym /app

# Copy the venv we built.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application code. eval/ holds verifiers + Pydantic schemas + CSVs that the
# gym server imports. gym/ holds the FastAPI app. Keep the COPY order tight so
# layer cache stays warm when only gym/ changes.
COPY --chown=gym:gym eval/ /app/eval/
COPY --chown=gym:gym gym/ /app/gym/

USER gym

EXPOSE 8000

# Healthcheck pings /healthz — the curl-less Python form avoids adding curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "gym.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
