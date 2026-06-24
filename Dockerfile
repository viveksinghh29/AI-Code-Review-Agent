# ─────────────────────────────────────────────────────────────────────────────
# AI Code Review Agent — Dockerfile
# Multi-stage build: slim production image with only runtime dependencies
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install all dependencies into an isolated prefix
RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="AI Code Review Agent"
LABEL description="Autonomous AI-powered code review system"
LABEL version="1.0.0"

# Runtime system deps (git is required for GitPython to clone repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup . .

# Create necessary runtime directories
RUN mkdir -p /tmp/ai_code_review_repos reports \
 && chown -R appuser:appgroup /tmp/ai_code_review_repos reports

# Copy env template (actual secrets injected at runtime via env vars)
RUN cp .env.example .env.example

# Switch to non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Health check — verifies Streamlit is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
