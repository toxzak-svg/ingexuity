# ============================================================================
# IngExuity Dockerfile — multi-stage build
# Stage 1: prep (downloads the GGUF model + Julia precompiles once)
# Stage 2: runtime (copies prewarmed artifacts; no network at build)
# ============================================================================

# ---------- Stage 1: prep ----------
FROM julia:1.12 AS prep

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Julia project + source
COPY Project.toml .
COPY src/ src/

# Resolve and precompile Julia deps (slow; do it once here, not at startup)
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
 && julia --project=. -e 'using IngExuity' || true

# Download the Llama 3.2 1B Q4_K_M GGUF (~700MB) so the runtime image
# carries it. start.sh will still re-validate and skip if present.
RUN mkdir -p /app/models
RUN curl -L -o /app/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ---------- Stage 2: runtime ----------
FROM julia:1.12

WORKDIR /app

ENV PORT=8000
ENV JULIA_PKG_PRECOMPILE_AUTO=0

# System deps for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy prep stage artifacts (deps already precompiled, model already downloaded)
COPY --from=prep /app/Project.toml /app/Manifest.toml /app/
COPY --from=prep /app/src /app/src
COPY --from=prep /app/models /app/models
COPY --from=prep /app/start.sh /app/start.sh
COPY --from=prep /root/.julia /root/.julia

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["/app/start.sh"]
