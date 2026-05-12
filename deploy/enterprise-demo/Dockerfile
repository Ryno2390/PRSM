# Sprint 318a — Dockerfile for the §7 enterprise +
# federated inference demo runnable.
#
# Slim image (no Postgres / Redis / IPFS / libtorrent —
# the demo runs entirely in-process). Default CMD runs
# the bringup-cli `demo` subcommand as a post-bringup
# health check.
#
# Build from the repo root:
#   docker build -f deploy/enterprise-demo/Dockerfile \
#       -t prsm-enterprise-demo:latest .
# Or via docker-compose:
#   docker compose -f deploy/enterprise-demo/docker-compose.yml build

# ── Stage 1: build venv ─────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for cryptography wheel (libffi, openssl)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libffi-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create venv + install deps. Torch CPU wheel is ~200MB
# but lands once + caches in the builder layer.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY deploy/enterprise-demo/requirements-enterprise-demo.txt /tmp/req.txt
RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r /tmp/req.txt && \
    rm /tmp/req.txt

# ── Stage 2: runtime ───────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Runtime libs needed by torch + cryptography
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy only the prsm package source. Slim __init__.py
# files at the package boundary REPLACE the repo-canonical
# ones to keep transitive imports off the demo image —
# the canonical inits eagerly pull prsm.core / prsm.node /
# prsm.economy / prsm.storage which the demo doesn't need.
WORKDIR /app
COPY deploy/enterprise-demo/prsm_init_slim.py /app/prsm/__init__.py
COPY prsm/compute /app/prsm/compute
COPY prsm/enterprise /app/prsm/enterprise
COPY deploy/enterprise-demo/inference_init_slim.py /app/prsm/compute/inference/__init__.py

# Some prsm.compute / prsm.enterprise modules import from
# adjacent prsm.* subpackages — add empty __init__ for any
# that would otherwise fail to import.  (Currently nothing
# in the demo path needs this, but the shim keeps the
# package tree well-formed if a future sprint adds one.)
RUN find /app/prsm -type d -not -path '*/\.*' \
        -exec test ! -f {}/__init__.py \; -print \
        | xargs -I{} touch {}/__init__.py

# Persistence base dir — matches the bringup CLI's default
RUN mkdir -p /var/lib/prsm
VOLUME ["/var/lib/prsm"]

# Health check: bringup status returns rc=1 on incomplete
# config. For the demo image, we run `demo` which exits
# rc=0 only when both flows verify end-to-end.
HEALTHCHECK --interval=30s --timeout=120s --start-period=10s --retries=2 \
    CMD python -m prsm.enterprise.bringup_cli demo \
        --persist-root /var/lib/prsm/health || exit 1

# Default command runs the demo as proof-of-life. Operators
# pivot to `bringup status` / `generate` / their own
# entrypoint via `docker run ... <command>`.
CMD ["python", "-m", "prsm.enterprise.bringup_cli", "demo"]
