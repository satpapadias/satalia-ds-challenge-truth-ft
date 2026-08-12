# One image, six services.
#
# Each service in docker-compose.yml is its own container and its own deployable
# unit; they differ by command and environment, not by image contents. The A2A
# boundaries are real process boundaries either way, and a single image means one
# dependency resolution and one thing to keep reproducible.
#
# The separations that matter are enforced by configuration rather than by what
# is on disk:
#   - only model-tools receives the provider credential
#   - only the orchestrator receives its peers' addresses, and it is NOT given
#     the model-tools address, so it cannot bypass the predictor agents
# If those boundaries ever need to be physical, this file grows per-service
# stages; nothing else changes.

FROM python:3.12-slim AS base

# uv resolves from the committed lockfile, so an image build installs the same
# versions the results were produced with rather than re-resolving.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so editing source does not invalidate the install layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra mcp --extra a2a

COPY truthclf/ ./truthclf/
COPY truthclf_mcp/ ./truthclf_mcp/
COPY truthclf_agents/ ./truthclf_agents/
RUN uv sync --frozen --extra mcp --extra a2a

# Data and artifacts. Copied rather than mounted so a container is self-contained
# and behaves identically wherever it runs.
#   data.csv               -> data-tools only, but see the note above on images
#   ft_eval_cache.json     -> the recorded fine-tuned probabilities
#   ft_eval_identity.json  -> which statement each of those belongs to; without
#                             it the stored path refuses to serve at all
#   results/calibrators/   -> the fitted calibrator artifacts
COPY data.csv ft_eval_cache.json ft_eval_identity.json ./
COPY results/calibrators/ ./results/calibrators/

# Run as a non-root user. The response cache is the only thing written at
# runtime, and it lives under the source tree, so its parent must be writable.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/.llm_cache \
    && chown -R app:app /app
USER app

# Overridden per service in docker-compose.yml.
CMD ["python", "-m", "truthclf_agents.orchestrator"]
