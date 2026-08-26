# Two images from one file.
#
#   target `tools`  -> the MCP servers. Imports truthclf, so it carries the full
#                      data-science stack, the dataset and the fitted artifacts.
#   target `agent`  -> the four A2A agents. Speaks A2A and MCP and nothing else.
#
# The agents are pure MCP clients: they reach every capability by calling a tool,
# and none of them imports truthclf. The split makes that structural rather than
# conventional -- the agent image is built WITHOUT the truthclf package and
# without the libraries it needs, and the build fails if either turns out to be
# importable. A future edit that reaches into truthclf from an agent cannot be
# merged as a passing build.

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

# uv resolves from the committed lockfile, so a build installs the versions the
# results were produced with rather than re-resolving.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./

# ---------------------------------------------------------------------------
# The MCP servers. These are the only processes that import truthclf.
# ---------------------------------------------------------------------------
FROM base AS tools

RUN uv sync --frozen --no-install-project --extra mcp
COPY truthclf/ ./truthclf/
COPY truthclf_mcp/ ./truthclf_mcp/
RUN uv sync --frozen --extra mcp

# Data and artifacts, copied rather than mounted so a container behaves the same
# wherever it runs.
#   data.csv               -> data-tools
#   ft_eval_cache.json     -> the recorded fine-tuned probabilities
#   ft_eval_identity.json  -> which statement each of those belongs to; without
#                             it the stored path refuses to serve at all
#   results/calibrators/   -> the fitted calibrator artifacts
#
# THIS IMAGE CONTAINS THE CHALLENGE DATASET. It must only ever be pushed to a
# private registry inside the same trust boundary as the data itself.
COPY data.csv ./
COPY results/calibrators/ ./results/calibrators/

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/.llm_cache \
    && chown -R app:app /app
USER app
CMD ["python", "-m", "truthclf_mcp.data_tools", "--host", "0.0.0.0", "--port", "8081"]

# ---------------------------------------------------------------------------
# The agents. No truthclf, no dataset, no data-science stack.
# ---------------------------------------------------------------------------
FROM base AS agent

# --only-group installs the agent dependencies alone; --no-install-project keeps
# the truthclf distribution out entirely. Only truthclf_agents/ is copied, so
# there is nothing to import even by accident.
RUN uv sync --frozen --no-install-project --only-group agents
COPY truthclf_agents/ ./truthclf_agents/

# The assertion the split exists for. `import truthclf` must fail, the
# data-science stack must be absent, and all four agents must still import.
# Checked at build time so the property cannot silently regress: an agent that
# starts reaching into truthclf directly stops producing a buildable image.
RUN set -eu; \
    for mod in truthclf truthclf_mcp sklearn scipy pandas statsmodels numpy together tiktoken diskcache; do \
        if python -c "import $mod" >/dev/null 2>&1; then \
            echo "BUILD FAILED: the agent image can import '$mod'." >&2; \
            echo "Agents are pure MCP clients and must reach capabilities only" >&2; \
            echo "through tool calls. Move the logic behind an MCP tool." >&2; \
            exit 1; \
        fi; \
    done; \
    python -c "import truthclf_agents.orchestrator, truthclf_agents.zero_shot, \
               truthclf_agents.fine_tuned, truthclf_agents.explainer"; \
    echo "verified: no truthclf, no data-science stack, all four agents import"

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app
CMD ["python", "-m", "truthclf_agents.orchestrator", "--host", "0.0.0.0", "--port", "9100"]
