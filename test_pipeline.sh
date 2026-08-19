#!/bin/bash

export GOOGLE_CLOUD_PROJECT="x-wppai-researchlab-wpptestbed"
export GOOGLE_CLOUD_LOCATION="us-central1"
export ORCHESTRATOR_TOKEN="super_secret_test_token"

# Override the legacy defaults so the tools use our new Vertex client
export TRUTHCLF_BASE_MODEL="gemini-2.5-flash"
export TRUTHCLF_FT_MODEL="gemini-2.5-flash" # Temporary fallback until we locate the true FT endpoint

export DATA_TOOLS_URL="http://127.0.0.1:8081/mcp"
export MODEL_TOOLS_URL="http://127.0.0.1:8082/mcp"
export ZERO_SHOT_AGENT_URL="http://127.0.0.1:9101"
export FINE_TUNED_AGENT_URL="http://127.0.0.1:9102"
export EXPLAINER_AGENT_URL="http://127.0.0.1:9103"

echo "Starting MCP Servers..."
PORT=8081 .venv/bin/python -m truthclf_mcp.data_tools --port 8081 &
PID_DATA=$!
PORT=8082 .venv/bin/python -m truthclf_mcp.model_tools --port 8082 &
PID_MODEL=$!

echo "Starting Agent Services..."
PORT=9101 .venv/bin/python -m truthclf_agents.zero_shot &
PID_ZERO=$!
PORT=9102 .venv/bin/python -m truthclf_agents.fine_tuned &
PID_FINE=$!
PORT=9103 .venv/bin/python -m truthclf_agents.explainer &
PID_EXPLAIN=$!
PORT=9100 .venv/bin/python -m truthclf_agents.orchestrator &
PID_ORCH=$!

echo "Waiting 10 seconds for all services to boot..."
sleep 10

echo "Generating test payload..."
.venv/bin/python generate_payload.py > payload.json

echo "Firing end-to-end request to Orchestrator..."
curl -s -X POST http://127.0.0.1:9100/verify \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $ORCHESTRATOR_TOKEN" \
     -d @payload.json | jq .

echo -e "\n\nShutting down all services..."
kill $PID_DATA $PID_MODEL $PID_ZERO $PID_FINE $PID_EXPLAIN $PID_ORCH
wait
echo "Done."
