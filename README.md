# 🏛️ Truthfulness Agent: Multi-Agent MCP Pipeline

The Truthfulness Agent is a multi-agent, microservice-based system designed for binary truthfulness classification of political statements. Built on the Agent Development Kit (ADK) 2.0, this project implements a highly modular architecture using the Model Context Protocol (MCP) and Agent-to-Agent (A2A) communication.

The system evaluates statements using both a baseline zero-shot model and a custom Vertex AI Supervised Fine-Tuned (SFT) model. An Orchestrator manages routing, aggregates predictions using a log-odds pool, and delegates to an Explainer agent for rigorous feature-attribution rationales.

## 🚀 Key Features

* **Multi-Agent A2A Orchestration:** The Orchestrator fans out verification tasks to specialist sub-agents (`truthclf-zero-shot-predictor`, `truthclf-fine-tuned-predictor`, `truthclf-explainer`) using `.well-known/agent-card.json` peer discovery.
* **Vertex AI Supervised Fine-Tuning (SFT):** A custom `gemini-2.5-flash` model natively fine-tuned on GCP infrastructure, proving measurable accuracy gains on unseen, speaker-disjoint data.
* **Platt Scaling Calibration:** Both the zero-shot and fine-tuned models are calibrated via Platt scaling fitted specifically on a held-out validation split, emitting true probabilistic confidence scores.
* **Model Context Protocol (MCP):** All agents delegate their dataset retrieval and LLM inference down to centralized `data-tools` and `model-tools` MCP servers.
* **Explainability via Occlusion:** The Explainer agent programmatically performs leave-one-field-out occlusion (removing metadata like `speaker_name`), measures the mathematical shift in predicted probability, and definitively identifies the driver of the model's decision.
* **GCP Infrastructure as Code:** The entire cloud environment is provisioned via Terraform, with microservices deployed to Cloud Run behind a secure IAM posture.

## 📊 Benchmark Performance (Held-Out Test Set: N=1,991)

The custom Vertex AI SFT model was benchmarked against the zero-shot baseline. *Both models were calibrated strictly on the validation set to prevent data leakage.*

| Model | Variant | Accuracy | Balanced Acc. | ROC-AUC | PR-AUC | Brier | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Baseline** | Zero-shot (`gemini-2.5-flash`) | 70.52% | 69.78% | 0.7716 | 0.7810 | 0.1935 | 0.0280 |
| **SFT Model (GCP)** | Fine-Tuned (`gemini-2.5-flash-sft`) | **72.38%** | **72.21%** | **0.7961** | **0.8037** | **0.1900** | **0.0559** |

## 🧠 Data Processing & Label Mapping

The original dataset contained 6-way human labels. These were mapped to a strict binary target to focus the model on the core distinction of truthfulness:
* **True:** `true`, `mostly-true`, `half-true`
* **False:** `barely-true`, `false`, `extremely-false`

## 🛠 Prerequisites & Setup

* **Python 3.12+** (configured via `pyproject.toml`)
* **Google Cloud CLI** (`gcloud`)
* **Terraform** (≥ 1.6)

### 1. GCP Authentication
Ensure you are authenticated with GCP Application Default Credentials, passing the specific project for billing/quota purposes:
```bash
gcloud auth application-default login --project="x-wppai-researchlab-wpptestbed"
```

### 2. Environment Setup (using `uv`)
We highly recommend using `uv`, a fast Python package installer and resolver.
```bash
# Sync the virtual environment with all optional groups
uv sync --all-groups --all-extras
```
*(Alternatively, via pip: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[viz,dev,mcp]"`)*

## ⚙️ Configuration (.env)

Create a `.env` file from the example (`cp .env.example .env`) and populate it. These variables configure the entire stack, for both local `docker-compose` and cloud deployments.

*(Note: The codebase uses `GOOGLE_CLOUD_PROJECT` for GCP project identification. Some individual scripts may internally use `GCP_PROJECT`, but configuration should always be supplied via `GOOGLE_CLOUD_PROJECT` in the `.env` file.)*

```env
# --- GCP Configuration ---
# Required for Vertex AI and other GCP services.
GOOGLE_CLOUD_PROJECT="x-wppai-researchlab-wpptestbed"
GOOGLE_CLOUD_LOCATION="us-central1"

# --- Model Endpoints ---
# Fine-Tuned model, either a Vertex Endpoint or a deployed model garden ID.
# To run against the pre-trained model, use this exact endpoint:
TRUTHCLF_FT_MODEL="projects/221040857484/locations/us-central1/endpoints/4351713640865333248"

# Zero-shot baseline model (used by model-tools server)
TRUTHCLF_ZEROSHOT_MODEL="gemini-2.5-flash"

# --- Service Authentication (for local docker-compose cluster) ---
# Bearer token for authenticating external requests to the orchestrator
ORCHESTRATOR_TOKEN="a-secure-random-string"
# Internal bearer token for agent-to-agent (A2A) communication
AGENT_TOKEN="another-secure-random-string"

# --- Pipeline Behavior ---
# Weight on the fine-tuned predictor when both answer (log-odds pool). 1.0 defers to it.
POOL_WEIGHT=1.0
# Maximum number of statements per /verify request.
# NOTE: Set this to a lower number (e.g., 2) for local testing to avoid 1 MiB SSE stream limits from the Explainer agent.
MAX_POINTS=2

# --- Agent Timeouts (seconds) ---
# Used by the orchestrator for A2A calls.
PREDICTOR_TIMEOUT_S=120
EXPLAINER_TIMEOUT_S=600
```

## 💻 Running Locally (End-to-End Pipeline)

To boot the entire local cluster and test the verification payload, simply execute the pipeline bash script. This script is a wrapper around `docker-compose.yml`, which orchestrates the entire stack of microservices locally.

```bash
export TRUTHCLF_FT_MODEL="projects/221040857484/locations/us-central1/endpoints/4351713640865333248"
./test_pipeline.sh
```

**Execution Flow:**
1. Boots the `data-tools` and `model-tools` MCP servers via `docker-compose`.
2. Boots the Orchestrator and 3 specialist A2A Agents.
3. Fires a JSON verification payload for 10 statements to the Orchestrator.
4. Predicts via live Vertex AI calls, reconciles log-odds, runs occlusion explanations, and shuts down cleanly.

## ☁️ Cloud Deployment & Security

The application stack deploys natively to GCP Cloud Run. The cloudbuild.yaml file defines a CI/CD pipeline that automatically builds and pushes the agent and tools images to Google Artifact Registry.

**Infrastructure Provisioning:**
```bash
cd terraform
terraform init
terraform apply -var="project_id=x-wppai-researchlab-wpptestbed"
```

**Service Deployment:**
Once the infrastructure (Artifact Registry, VPC, GCS) is ready, deploy the microservices using:
```bash
./deploy.sh
```

### Verifying the Cloud Deployment

To test the live, deployed pipeline on Cloud Run, you can hit the Orchestrator's `/verify` endpoint directly.

*Note: For the best results and to prevent timeout/memory constraints during inference, ensure your `payload.json` contains a smaller batch (e.g., 2–5 statements).*

```bash
# 1. Fetch an OIDC identity token for authentication
export GCP_TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=https://truthclf-orchestrator-221040857484.us-central1.run.app")

# 2. Fire the payload to the Orchestrator
curl -s -X POST https://truthclf-orchestrator-221040857484.us-central1.run.app/verify \
  -H "Authorization: Bearer $GCP_TOKEN" \
  -H "X-API-Key: <INSERT_API_KEY_HERE>" \
  -H "Content-Type: application/json" \
  -d @payload.json | python3 -m json.tool
```

**Defense-in-Depth Security:** 
Service-to-service communication relies on **per-hop OIDC authentication**. The Orchestrator mints a Google-signed ID token using its compute service account and attaches it as a Bearer token. Cloud Run’s IAM edge validates the token on the receiving agent/MCP server. There are **zero `allUsers` ingress bindings**—all traffic is strictly IAM-locked and routed internally via VPC Private Google Access. 

## 🔍 Observability & Distributed Tracing

Every request is identified by a `run_id` (scoped to the `/verify` call) and a `context_id` (derived from the payload hash). 

The `run_id` is attached as a structured field to all orchestrator log entries, while the `context_id` propagates automatically through A2A task dispatches to all four A2A agents (the orchestrator, both predictors, and the explainer). This makes the multi-agent fan-out highly debuggable. A single Cloud Logging query can reconstruct the complete lifecycle of a request:

```bash
# Replace 'ed7bd54e40b14361b197bf66315a99dc' with the context_id from your JSON output
gcloud logging read \
  'resource.type="cloud_run_revision" AND "ed7bd54e40b14361b197bf66315a99dc"' \
  --project="x-wppai-researchlab-wpptestbed" \
  --format="table(timestamp, resource.labels.service_name, textPayload, jsonPayload.message)" \
  --order=asc \
  --freshness=2h
```

<!-- ## 🔬 The MLOps Fine-Tuning Pipeline

If you wish to recreate the fine-tuning process, the MLOps pipeline scripts are located in `scripts/`.

1. **Triggering Vertex SFT:** 
   `scripts/vertex_finetune.py` extracts a 20% speaker-disjoint split, strictly formats it into Vertex AI's `{"contents": [...]}` JSONL schema, uploads to GCS, and submits the `sft.train` job to Google.
2. **Generating the Decision Artifact:** 
   `scripts/fit_new_calibrator.py` queries the deployed Vertex endpoint to extract log-probabilities across the validation split, calculating and exporting the Platt Scaling artifact (`results/calibrators/`). -->

## 🔬 Reproducing the Evaluation & SFT Pipeline

Reviewers can fully reproduce the published 72.38% accuracy benchmark and regenerate the Platt scaling decision artifacts directly from the Vertex AI endpoints.

**1. Set your GCP environment variables:**
Ensure your terminal is authenticated and pointing to the live endpoints:
```bash
export GOOGLE_CLOUD_PROJECT="x-wppai-researchlab-wpptestbed"
export GOOGLE_CLOUD_LOCATION="us-central1"
export TRUTHCLF_FT_MODEL="projects/221040857484/locations/us-central1/endpoints/4351713640865333248"
```

**2. Evaluate and calibrate the Zero-Shot Baseline:**
```bash
python3 scripts/fit_new_calibrator.py --model "gemini-2.5-flash" --elicitation "logprob"
```

**3. Evaluate and calibrate the Fine-Tuned Model:**
```bash
python3 scripts/fit_new_calibrator.py --model "$TRUTHCLF_FT_MODEL"
```

*(Note: To recreate the fine-tuning process from scratch, scripts/vertex_finetune.py extracts a 20% speaker-disjoint split, formats it into Vertex AI's JSONL schema, uploads to GCS, and submits the sft.train job.)*

## 📂 Project Structure

```text
satalia-ds-challenge-truth-ft/
├── pyproject.toml / uv.lock    # Modern dependency management
├── .env.example                # Environment variable template
├── Dockerfile                  # Multi-stage container definitions
├── docker-compose.yml          # Local microservice orchestration
├── cloudbuild.yaml             # GCP Cloud Build CI/CD pipeline
├── test_pipeline.sh            # Master boot & test execution script
├── deploy.sh                   # Cloud Run deployment script
├── terraform/                  # GCP Infrastructure as Code
├── truthclf/                   # Core application logic & domain objects
├── truthclf_agents/            # A2A Agent implementations (pure MCP clients)
│   ├── orchestrator.py
│   ├── zero_shot.py
│   ├── fine_tuned.py
│   └── explainer.py
├── truthclf_mcp/               # Model Context Protocol servers
│   ├── data_tools.py
│   └── model_tools.py
├── scripts/                    # Standalone MLOps scripts
│   ├── vertex_finetune.py      # Data prep and SFT trigger
│   └── fit_new_calibrator.py   # Platt scaling generation
├── tests/                      # Pytest integration & unit tests
└── results/
    └── calibrators/            # JSON Platt-scaling configurations
```