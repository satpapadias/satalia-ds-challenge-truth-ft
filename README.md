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
Ensure you are authenticated with GCP Application Default Credentials:
```bash
gcloud auth application-default login
```

### 2. Environment Setup (using `uv`)
We highly recommend using `uv`, a fast Python package installer and resolver.
```bash
# Sync the virtual environment with all optional groups
uv sync --all-groups --all-extras
```
*(Alternatively, via pip: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[viz,dev,mcp]"`)*

## ⚙️ Configuration (.env)

Create a `.env` file in the root directory. To run against the already-deployed custom model, use the exact endpoint below:

```env
GOOGLE_CLOUD_PROJECT="x-wppai-researchlab-wpptestbed"
GOOGLE_CLOUD_LOCATION="us-central1"
TRUTHCLF_FT_MODEL="projects/221040857484/locations/us-central1/endpoints/4351713640865333248"
```

## 💻 Running Locally (End-to-End Pipeline)

To boot the entire local cluster and test the verification payload, simply execute the pipeline bash script. 

```bash
export TRUTHCLF_FT_MODEL="projects/221040857484/locations/us-central1/endpoints/4351713640865333248"
./test_pipeline.sh
```

**Execution Flow:**
1. Boots the `data-tools` and `model-tools` MCP servers.
2. Boots the Orchestrator and 3 specialist A2A Agents.
3. Fires a JSON verification payload for 10 statements to the Orchestrator.
4. Predicts via live Vertex AI calls, reconciles log-odds, runs occlusion explanations, and shuts down cleanly.

## ☁️ Cloud Deployment & Security

The application stack deploys natively to GCP Cloud Run. 

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

**Defense-in-Depth Security:** 
Service-to-service communication relies on **per-hop OIDC authentication**. The Orchestrator mints a Google-signed ID token using its compute service account and attaches it as a Bearer token. Cloud Run’s IAM edge validates the token on the receiving agent/MCP server. There are **zero `allUsers` ingress bindings**—all traffic is strictly IAM-locked and routed internally via VPC Private Google Access.

## 🔬 The MLOps Fine-Tuning Pipeline

If you wish to recreate the fine-tuning process, the MLOps pipeline scripts are located in `scripts/`.

1. **Triggering Vertex SFT:** 
   `scripts/vertex_finetune.py` extracts a 20% speaker-disjoint split, strictly formats it into Vertex AI's `{"contents": [...]}` JSONL schema, uploads to GCS, and submits the `sft.train` job to Google.
2. **Generating the Decision Artifact:** 
   `scripts/fit_new_calibrator.py` queries the deployed Vertex endpoint to extract log-probabilities across the validation split, calculating and exporting the Platt Scaling artifact (`results/calibrators/`).

## 📂 Project Structure

```text
truthclf-agent/
├── pyproject.toml / uv.lock    # Modern dependency management
├── test_pipeline.sh            # Master boot & test execution script
├── deploy.sh                   # Cloud Run deployment script
├── terraform/                  # GCP Infrastructure as Code
├── truthclf/                   # Core application logic & domain objects
├── truthclf_agents/            # A2A Agent implementations
│   ├── orchestrator/
│   ├── zero_shot/
│   ├── fine_tuned/
│   └── explainer/
├── truthclf_mcp/               # Model Context Protocol servers
│   ├── data_tools/
│   └── model_tools/
├── scripts/                    # Standalone MLOps scripts
│   ├── vertex_finetune.py      # Data prep and SFT trigger
│   └── fit_new_calibrator.py   # Platt scaling generation
└── results/
    └── calibrators/            # JSON Platt-scaling configurations
```