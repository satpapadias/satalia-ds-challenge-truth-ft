# Truthfulness Agent: Multi-Agent MCP Pipeline

This repository contains a multi-agent system built using the Agent Development Kit (ADK) 2.0 for evaluating the truthfulness of political statements. The system leverages the Model Context Protocol (MCP) for communication between agents, Agent-to-Agent (A2A) routing, and a custom Vertex AI Supervised Fine-Tuned (SFT) model to provide accurate and reliable truthfulness classifications.

## 🚀 Features

*   **Multi-Agent Orchestration:** A sophisticated multi-agent system that coordinates the flow of information and tasks between different agents to achieve a common goal.
*   **Vertex AI Supervised Fine-Tuning (SFT):** A custom-trained model on Vertex AI using supervised fine-tuning techniques, enhanced with Platt scaling calibration for improved accuracy.
*   **Model Context Protocol (MCP) Tool Servers:** A set of tool servers that expose the models and other tools to the agents using the MCP protocol, enabling seamless integration and communication.
*   **Explainability:** Leave-one-field-out occlusion is used to provide insights into the model's decision-making process, enhancing transparency and trust.
*   **GCP/Cloud Run Deployment:** The entire system is deployed on Google Cloud Platform (GCP) using Terraform, with the application running on Cloud Run for scalability and ease of management.

## 📊 Benchmark Performance

The fine-tuned model was benchmarked against a zero-shot baseline to evaluate its performance on a held-out test set. The results demonstrate a significant improvement in accuracy and ROC-AUC, validating the effectiveness of the fine-tuning process.

| Model                       | Accuracy | ROC-AUC |
| --------------------------- | -------- | ------- |
| Zero-Shot Baseline (gemini-2.5-flash) | 65.22%   | 0.7231  |
| Fine-Tuned Model (gemini-2.5-flash-sft) | **72.38%** | **0.7961** |

The Fine-Tuned model achieved **72.38% Accuracy** and **0.7961 ROC-AUC**, proving a measurable fine-tuning gain on the held-out test set without data leakage.

## 🧠 Data Processing & Label Mapping

The original 6-way human labels from the dataset were mapped to a binary target to simplify the classification task. The mapping is as follows:

*   **True:** `true`, `mostly-true`, `half-true`
*   **False:** `barely-true`, `false`, `extremely-false`

This binary classification approach allows the model to focus on the core task of distinguishing between truthful and untruthful statements.

## 🛠 Prerequisites & Setup

To run the pipeline locally and deploy it to GCP, you will need the following tools:

*   **Python 3.10+:** The application is written in Python 3.10.
*   **Virtual Environments:** It is recommended to use a virtual environment to manage dependencies.
*   **Google Cloud CLI:** The `gcloud` CLI is required for authentication and interacting with GCP services.
*   **Terraform:** Terraform is used for infrastructure as code to provision the necessary GCP resources.

You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

To authenticate with GCP, run the following command:

```bash
gcloud auth application-default login
```

## ⚙️ Configuration (.env)

The following environment variables are required to run the pipeline. Create a `.env` file in the root directory and add the following variables:

```bash
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
TRUTHCLF_FT_MODEL=<your-finetuned-model-endpoint>
```

## 💻 Running Locally & Testing

To run the pipeline locally, you need to export the `TRUTHCLF_FT_MODEL` environment variable and then execute the `test_pipeline.sh` script.

```bash
export TRUTHCLF_FT_MODEL=<your-finetuned-model-endpoint>
./test_pipeline.sh
```

The `test_pipeline.sh` script boots the MCP servers, A2A agents, and runs a verification payload to test the entire pipeline.

## ☁️ Deploying to GCP & Security

The application is deployed to GCP using Terraform. The Terraform scripts in the `/terraform` directory provision the following resources:

*   **Cloud Run:** The application is deployed as a serverless container on Cloud Run.
*   **Google Cloud Storage (GCS):** GCS buckets are used to store artifacts and data.
*   **Artifact Registry:** The Docker image is stored in Artifact Registry.

### Security

Service-to-service communication between the MCP tools and sub-agents is secured using IAM restrictions and network-only authentication. This ensures that only authorized services can communicate with each other, preventing unauthorized access to the system.

## 🔬 The Fine-Tuning Pipeline

The fine-tuning pipeline consists of two main scripts:

*   `scripts/vertex_finetune.py`: This script formats the data into JSONL, uploads it to GCS, and triggers a Vertex AI training job to fine-tune the model.
*   `scripts/fit_new_calibrator.py`: This script fits the Platt scale on the validation set to calibrate the model's predictions, improving the accuracy of the final classification.

## 📂 Project Structure

```
.
├───.coverage
├───.dockerignore
├───.env.example
├───.gitignore
├───.llm_cache.json
├───arxiv23-perline.pdf
├───build_deck.py
├───CLAUDE.md
├───cloudbuild.yaml
├───Data Science Challenge - Truth - Full.pdf
├───data.csv
├───deploy.sh
├───docker-compose.yml
├───Dockerfile
├───explain_results.json
├───final_pipeline_run.log
├───Finetuning_Guide.ipynb
├───ft_artifacts.json
├───ft_eval_cache.json
├───ft_eval_identity.json
├───ft_eval_results.json
├───generate_payload.py
├───LICENSE.txt
├───payload.json
├───predict.py
├───pyproject.toml
├───qwen_serve_artifacts.json
├───README.md
├───run_tests.py
├───run_vertex_probe.py
├───test_pipeline.sh
├───train.jsonl
├───uv.lock
├───docs/
├───ft_data/
├───NOTES/
├───results/            # Output directory for evaluation results and artifacts
├───scripts/            # Standalone scripts for MLOps, evaluation, and data processing
├───terraform/          # Terraform infrastructure-as-code for GCP deployment
├───tests/              # Unit and integration tests
├───truthclf/           # Core application source code
├───truthclf_agents/    # Agent implementations
├───truthclf_mcp/       # MCP tool and model servers
└───truthclf.egg-info/
```
