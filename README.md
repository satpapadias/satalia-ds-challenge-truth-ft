# Truthfulness Verification System: Multi-Agent MCP Pipeline

The Truthfulness Agent is a multi-agent, microservice-based system designed for binary truthfulness classification of political statements. 

Built using the Model Context Protocol (MCP) and Agent-to-Agent (A2A) communication, the system evaluates statements using both a zero-shot baseline and a custom Vertex AI Supervised Fine-Tuned (SFT) model. An Orchestrator manages the routing, aggregates probabilities using a log-odds pool, and triggers an Explainer agent to generate feature-attribution rationales.

## Key Features

*   **Custom Vertex AI SFT Model:** We fine-tuned `gemini-2.5-flash` on a speaker-disjoint dataset via Google Cloud. The tuned model demonstrably outperforms the baseline on held-out test data.
*   **Platt Scaling Calibration:** Both models emit raw log-probabilities that are calibrated into true confidence scores via a Platt scaling artifact fitted on a strict validation split.
*   **Multi-Agent A2A Architecture:** 
    *   `truthclf-orchestrator`: Fans out requests, handles peer discovery, and reconciles log-odds.
    *   `truthclf-zero-shot-predictor`: Queries the base Gemini model.
    *   `truthclf-fine-tuned-predictor`: Queries the custom Vertex AI SFT endpoint.
    *   `truthclf-explainer`: Explains predictions using leave-one-field-out occlusion.
*   **Model Context Protocol (MCP):** All agents delegate data retrieval and LLM inference to centralized `data-tools` and `model-tools` MCP servers running locally.
*   **Explainability via Occlusion:** The explainer does not just ask the LLM "why". It programmatically removes metadata fields (e.g., `speaker_name`) one by one, measures the mathematical shift in the predicted probability, and identifies the exact driver of the model's decision.

## Benchmark Performance (Held-Out Test Set: N=1,991)

The custom Vertex AI fine-tuning job yielded a measurable performance gain over the zero-shot baseline. *Both models were calibrated strictly on the validation set to prevent data leakage.*

| Model / Variant | Accuracy | Balanced Acc. | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot Baseline** (`gemini-2.5-flash`) | 70.52% | 69.78% | 0.7716 | 0.7810 | 0.1935 | 0.0280 |
| **Fine-Tuned Model** (Vertex SFT Endpoint) | **72.38%** | **72.21%** | **0.7961** | **0.8037** | **0.1900** | 0.0559 |

---

## Prerequisites & Setup

1. **Python Environment:** Ensure Python 3.10+ is installed.
2. **Google Cloud Auth:** You must be authenticated with GCP Application Default Credentials and have access to the project `x-wppai-researchlab-wpptestbed`.
   ```bash
   gcloud auth application-default login