#!/bin/bash
set -euo pipefail

# Deploys the entire agent network to Google Cloud Run.
#
# This script orchestrates the three main phases of deployment:
#   1. Bootstrapping the GCP environment (if not already done).
#   2. Building the container images with Cloud Build.
#   3. Applying the Terraform configuration to create the services.
#
# It requires `gcloud`, `gsutil`, and `terraform` to be installed and on the PATH.

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
    echo "ERROR: GOOGLE_CLOUD_PROJECT is not set." >&2
    echo "Set it to your target GCP project ID and re-run." >&2
    exit 1
fi

if [[ -z "${ORCHESTRATOR_TOKEN:-}" ]]; then
    echo "ERROR: ORCHESTRATOR_TOKEN is not set." >&2
    echo "This is the public API key for the /verify endpoint." >&2
    echo "Generate a secure random string and export it, e.g.:" >&2
    echo "  export ORCHESTRATOR_TOKEN=\$(openssl rand -hex 32)" >&2
    exit 1
fi

PROJECT_ID=$GOOGLE_CLOUD_PROJECT
REGION=${GCP_REGION:-us-central1}
TF_STATE_BUCKET="${PROJECT_ID}-truthclf-tfstate"

echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo

# --- 1. Bootstrap Environment ---
echo "--- Running bootstrap ---"
cd terraform
./bootstrap.sh
cd ..
echo "Bootstrap complete."
echo

# --- 2. Create Artifact Registry with Terraform ---
# This targeted pass creates ONLY the Artifact Registry repository, so that
# the build step has a place to push the images.
echo "--- Creating Artifact Registry with Terraform ---"
cd terraform

echo "Removing old .terraform.lock.hcl to ensure a clean provider installation..."
rm -f .terraform.lock.hcl

echo "Initializing Terraform..."
terraform init -upgrade

echo "Applying Terraform plan to create repository..."
terraform apply -auto-approve \
    -target=google_artifact_registry_repository.repo \
    -var="project_id=${PROJECT_ID}" \
    -var="region=${REGION}" \
    -var="orchestrator_token=${ORCHESTRATOR_TOKEN}"

cd ..
echo "Artifact Registry created."
echo

# --- 3. Build Container Images ---
echo "--- Building container images with Cloud Build ---"
# The _REPOSITORY substitution corresponds to the `repository` variable in main.tf
gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION="${REGION}",_REPOSITORY="truthclf-images"
echo "Image builds complete."
echo

# --- 4. Deploy All Services with Terraform ---
# This pass creates all other resources. Since the images now exist, the
# Cloud Run services will be able to pull them and start successfully.
echo "--- Deploying all services with Terraform ---"
cd terraform
 
echo "Applying full Terraform plan..."
terraform apply -auto-approve \
    -var="project_id=${PROJECT_ID}" \
    -var="region=${REGION}" \
    -var="orchestrator_token=${ORCHESTRATOR_TOKEN}"

echo
echo "--- Deployment complete ---"
terraform output orchestrator_url

cd ..