#!/bin/bash
set -euo pipefail

# One-time bootstrap for the GCP environment.
#
# This script enables the required APIs and creates the GCS bucket for Terraform's
# remote state. It should be run once before the first `terraform init`.
#
# The GCS backend cannot create its own bucket; that is a genuine chicken-and-egg
# problem that this script resolves.

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
    echo "ERROR: GOOGLE_CLOUD_PROJECT is not set." >&2
    echo "Set it to your target GCP project ID and re-run." >&2
    exit 1
fi

PROJECT_ID=$GOOGLE_CLOUD_PROJECT
REGION=${GCP_REGION:-us-central1}
BUCKET_NAME="${PROJECT_ID}-truthclf-tfstate"

echo "Project:         ${PROJECT_ID}"
echo "Region:          ${REGION}"
echo "TF state bucket: gs://${BUCKET_NAME}"
echo

if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud command not found. Please install the Google Cloud SDK." >&2
    exit 1
fi

echo "Enabling required GCP APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com iam.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com storage.googleapis.com --project="${PROJECT_ID}"

echo
echo "Creating GCS bucket for Terraform state..."
if gsutil ls "gs://${BUCKET_NAME}" &>/dev/null; then
    echo "Bucket gs://${BUCKET_NAME} already exists. Skipping creation."
else
    gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${BUCKET_NAME}"
    gsutil versioning set on "gs://${BUCKET_NAME}"
    echo "Bucket gs://${BUCKET_NAME} created and versioning enabled."
fi

echo
echo "Updating terraform/main.tf with bucket name: ${BUCKET_NAME}"
sed -i.bak "s/bucket = \".*\"/bucket = \"${BUCKET_NAME}\"/" main.tf && rm main.tf.bak

echo
echo "Bootstrap complete."
echo "You can now run 'terraform init' from the 'terraform/' directory."