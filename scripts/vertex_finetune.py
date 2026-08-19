"""Trigger a Vertex AI Supervised Fine-Tuning (SFT) job for gemini-2.5-flash.

This script performs the following steps:
1. Loads the data from data.csv.
2. Prepares the training data using the speaker-disjoint split.
3. Formats the data into the JSONL format required by Vertex AI.
4. Uploads the formatted data to Google Cloud Storage.
5. Kicks off a supervised fine-tuning job using the Vertex AI SDK.
"""

import os
import pathlib
import time

from google.cloud import storage
import vertexai
from vertexai.tuning import sft

from truthclf import data

# --- Configuration ---
# GCP and Vertex AI
PROJECT_ID = os.environ.get("GCP_PROJECT")
LOCATION = "us-central1"
GCS_BUCKET_NAME = "x-wppai-researchlab-wpptestbed-ft-data"

# Data and model
SOURCE_DATA_FILE = "data.csv"
TRAIN_JSONL_FILENAME = "train.jsonl"
GCS_TRAIN_DATA_PATH = f"gs://{GCS_BUCKET_NAME}/{TRAIN_JSONL_FILENAME}"
BASE_MODEL = "gemini-2.5-flash"
TUNED_MODEL_DISPLAY_NAME = "truthclf-gemini-2.5-flash-sft"

# --- Main script ---
def main():
    """Main function to run the fine-tuning pipeline."""
    print("--- Starting Vertex AI SFT Pipeline for Gemini-2.5-Flash ---")

    # 1. Data Prep
    print(f"1. Loading data from '{SOURCE_DATA_FILE}'...")
    rows = data.load(SOURCE_DATA_FILE)
    clean_df, _ = data.clean_dataset(rows, scheme="primary")
    train_rows, _, _ = data.speaker_disjoint_3way(clean_df, 0.2, 0.2, 0, scheme="primary")
    print(f"   -> Prepared {len(train_rows)} rows for training.")

    # 2. JSONL Formatting
    print(f"2. Formatting data into JSONL file: '{TRAIN_JSONL_FILENAME}'...")
    local_train_path = pathlib.Path(TRAIN_JSONL_FILENAME)
    data.write_sft_jsonl(train_rows, local_train_path)
    print(f"   -> Wrote training data to '{local_train_path}'.")

    # 3. GCS Upload
    print(f"3. Uploading '{local_train_path}' to '{GCS_TRAIN_DATA_PATH}'...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(TRAIN_JSONL_FILENAME)
    blob.upload_from_filename(local_train_path)
    print("   -> Upload complete.")

    # 4. Trigger Tuning
    print(f"4. Initializing Vertex AI and triggering fine-tuning job for '{BASE_MODEL}'...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    sft_job = sft.train(
        source_model=BASE_MODEL,
        train_dataset=GCS_TRAIN_DATA_PATH,
        tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
    )

    job_resource_name = sft_job.resource_name
    print(f"   -> Fine-tuning job started successfully.")
    print(f"   -> Job Resource Name: {job_resource_name}")
    print("   -> Polling for job status (updates every 60 seconds)...")

    # Polling loop
    while not sft_job.has_ended:
        time.sleep(60)
        try:
            sft_job.refresh()
            state_str = str(sft_job.state)
            print(f"   -> Job state: {state_str} (Updated at {time.ctime()})")
        except Exception as e:
            print(f"   -> Error refreshing job state: {e}")
            break

    sft_job.refresh()
    state = sft_job.state
    
    # Check if state is 4 (integer, enum with value 4, or string "4" / "JOB_STATE_SUCCEEDED")
    is_success = False
    if state == 4 or getattr(state, "value", None) == 4:
        is_success = True
    elif "SUCCEEDED" in str(state).upper() or str(state) == "4":
        is_success = True

    print(f"--- Job finished with state: {state} ---")
    if is_success:
        print("🎉 Fine-tuning completed successfully!")
        print(f"   -> Tuned model available: {sft_job.tuned_model_name}")
        print(f"   -> Tuned model endpoint name: {sft_job.tuned_model_endpoint_name}")
    else:
        print(f"   -> Job did not succeed. Please check the logs in the Vertex AI console.")
        print(f"   -> Final job object: {sft_job}")


if __name__ == "__main__":
    if not PROJECT_ID:
        raise ValueError("GCP_PROJECT environment variable not set. Please set it to your GCP project ID.")
    main()
