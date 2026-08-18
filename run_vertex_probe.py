"""A small script to make one live call to the base model using the new
VertexClient and print p(True).

This script requires the GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
environment variables to be set to correctly initialize the Vertex AI client.
"""
import os
import sys
from truthclf.llm import make_client

# The new VertexClient's __init__ requires GOOGLE_CLOUD_PROJECT and
# GOOGLE_CLOUD_LOCATION to be set. This is a check to ensure the user
# has set them.
if not os.environ.get("GOOGLE_CLOUD_PROJECT") or not os.environ.get("GOOGLE_CLOUD_LOCATION"):
    print("ERROR: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables must be set.", file=sys.stderr)
    sys.exit(1)

# A sample prompt for testing the classification.
# The `classify_one` method expects a list of message dictionaries.
PROMPT_TEXT = "Statement: The sky is blue. Is this statement true or false?"
messages = [{"role": "user", "content": PROMPT_TEXT}]

def run_probe():
    """Instantiates the VertexClient and runs a single classification."""
    print("Initializing VertexClient for gemini-2.5-flash...")
    # Use make_client to get the new VertexClient instance
    client = make_client("gemini-2.5-flash", backend="vertex")

    print(f"Submitting prompt for classification:\n'{PROMPT_TEXT}'")
    
    # The classify method takes a list of message lists.
    results = client.classify([messages])
    
    if results and "p_true" in results[0]:
        p_true = results[0]["p_true"]
        print(f"\nSuccess! The model returned p(True) = {p_true:.4f}")
    else:
        print("\nFailed to get a valid response from the model.")
        print("Received:", results)

if __name__ == "__main__":
    run_probe()
