"""Fit and save a new calibrator for a zero-shot model.

This script runs a new model on the validation and test splits, fits a
calibrator on the validation probabilities, and saves the resulting
DecisionArtifact to the calibrators directory.

This is intended for generating calibrators for new models not yet in the cache.
It will make live provider calls and will be billed.

Usage:
    python scripts/fit_new_calibrator.py --model "gemini-2.5-flash" --elicitation "logprob"
"""

import argparse
import os
import sys

# Add project root to path to allow running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import data, evaluation, experiments


def main():
    parser = argparse.ArgumentParser(description="Fit and save a new calibrator for a zero-shot model.")
    parser.add_argument("--model", type=str, required=True, help="The model identifier (e.g., 'gemini-2.5-flash').")
    parser.add_argument("--elicitation", choices=["logprob", "score"], default="logprob", help="The probability elicitation mode.")
    parser.add_argument("--variant", choices=["full", "statement_only"], default="full", help="The prompt variant to use.")
    parser.add_argument("--scheme", default="primary", help="The binarization scheme.")
    parser.add_argument("--output-dir", default="results/calibrators", help="Directory to save the calibrator artifact.")
    args = parser.parse_args()

    print(f"Starting calibration for model='{args.model}' with elicitation='{args.elicitation}'...")

    # 1. Load data and create the standard speaker-disjoint splits
    print("Loading and splitting data...")
    clean, _ = data.clean_dataset(data.load("data.csv"), args.scheme)
    _, val_rows, test_rows = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, args.scheme)
    val_labels = [r.y(args.scheme) for r in val_rows]
    test_labels = [r.y(args.scheme) for r in test_rows]
    print(f"Using {len(val_rows)} validation rows and {len(test_rows)} test rows.")

    # 2. Run the new model on both splits to get raw probabilities.
    # This will make live calls to the provider (Vertex AI) and cache the results.
    # Ensure your provider credentials (e.g., GOOGLE_CLOUD_PROJECT) are set.
    print("Getting probabilities from validation split (this may take a while)...")
    val_probs, _ = experiments.run_on_rows(
        args.model, args.variant, val_rows, scheme=args.scheme,
        backend="vertex", use_logprobs=(args.elicitation == "logprob")
    )

    print("Getting probabilities from test split...")
    test_probs, _ = experiments.run_on_rows(
        args.model, args.variant, test_rows, scheme=args.scheme,
        backend="vertex", use_logprobs=(args.elicitation == "logprob")
    )
    print("Successfully retrieved all probabilities.")

    # 3. Run the calibrated evaluation pipeline.
    # This fits the calibrator on validation probabilities and tunes the threshold.
    print("Fitting calibrator and tuning threshold on validation data...")
    eval_result = evaluation.calibrated_evaluation(
        val_probs, val_labels, test_probs, test_labels, objective="balanced_accuracy"
    )
    print(f"Calibration complete. Method: {eval_result.calibrator['method']}, Threshold: {eval_result.threshold:.4f}")

    # 4. Build the shippable artifact containing the calibrator and threshold.
    print("Building decision artifact...")
    artifact = evaluation.build_artifact(
        eval_result,
        model=args.model,
        elicitation=args.elicitation,
        fitted_on=f"speaker-disjoint validation split (seed 0, scheme {args.scheme})",
        n_val=len(val_labels),
        val_probs=val_probs,
        val_labels=val_labels,
        objective="balanced_accuracy"
    )

    # 5. Save the artifact to the specified directory.
    safe_model_name = args.model.replace("/", "_")
    artifact_filename = f"{safe_model_name}_{args.elicitation}.json"
    artifact_path = os.path.join(args.output_dir, artifact_filename)

    os.makedirs(args.output_dir, exist_ok=True)
    artifact.save(artifact_path)

    print(f"\nSuccessfully saved new calibrator artifact to:\n{artifact_path}")
    print("\nTest metrics with new calibrator:")
    for k, v in eval_result.metrics.items():
        print(f"  {k:<20} {v:.4f}")


if __name__ == "__main__":
    main()
