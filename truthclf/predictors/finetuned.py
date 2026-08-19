"""Fine-tuned predictor: supervised fine-tuning, then prediction
through the resulting model.

`predict(points, labels=None)` has the same signature and behaviour as
ZeroShotPredictor.predict (it delegates to a ZeroShotPredictor pointed at the
served fine-tuned model), so the two are interchangeable.

Serving note: the fine-tuned model is served via a dedicated
endpoint for evaluation (see evaluate_finetuned.py and the README). Pass a `client`
configured for the served model, or set `served_model` to the endpoint/model id.
"""

from __future__ import annotations

import os
import time

from .. import data, llm
from .base import Predictor
from .zeroshot import ZeroShotPredictor


class FinetunedPredictor(Predictor):
    def __init__(self, base_model, served_model=None, variant="full", scheme="primary",
                 client=None, threshold=0.5, use_logprobs=True, calibrator=None):
        self.base_model = base_model
        self.served_model = served_model      # endpoint/model id used at predict time
        self.variant = variant
        self.scheme = scheme
        self.client = client
        self.threshold = threshold
        self.use_logprobs = use_logprobs
        self.calibrator = calibrator       # DecisionArtifact or path; see ZeroShotPredictor
        self.job_id = None
        self.output_name = None

    def fine_tune(self, training_dataset, val_rows=None, n_epochs=3, learning_rate=1e-5,
                  suffix="truthclf_sft", lora=True, train_on_inputs="auto",
                  poll=True, workdir="ft_data", poll_interval=30):
        """Prepare the data, split off a validation set, and launch a LoRA SFT job.
        Pass a single labelled training_dataset (same format as data.csv);
        a speaker-disjoint train/validation split is made internally (an explicit
        val_rows overrides it). Polls to completion by default and stores output_name."""
        import subprocess
        import re

        print(f"Delegating to scripts/vertex_finetune.py to launch the SFT job...")
        
        # This assumes the script is run from the project root.
        script_path = "scripts/vertex_finetune.py"
        
        try:
            # We use subprocess.run to execute the script and capture its output.
            # The output is streamed to stdout/stderr in real time, and also captured.
            result = subprocess.run(
                ["python", "-u", script_path], 
                capture_output=True, 
                text=True, 
                check=True,  # Raises CalledProcessError on non-zero exit codes
                encoding='utf-8'
            )
            
            # Search for the job resource name in the script's output
            output = result.stdout
            print(output) # Also print the full output for visibility

            match = re.search(r"Job Resource Name: (projects/.*/locations/.*/tuningJobs/.*)", output)
            if match:
                self.job_id = match.group(1).strip()
                print(f"Captured Job ID: {self.job_id}")

                # After a successful run, the model name might also be available
                match_model = re.search(r"Tuned model available: (projects/.*/locations/.*/models/.*)", output)
                if match_model:
                    self.output_name = match_model.group(1).strip()
                    print(f"Captured Tuned Model Name: {self.output_name}")
            else:
                 print("Could not find Job Resource Name in the script output.")


        except subprocess.CalledProcessError as e:
            print(f"Error running {script_path}:")
            print(e.stdout)
            print(e.stderr)
            raise RuntimeError(f"Fine-tuning script failed with exit code {e.returncode}") from e
        except FileNotFoundError:
            print(f"Error: The script at '{script_path}' was not found.")
            raise


    def _delegate(self):
        served = self.served_model or self.output_name
        if served is None:
            raise RuntimeError("no served model: call fine_tune() first or pass served_model=")
        client = self.client or llm.make_client(served)
        return ZeroShotPredictor(model=served, variant=self.variant, client=client,
                                 threshold=self.threshold, use_logprobs=self.use_logprobs,
                                 calibrator=self.calibrator)

    def predict(self, points, labels=None):
        return self._delegate().predict(points, labels=labels)

    def rationale(self, rows, max_tokens=64):
        return self._delegate().rationale(rows, max_tokens=max_tokens)


FT_PROB_CACHE = "ft_eval_cache.json"


def load_cached_probs(rows, path: str = FT_PROB_CACHE) -> list:
    """Per-row fine-tuned P(True), keyed by row_id, from the endpoint session.

    The fine-tuned model is served on a short-lived dedicated endpoint, so its
    predictions are persisted row-by-row rather than re-derived. This accessor is
    the single reader; three call sites previously inlined the json.load and the
    str(row_id) indexing, and a missing row silently became a KeyError deep in a
    metric call.
    """
    import json as _json
    import os as _os
    if not _os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — fine-tuned probabilities come from "
            "scripts/evaluate_finetuned.py, which provisions a dedicated endpoint.")
    with open(path, encoding="utf-8") as f:
        cache = _json.load(f)
    missing = [r.row_id for r in rows if str(r.row_id) not in cache]
    if missing:
        raise KeyError(f"{len(missing)} rows absent from {path} "
                       f"(first few: {missing[:5]})")
    return [cache[str(r.row_id)] for r in rows]
