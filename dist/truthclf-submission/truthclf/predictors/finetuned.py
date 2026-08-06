"""Fine-tuned predictor: LoRA supervised fine-tuning on Together, then prediction
through the resulting model.

`predict(points, labels=None)` has the same signature and behaviour as
ZeroShotPredictor.predict (it delegates to a ZeroShotPredictor pointed at the
served fine-tuned model), so the two are interchangeable.

Serving note: serverless serving of a custom LoRA adapter was not available for
our base model on Together, so the fine-tuned model is served via a dedicated
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
                 client=None, threshold=0.5, use_logprobs=True):
        self.base_model = base_model
        self.served_model = served_model      # endpoint/model id used at predict time
        self.variant = variant
        self.scheme = scheme
        self.client = client
        self.threshold = threshold
        self.use_logprobs = use_logprobs
        self.job_id = None
        self.output_name = None

    def fine_tune(self, training_dataset, val_rows=None, n_epochs=3, learning_rate=1e-5,
                  suffix="truthclf_sft", lora=True, train_on_inputs="auto",
                  poll=True, workdir="ft_data", poll_interval=30):
        """Prepare the data, split off a validation set, and launch a LoRA SFT job on
        Together. Pass a single labelled training_dataset (same format as data.csv);
        a speaker-disjoint train/validation split is made internally (an explicit
        val_rows overrides it). Polls to completion by default and stores output_name."""
        from together import Together
        if val_rows is None:
            train_rows, val_rows = data.speaker_disjoint_split(
                training_dataset, test_frac=0.2, seed=0, scheme=self.scheme)
        else:
            train_rows = training_dataset
        os.makedirs(workdir, exist_ok=True)
        client = Together()

        tr_path = os.path.join(workdir, "train.jsonl")
        data.write_sft_jsonl(train_rows, tr_path, self.variant, self.scheme)
        kwargs = dict(model=self.base_model,
                      training_file=client.files.upload(file=tr_path, purpose="fine-tune",
                                                        check=True).id,
                      lora=lora, n_epochs=n_epochs, learning_rate=learning_rate,
                      train_on_inputs=train_on_inputs, suffix=suffix)
        if val_rows:
            va_path = os.path.join(workdir, "val.jsonl")
            data.write_sft_jsonl(val_rows, va_path, self.variant, self.scheme)
            kwargs["validation_file"] = client.files.upload(file=va_path, purpose="fine-tune",
                                                            check=True).id
            kwargs["n_evals"] = 10

        resp = client.fine_tuning.create(**kwargs)
        self.job_id = getattr(resp, "id", None) or getattr(getattr(resp, "job", None), "id", None)

        if poll:
            while True:
                job = client.fine_tuning.retrieve(self.job_id)
                st = str(getattr(job, "status", "")).upper()
                if "COMPLET" in st:
                    break
                if any(x in st for x in ("FAIL", "ERROR", "CANCEL")):
                    raise RuntimeError(f"fine-tuning job ended with status {st}")
                time.sleep(poll_interval)
            job = client.fine_tuning.retrieve(self.job_id)
            self.output_name = (getattr(job, "model_output_name", None)
                                or getattr(job, "output_name", None))
        return self.output_name

    def _delegate(self):
        served = self.served_model or self.output_name
        if served is None:
            raise RuntimeError("no served model: call fine_tune() first or pass served_model=")
        client = self.client or llm.make_client(served)
        return ZeroShotPredictor(model=served, variant=self.variant, client=client,
                                 threshold=self.threshold, use_logprobs=self.use_logprobs)

    def predict(self, points, labels=None):
        return self._delegate().predict(points, labels=labels)

    def rationale(self, rows, max_tokens=64):
        return self._delegate().rationale(rows, max_tokens=max_tokens)
