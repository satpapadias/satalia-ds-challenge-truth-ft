# Deployment plan — Cloud Run

The GCP posture for deploying the six services. Written 2026-08-14, before any
Terraform has been written.

**Source of the facts below.** A read-only review of a peer solution to the same
challenge, already deployed against this organisation's policies. Provenance in
`docs/decisions.md` (2026-08-14); the review itself is kept outside this
repository. These are **observations from a running deployment**, not inference
from documentation — which is exactly why they are worth more than the guesses
they replace.

One caveat carried through: the peer solution targets project
`x-wppai-dataspine-choreo-dev`. Whether our target project carries identical org
policies is **not** established, and every item below should be re-verified
against the project we actually deploy into.

---

## 1. `allUsers` is blocked

**`iam.allowedPolicyMemberDomains` (Domain Restricted Sharing) blocks `allUsers`
bindings on Cloud Run in this organisation.** This is direct testimony from the
peer repository's own README, describing a design they had to abandon:

> The original design ran sub-agents with `--allow-unauthenticated` … simple
> code, but relies on the org allowing `allUsers` bindings. **When our org's
> `iam.allowedPolicyMemberDomains` policy blocked those bindings on fresh service
> creation**, we moved to per-hop OIDC.

**Consequence: our planned public orchestrator does not survive the port.** The
design of "one Cloud Run service with `--allow-unauthenticated` and a bearer
token checked in the application" is not deployable here. The bearer check itself
stays — it is still the right thing at the public edge — but it can no longer be
the *only* thing standing between the internet and the service, because the
service cannot be reachable from the internet that way at all.

---

## 2. A green deploy is not evidence

**`gcloud run deploy --allow-unauthenticated` does not fail when the policy
blocks the binding.** It emits a warning, the service is created **private**, and
the command exits 0.

The peer repository demonstrates the trap rather than describing it: its README
states there are no `allUsers` bindings anywhere, while its deploy code still
passes `--allow-unauthenticated` for four of five services and its own flow
diagram labels the sub-agents "allow-unauthenticated". All three can be true at
once only if the flag silently did nothing — which is what happened. Their
services work because of the ID token on every hop, not because of the flag.

### The rule this gives us

**Any deploy step must verify reachability from outside, not trust exit code 0.**

Concretely, after each service deploys:

- assert the intended IAM state explicitly —
  `gcloud run services get-iam-policy <svc>` — rather than inferring it from the
  deploy succeeding
- for anything meant to be publicly reachable, **make an unauthenticated request
  from outside the perimeter and assert the status code** you expect
- treat a mismatch between intended and actual posture as a deploy failure

This generalises past IAM. The same class of error — a step that reports success
while having done nothing — is why the local checks assert reachability and
content (`/healthz` reports what was loaded, not just `ok`) rather than that a
container is running.

---

## 3. Exposure: the shared A2A Agents Gateway

The sanctioned path for exposing an agent to callers outside the project is a
**shared A2A Agents Gateway**, already running in this organisation. The pattern,
observed end to end:

1. The orchestrator deploys with `ingress=all` and **`--no-allow-unauthenticated`**.
2. The gateway's service account is granted `roles/run.invoker` on it.
3. An agent config is uploaded to a gateway config bucket
   (`gs://<project>-a2a-gateway-agent-config/agents/<agent-id>.json`) with
   **`--forward-id-token`**, so the gateway mints a Google ID token for the
   orchestrator on each call.
4. A **bearer token is minted into Secret Manager**; the gateway SA is granted
   `roles/secretmanager.secretAccessor` on that secret.
5. External callers present that bearer plus a caller header to the gateway; the
   gateway validates it and calls the orchestrator with the ID token.

**Registration carries budget metadata** — a monthly budget in USD, an estimated
cost per call, and optional content capture with a character cap. That is an
org-level spend control we did not know existed, and it is a partial answer to
the gap recorded in `CLAUDE.md` §2: the spend gate that cannot live inside the
tool can live at the gateway.

**One structural note.** The orchestrator cannot be `ingress=internal`: the
gateway runs in a different project, so its traffic is not classified as internal
to our VPC. Ingress stays `all`, and IAM is what enforces auth.

### What this changes for us

- `/verify` remains the public contract, reached through the gateway rather than
  directly.
- The public bearer moves from an environment variable to Secret Manager.
- The reviewer is given a gateway URL, an agent id and a bearer — not a Cloud Run
  URL.
- Our in-app `BearerAuth` stays as defence in depth. It is no longer the only
  control, which is the point.

---

## 4. Agent-to-agent auth becomes per-hop OIDC

The shared `AGENT_TOKEN` is replaced by **Google-signed OIDC ID tokens minted per
hop**, bound to the target service's base URL. Cloud Run's IAM edge validates the
token and checks the caller's service account for `roles/run.invoker`.

This is better than what we planned, independently of the policy that forces it:
per-hop caller identity appears in Cloud Run logs, and there is no shared secret
to distribute or rotate.

### Exactly which files change

| file | change |
|---|---|
| **new** `truthclf_agents/gcp_auth.py` | `mint_id_token(audience)` via `google.oauth2.id_token.fetch_id_token`; an `httpx.Auth` flow for the A2A hop; a `header_provider`-style callable for the MCP hop; an `is_cloud_run_url()` guard so `localhost` skips minting and compose keeps working unchanged |
| `truthclf_agents/peers.py` | `discover()` builds its long-lived `httpx2.AsyncClient` with the ID-token auth flow instead of `headers={"Authorization": f"Bearer {token}"}`. The existing per-request `event_hooks` trace injection is untouched and still needed |
| `truthclf_agents/mcp_client.py` | `call_tool` and `probe` attach an ID token for the MCP server's base URL alongside the trace headers already set at construction |
| `truthclf_agents/serve.py` | the middleware stops requiring `AGENT_TOKEN` on the A2A route for agent-to-agent traffic — Cloud Run's IAM edge has already authenticated the caller before the request reaches the app. The orchestrator's public `/verify` keeps its bearer check |
| `truthclf_agents/common.py` | `BearerAuth` narrows to the public edge. It must not silently become optional: if the public token is unset, the orchestrator still refuses to start |
| `truthclf_agents/orchestrator.py`, `zero_shot.py`, `fine_tuned.py`, `explainer.py` | `BearerAuth("AGENT_TOKEN")` removed from the three specialists; the orchestrator keeps `ORCHESTRATOR_TOKEN` |
| `docker-compose.yml` | `AGENT_TOKEN` drops out of the shared agent env; local runs rely on the `localhost` guard |
| `tests/` | a test that the guard skips minting for `localhost` and attaches for a `.run.app` audience, so local dev cannot silently start requiring credentials |

**Do not remove the bearer check from the public edge.** IAM authenticates the
*gateway*; the bearer authenticates the *caller*. They answer different questions.

### One dependency to verify, not assume

The peer repository's README claims Terraform grants the runtime compute SA
`roles/run.invoker` on the inner services. **No such grant exists in its code.**
The calls most likely succeed because the default compute service account holds
project-level Editor via `iam.automaticIamGrantsForDefaultServiceAccounts`.

**If that constraint is enforced in our project, per-hop OIDC alone is not
enough** and explicit `run.invoker` grants become mandatory. Check this before
relying on the pattern, and grant explicitly regardless — a least-privilege
service account per service is the right shape, and depending on a default SA's
Editor role is not.

---

## 5. The fine-tuned model becomes a live service

**A tuned Gemini model is served per-token**, invoked through the ordinary
`generate_content(model=<endpoint>)` path using the endpoint resource name from
the finished tuning job. Observed in a real deployment in this organisation:
no machine type, no replica count, no `deploy()` call, no teardown, and nothing
in that repository's Terraform that provisions or destroys a serving endpoint.

Two operational facts that came with it:

- `job.tuned_model.endpoint` is the inference handle. **`job.tuned_model.model`
  returns 404.**
- Vertex Gemini SFT requires a **regional** location; `global` rejects tuning,
  and the client must be pinned to that region for tuned-endpoint inference.

### The consequence for our architecture

**`cached_replay` stops being the common case, and the fine-tuned agent becomes a
live service.** The stored-probability path exists because Together could not
serve our adapter without an always-on 2×H100 endpoint. That constraint does not
exist here.

What follows:

- `fine_tuned_source` defaults to **`live`**. The `cached` path is retained for
  offline reproduction of the Together-era record, not as the serving mode.
- **`FineTunedRowNotCached` stops being the normal response** to a caller's own
  statement. Partial coverage — the `unavailable` status, `missing_row_ids`, the
  `on_missing="omit"` mode — becomes an offline-replay concern rather than the
  everyday path.
- **The reconciliation logic does not change.** This is the payoff from the
  design: every non-`ok` source status lands in one branch, so a live endpoint
  *removes a condition* rather than changing the pooling. `applied: false` and
  `single_source` stay exactly as they are, and simply fire far less often.
- **`explain` with the fine-tuned model becomes possible.** It is currently
  refused with `CounterfactualNotAvailable` because a store keyed on `row_id`
  cannot answer occlusion queries. A live endpoint can. The refusal stays for
  the cached path.
- **The statement-identity binding stays.** `ft_eval_identity.json` guards the
  cached path, which still exists. Do not remove it because the live path made it
  usually-unnecessary.

**One thing to re-measure rather than assume:** the explainer's cost. Six
occlusion calls plus a rationale call per point was priced against Together. The
ceiling of 50 points per `explain` call should be re-derived against Gemini
pricing and latency before it is treated as settled.

---

## 6. Terraform

### What the peer solution does, and why we should not copy it

- **No `backend` block — state is local**, on the operator's laptop. For a system
  whose deploy also writes service URLs and a bearer token back into a local
  `.env`, that machine is the only source of truth for the deployment.
- Terraform declares four resources — Artifact Registry, a GCS bucket, one bucket
  IAM binding, and a `null_resource` that shells out to `gcloud` for Private
  Google Access. Everything that constitutes the running system — five images,
  five services, ingress posture, VPC egress, env vars, IAM, gateway
  registration — is a Python script invoked from a second `null_resource`, with a
  `deploy_trigger` variable whose only purpose is to force it to re-run.

It works, and the reasoning is documented honestly in their code. But the
challenge brief explicitly discourages "a purely imperative `deploy.sh` of CLI
calls", and this is that inside a Terraform wrapper. It also means
`terraform plan` cannot tell you what will change about the services.

### What we should do

**A GCS backend and real resources.**

```hcl
terraform {
  backend "gcs" {
    bucket = "<project>-truthclf-tfstate"
    prefix = "cloud-run"
  }
}
```

Declared as `google_cloud_run_v2_service` resources — six of them — plus
`google_artifact_registry_repository`, `google_service_account` per service,
`google_cloud_run_v2_service_iam_member` for the invoker grants,
`google_secret_manager_secret` and `_version` for the public bearer, and
`google_project_iam_member` for the Vertex and GCS roles. `terraform plan` then
shows a change of ingress, image or env var as a diff.

### What must be bootstrapped, and the ordering problem

**A GCS backend cannot create the bucket that holds its own state.** That is a
genuine chicken-and-egg, and it has exactly three honest resolutions:

1. **Use an existing org-standard state bucket** if one exists. Cheapest, and
   the reason `docs/deployment-plan.md` asks the question — the peer repository
   establishes only that *they* did not use one.
2. **Create the bucket once, out of band**, with a documented one-liner, then
   `terraform init` against it. One command in the README, never repeated.
3. **A two-stage layout** — a tiny `bootstrap/` with local state that creates
   only the bucket, then the main configuration with the GCS backend. Most
   rigorous, most machinery.

**Recommendation: (1) if a convention exists, otherwise (2).** A single
documented `gcloud storage buckets create` is honest about being a bootstrap
step; a second Terraform root to create one bucket is more moving parts than the
problem deserves. **This is a decision to confirm, not to assume** — it depends
on the answer to the state-location question, which the peer repository does not
provide.

Also to bootstrap before the first apply, in the same documented step:

- **APIs enabled**: `run`, `cloudbuild`, `artifactregistry`, `secretmanager`,
  `compute`, `storage`, `aiplatform`.
- **Private Google Access on the subnet**, if any service uses
  `--vpc-egress=all-traffic`. Terraform cannot own the default subnet without
  importing it. **Only needed if we route egress through the VPC** — which
  follows from making inner services `ingress=internal`, and is a posture
  decision to take deliberately rather than inherit.
- **The public bearer value**, supplied as a variable and written to Secret
  Manager; never committed.

### Images

Build with **Cloud Build** rather than locally: no local Docker on the work
laptop, images land in Artifact Registry directly. Note `options: logging:
CLOUD_LOGGING_ONLY` in the build config — without it Cloud Build wants a logs
bucket and fails when the default service account cannot write one.

Two images, matching the local split: `truthclf-tools` for the two MCP servers
and `truthclf-agent` for the four agents. **The tools image contains `data.csv`
and is as confidential as the dataset — private Artifact Registry only, inside
the same project.**

---

## 7. Open questions for the target project

Not answered by the peer repository, and each changes the Terraform:

1. **Is there an org-standard Terraform state bucket?** Decides §6's bootstrap.
2. **Is `iam.disableServiceAccountCreation` enforced?** Decides whether six
   least-privileged service accounts are possible or whether all six services
   share a provided one.
3. **Is `iam.automaticIamGrantsForDefaultServiceAccounts` enforced?** Decides
   whether the per-hop OIDC pattern works without explicit `run.invoker` grants
   (§4).
4. **Is Binary Authorization enforced?** Would add an attestation step to the
   build.
5. **Which regions does `gcp.resourceLocations` permit?** Decides every
   `location` argument, and must include a region that supports Vertex Gemini
   tuning (§5).
6. **Does our target project have the A2A Gateway available**, or is that
   specific to the project the peer solution deployed into?
