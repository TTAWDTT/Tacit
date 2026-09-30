# Model experiment readiness report — 2026-09-30

**Purpose:** select the next executable model task from the existing research plan and report what is genuinely ready.
**Current disposition:** prepared; waiting for the resource manager to allocate the local CPU/GPU slot. No service, inference, training, or new resource poll was started for this report.

## Recommendation

There are two different “next experiments” in the project:

1. **Next model run, after resource allocation:** execute the already-preregistered [INDEX_m Qwen3-1.7B capability/direct-message pilot](../experiments/index_v0_3/README.md). It is the smallest frozen model screen in the formal plan, has an integrated idle gate and automatic cleanup, uses local-only weights, and is capped at 12 requests/300 seconds. It tests whether a candidate model can do the task and carry one structured message. It is a feasibility result, **not** evidence that a new language is better.
2. **Main scientific comparison still to complete:** run the [Emergent OOD v0.4 representation/onboarding study](../experiments/emergent_ood_v0_4/README.md) only after receiver capability is established, validation messages are available, the protocol/search and cap grid are frozen, and a cluster-aware confirmatory plan is complete. The current v0.4 infrastructure is substantial, but the formal clustered-analysis audit explicitly says it is not yet a confirmatory study plan.

This follows the existing [experiment plan](../docs/EXPERIMENT_PLAN.md): do a capability-positive local screen first, then compare optimized natural language, AutoForm, and valid structured text under a fixed communication policy and equal budgets. Do not present INDEX_m JSON or the four-task outcome as a Tacit language result.

## First command after the manager allocates the machine

Use one bounded launch. The PowerShell launcher performs its own three-sample resource gate before hashing task/model artifacts; if eligible, it starts the local service and runner, applies the run limits, and cleans up only its own child processes:

```powershell
& .\experiments\index_v0_3\run_local_capability.ps1
```

Do not run `-PrepareOnly` and then immediately repeat the full launch: that would take a second resource sample without a material state change. Do not run the command during the current resource reservation. If its integrated gate rejects, it exits before model hashing/service startup; retry only after the resource manager reports a materially changed allocation.

### Frozen run contract

- Four preregistered tasks: two episodes each at `m=4` and `m=8`, selected by fixed indices from the pinned v0.2 task shard.
- Up to four full-information calls first. The communication arm runs only if all four answers are strict, exact, and untruncated; it then adds one sender and one receiver call per episode, for at most 12 total requests. No retries.
- Model: `Qwen3-1.7B`, repository-local safetensors, float16 on CUDA, pinned Hugging Face revision and shard/tokenizer hashes in [`preregistration.json`](../experiments/index_v0_3/preregistration.json). The communication payload is the existing one-way JSON bit vector.
- Maximum 24 generated tokens per request, 60-second request timeout, 120-second server startup allowance, 300-second total run cap. Output and resource telemetry stay under ignored `.cache/index_v0_3/<timestamp>/`; publish sanitized outcomes only.
- The gate requires mean host CPU `<20%`, every sample `<30%`, GPU utilization `<25%`, GPU memory `<1,800 MiB` before load, free system RAM `≥6,000 MiB`, and port 8001 idle. Runtime is one worker/one PyTorch-OMP-MKL CPU thread at BelowNormal process priority. It stops its own run at CPU `≥45%`, GPU `≥85%` in two consecutive 10-second samples, free RAM `<4,000 MiB`, or 300 seconds.

The two pinned model shards total 4,063,515,592 bytes. This is the artifact payload only; CUDA runtime/working memory is additional. The local model directory and `.cache/index_v0_2/tasks.jsonl` currently exist by path metadata, but **their contents and hashes have not been validated in this state**. Attempt 10 rejected before either artifact was verified. The launcher performs the authoritative checks only after its resource gate passes.

## Resource-gate evidence

No resource poll was repeated for this report, per the active resource coordination. These are archived observations, not claims about the machine's present state:

| Evidence | Timestamp / result | Measured reason for rejection | What the record says did not happen |
|---|---|---|---|
| Shared three-port report `.cache/emergent_ood_v0_3/resource_preflight_followup-20260930-0818-allports.json` | `2026-09-30T08:19:04Z`, `rejected`; SHA-256 `5ce17199359d40e601757246a01e75ca6177449ce156150b801d513a713c42e2` | CPU samples 10.98%, 35.22%, 17.23%; mean 21.14% (limit `<20%`), max 35.22% (limit `<30%`); GPU utilization 49% (limit `<25%`). Free RAM 12,502 MiB and GPU memory 1,779/8,188 MiB passed; ports 8000/8001/8002 were idle. | No model artifact hashed/read, no load, service start, or inference request. |
| Candidate-specific `experiments/index_v0_3/PRECHECK_ATTEMPT_10.json` | `2026-09-29T01:30:23Z`, `idle_gate_not_met`; SHA-256 `BFCC6572CC5B95CEDFD01BA05D7631DAE4E983EB0F84C121156BA0797B858C4E` | CPU mean 33.8%, max 35.5%; GPU utilization 60%; free RAM 2,892 MiB. GPU memory use was 953 MiB; port 8001 was unused. | Rejected before artifact verification; no model load, service start, or inference request. |

The first is the latest archived shared-host evidence, not a fresh check. The manager's allocation is the authority for whether the resource slot has changed; this session did not inspect or alter any other task's processes.

## Existing scientific entry points and actual gaps

### INDEX_m prerequisite screen

| Component | Current state |
|---|---|
| Preregistration | Frozen and hash-identified: [`preregistration.json`](../experiments/index_v0_3/preregistration.json), SHA-256 `6de313be4242c524ea05e3d827d660316c98dd7a4cf1aecbef9f81703767b861`. It pins four episode IDs, exact success rules, the full-information gate, request ceiling, stop rules, and exclusions. |
| Data | Source generator is `experiments/index_v0_1/generate_tasks.py`; v0.2 task bundle is `.cache/index_v0_2/tasks.jsonl`, manifest SHA-256 `637e79254f4dc8f10e65c184a5653ffbf784ab86ecfa98795073e2c68ddedcab`. Path exists, but bundle verification remains pending because the resource gate rejected first. |
| Model | Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, float16/CUDA. Both weight SHAs and tokenizer SHA are preregistered. Local directory exists; artifact identity is unverified. |
| Runner/server | [`run_local_capability.ps1`](../experiments/index_v0_3/run_local_capability.ps1) integrates gate, artifact verification, hidden local server, runner, resource monitor, and scoped cleanup. [`run_capability_pilot.py`](../experiments/index_v0_3/run_capability_pilot.py) enforces the exact task and output scorer. Server implementation: `research/local_chat_server.py`. |
| Remaining gap | No capability-positive result exists. The four-task screen cannot estimate a stable accuracy rate and does not compare languages. |

### Emergent OOD v0.4 main representation study

- **Task / data:** one sender receives a private four-attribute meaning; the receiver receives a balanced four-candidate table. The default ontology has `4^4=256` meanings, with 192 lower-order training meanings and 64 held-out compositions. The episode generator allocates held-out meanings to validation/test and samples repeated balanced candidate sets; target rows within one composition split are not independent population replications. Keep the split seed as the inferential cluster and follow the [clustered-analysis audit](EMERGENT_OOD_CLUSTERED_ANALYSIS_AUDIT_V0_1.md). The current 16-set default is a plumbing fixture, not the confirmatory sample size.
- **Available arms:** no message, full information, plain natural language, AutoForm-style format choice, JSON, fixed symbolic code, shared protocol card, and usage-only receiver transfer. English is not development-optimized. The full OPRO-style natural-language controller exists but has no model run; the GEPA path is guarded, the real GEPA package is uninstalled, and no GEPA search has run.
- **Implementation / evaluation:** [`runner.py`](../experiments/emergent_ood_v0_4/runner.py) enforces role separation, exact candidate-ID scoring, 12 calls per batch, measured application bytes, and a hard per-episode wire cap. `induce_protocol_cards.py`, `generate_usage_examples.py`, `shuffle_usage_examples.py`, and `replay_usage_messages.py` provide train-only proposal/onboarding/control paths. `tools.paired_report` and `tools.frontier_report` analyze paired results and matched-budget strata.
- **Frozen design protections:** fixed train/validation/test support split and scorer rules; test ledger must remain unopened during card/search/cap selection; protocol card, model/tokenizer population, common cap grid, comparison family and stopping rule must be frozen before test. The resource/capability gates remain mandatory. A previous Qwen3-4B interface screen returned zero strict-format-valid receiver outputs in 12 attempts under each of three server response modes, although some raw strings ended in the gold ID; it is a format diagnostic, not authorization or evidence for changing the scorer.
- **Blocking gaps:** no fresh resource clearance; no eligible, independently verified receiver capability ledger for the selected model; no model-backed validation messages or protocol induction; no selected/pinned sender/receiver model and tokenizer population; no independent-split confirmatory count, practical effect, multiplicity family, or stopping plan; no validation-derived binding common cap grid; no locked final protocol. The 4,096-byte default alone does not test bandwidth pressure. These are why v0.4 is not the immediate run.
- **Directly blocking documentation defect fixed in this report:** the v0.4 README invoked `resource_preflight.ps1` through `python`, even though that file starts with a PowerShell `param(...)` block. The documented invocation now uses PowerShell script syntax. No script was executed to make this correction.

## Required sequence after resource coordination

1. Wait for the resource manager to assign this task a slot; do not poll or compete with the ocean-solver session.
2. Run the single INDEX_m launcher command above. Its own gate is the first resource sample for that run.
3. If the full-information gate fails, stop at four calls and report only the capability/interface failure. If it passes, allow its eight preregistered message calls to complete; inspect the sanitized exact outcomes and scoped telemetry. Treat any result as feasibility only.
4. Only after that, choose a capability-qualified model/task configuration for the v0.4 validation pilot. Use train-only development/search and validation-only protocol/cap selection. Freeze an independent-split analysis plan and test seed derivation before opening any test role files. Do not claim superiority if the local resource ceiling cannot support the declared sample.

No model or resource command was run while preparing this report. The project remains active and incomplete.
