# INDEX_m Qwen3-1.7B capability pilot v0.3

This preregistered pilot tests one necessary prerequisite for protocol research: can one pinned local model read a tiny random bit vector and exact index, then pass the vector through a role-separated structured message? The message is plain JSON, an existing baseline. This is not a new-language proposal or a comparison of communication formats.

## Frozen design

The pilot selects episodes 0 and 1 at `m=4` and `m=8` from the public v0.2 task shard (seed `20260929`, SHA-256 pinned in [`preregistration.json`](preregistration.json)). It first makes at most four full-information model requests. Communication is attempted only if all four full-information answers are correct and untruncated. If eligible, each task gets one sender request and one receiver request, for a maximum of 12 model requests total. There are no retries. The sender sees only the vector; the receiver sees only the index and the exact sender output. The caller scores results against the private evaluator record after the request returns.

The references are a fixed-zero no-message baseline, a centralized oracle, and a one-way JSON-vector oracle. The pilot records UTF-8 bytes, standalone recipient tokenizer tokens from the pinned tokenizer artifact, model input/output token counts, request/generation time, and resource telemetry. Outputs containing private task data remain under ignored `.cache/`; only sanitized costs and outcomes should be published.

## Resource limits

The launcher rejects before hashing model files or starting the service unless three host CPU samples average below 20% and each is below 30%, GPU utilization is below 25%, GPU memory is below 1,800 MiB, and at least 6,000 MiB system memory is free. The service uses one CPU thread, one worker, BelowNormal priority, and binds to `127.0.0.1:8001`. It automatically stops the runner and its own service at the preregistered CPU/GPU/RAM thresholds or 300-second cap. A resource rejection makes no model request; retry the same preregistered attempt only after host load has changed and the gate can be re-evaluated.

## Run

Regenerate the pinned task shard as shown in [`index_v0_2/README.md`](../index_v0_2/README.md), then use the prepare-only command to verify local artifacts after the idle gate passes:

```powershell
./experiments/index_v0_3/run_local_capability.ps1 -PrepareOnly
```

The full, bounded screen uses the same launcher without the switch:

```powershell
./experiments/index_v0_3/run_local_capability.ps1
```

If the idle gate rejects, no model weights are loaded. Do not relax the resource limits to force a run. A successful full-information screen only makes this four-episode direct-message pilot eligible; any outcome remains too small to establish a stable success rate, language advantage, transfer, or scaling law.

The first prepare-only preflight was rejected at 55.8% mean CPU (88.4% peak) and 4,146 MiB free memory. No model files were hashed and no service was started. The sanitized record is [`PRECHECK_ATTEMPT_1.json`](PRECHECK_ATTEMPT_1.json); a later gate check is eligible only after the machine's resource state has changed.

A later read-only recheck observed 32.8% mean CPU (35.1% maximum) and 5,672 MiB free memory, so the gate still does not pass. No launcher or model service was started for this recheck; see [`PRECHECK_ATTEMPT_2.json`](PRECHECK_ATTEMPT_2.json).
