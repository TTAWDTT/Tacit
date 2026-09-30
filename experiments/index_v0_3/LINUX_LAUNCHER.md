# Controlled Linux launcher

`run_local_capability.py` supervises the **unchanged** v0.3 runner and local
server. It does not alter scoring, model response parsing, task selection, or
preregistration. Python 3.10+ and Linux `/proc` are required; the live path also
requires working `nvidia-smi`, the existing local inference dependencies and the
pinned local artifacts. Nothing is installed or downloaded by this launcher.

From the repository root, artifact verification after the idle gate is:

```sh
python experiments/index_v0_3/run_local_capability.py --prepare-only
```

A full pilot uses the same command without `--prepare-only`. Neither command was
run against a real model during this implementation. Do not run the full pilot
without separate authorization for local inference and eligible host resources.
There are no CLI overrides for limits, commands, model paths, or resource readings.

## Frozen contract and Linux mapping

| Control | Linux behavior |
| --- | --- |
| Idle CPU | Three aggregate `/proc/stat` delta samples ending two seconds apart, after a one-second initial baseline; mean <20%, maximum <30%. Guest counters are not double counted. |
| Idle GPU | First `nvidia-smi` row: utilization <25%, used memory <1800 MiB. Its UUID is passed as `CUDA_VISIBLE_DEVICES` so telemetry and inference target the same device. |
| Idle RAM | `/proc/meminfo` `MemAvailable` >=6000 MiB; this is Linux's available-memory estimate, not `MemFree` alone. |
| Port | Reject any observed TCP listener on local port 8001, including IPv6. Reserve an IPv4 loopback socket before resource checks; only listen after artifact verification. Pass its descriptor to Uvicorn to avoid a check-to-bind race. No existing listener is terminated. |
| Artifact verification | Invoke the frozen runner with `--artifacts-only` only after the resource gate passes. Its task, shard size/SHA-256, tokenizer hash and pilot checks remain authoritative. Prepare-only never starts the model server. |
| Concurrency | One Uvicorn worker; PyTorch/OMP/MKL/OpenBLAS threads set to one; tokenizer parallelism disabled. Children execute with nice increment +5 before imports, the Linux analogue of BelowNormal. |
| Network | Bound to 127.0.0.1; proxy bypass for localhost; model/tokenizer offline flags plus the existing server's local-only loading. This is application configuration, not an OS network sandbox. |
| Work | Unchanged runner: at most 12 requests, 24 new tokens/request, 60s request timeout, no retries; communication only after the four-item full-information gate passes. The launcher also rejects a reported count outside 0–12. It does not intercept or reinterpret model responses. |
| Startup/run time | Monotonic 120s startup and 300s runner deadlines; startup probes every 4s with <=2s timeout; process checks every 0.2s. Resource queries have a 2s timeout. A blocking query can delay detection by up to that timeout; teardown adds at most 2s grace before SIGKILL. The 300s budget starts when the runner starts, excluding hash preflight and startup, as in PowerShell. |
| Automatic stops | During startup and running, 10s resource sampling; CPU >=45% or GPU >=85% twice consecutively, or available RAM <4000 MiB immediately. CPU/GPU counters reset independently on a low sample. Missing/malformed telemetry fails closed. |
| Cleanup | SIGINT/SIGTERM, timeout, exception, runner/server failure and success all clean up only the process groups created by this launcher. TERM then KILL handles resistant descendants even after their leader exits. SIGKILL of the launcher itself cannot be trapped; intentionally detached descendants are outside this process-group contract. |

Each attempt creates a unique, private directory under `.cache/index_v0_3/`.
`launcher.jsonl` records gate acceptance/rejection, child roles/PIDs, artifact
acceptance and a terminal reason/exit code. `resources.jsonl` records resource
readings; separate preflight/server/runner stdout and stderr logs aid diagnosis.
A resource rejection launches **no child**, hashes no model artifact and sends no
model request. A port rejection occurs before resource sampling. A prepare result
means artifact verification only; a completed launcher result retains the runner's
own status (including full-information gate failure).

## Reproduce offline validation

No pytest, inference libraries, task data, GPU, external service or model files are
needed. The test harness replaces resource readers and child commands **in-process**;
the public launcher exposes no bypass. It uses ephemeral loopback ports and actual
lightweight Python children, including a fake HTTP model-list endpoint. Time limits
are shortened only in tests.

```sh
TACIT_LAUNCHER_TEST_REPORT=/tmp/tacit-launcher-offline.json \
  python tests/test_index_linux_launcher.py -v
python -m compileall -q experiments/index_v0_3/run_local_capability.py \
  tests/test_index_linux_launcher.py tests/fixtures/index_launcher_fake.py
```

`LINUX_LAUNCHER_VALIDATION.json` captures the offline results. Tests cover independent
CPU mean/peak, GPU utilization/memory, RAM and port refusal; exact boundaries and
counter resets; unavailable/malformed telemetry; prepare-only; delegated hash
preflight failure; startup failure/timeout; runner timeout/failure; server death;
normal completion; malformed summary and excess request count; startup and runtime
resource stops; injected exceptions; SIGTERM; and resistant descendant cleanup.
An unrelated process is asserted alive after every lifecycle case. The pinned
runner source hash is checked to protect its request/scoring contract.

These results establish supervisor behavior under simulated conditions, **not model
capability or a scientific pilot result**. Actual artifact hashes, production
Uvicorn descriptor handoff, CUDA device selection, real resource readings and
end-to-end inference remain unverified. An unavailable pytest installation was
avoided by using the standard-library unittest runner; the broader repository
suite was not run for this launcher-only change.
