# HiddenBench Qwen3-1.7B capability/resource screen v0.9

This is a preregistered local screening experiment, not a communication-format comparison. It checks whether the already-cached Qwen3-1.7B checkpoint can meet the same four-vote, full-information eligibility gate used by the Qwen3-8B v0.8 screen, with a GPU-backed Transformers endpoint and tighter resource limits. A pass does not establish hidden-information ability or protocol value.

The model and benchmark artifacts are pinned in [`preregistration.json`](preregistration.json). The runner refuses to proceed if either safetensors shard, the benchmark revision, or benchmark checksum differs. Raw votes, assigned facts, and request-usage logs stay in ignored `.cache/pilot_hiddenbench_v0_9/`; only a sanitized aggregate is suitable for publication.

## Safeguards

- At most four generated votes; each vote gets one model request and one format attempt.
- The endpoint binds only to `127.0.0.1`, uses two PyTorch CPU threads, one worker, and below-normal process priority.
- The launch gate requires three CPU samples averaging below 25%, GPU utilization below 50%, GPU memory below 2500 MiB, at least 6000 MiB of free system memory, and unused port 8000.
- The launcher samples host CPU, process CPU, working set, GPU use, and device memory every 15 seconds. It stops the runner and endpoint at the preregistered limits and always cleans both up.
- `/v1/models` readiness checks do not issue a generation request. No model warmup or unscored inference is performed.

## Run

First run the artifact and idle-resource preflight without loading the model:

```powershell
./experiments/hiddenbench_v0_9/run_local_capability.ps1 -PrepareOnly
```

If the machine is responsive and you want to perform the preregistered screen, run:

```powershell
./experiments/hiddenbench_v0_9/run_local_capability.ps1
```

The launcher rejects a busy machine without loading the model. It writes private runtime traces to `.cache/` and prints the sanitized local aggregate path after a successful run. Do not publish raw prompts, facts, votes, or rationales.
