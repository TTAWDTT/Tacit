# HiddenBench Qwen3-8B reasoning-mode capability screen v0.7

This four-vote screen asks whether the cached 8B checkpoint can solve the West City task with full information, using the official Qwen3 thinking-mode sampling settings. It does not compare communication protocols. The preregistration is frozen in [`preregistration.json`](preregistration.json).

**Observed status:** the 2026-09-28 run was automatically stopped after GPU utilization reached 97% and 95% in consecutive samples. No vote was persisted or scored. See the sanitized [resource-stop report](../../research/HIDDENBENCH_QWEN8B_V0_7_RESOURCE_STOP.md) and [JSON summary](../../research/HIDDENBENCH_QWEN8B_V0_7_RESOURCE_STOP.json). The result cannot be used as evidence about model capability.

The runner uses the pinned HiddenBench `collect_initial_votes` function, official task and prompts, official answer labels, and official model-data checksums. Private raw assignments and rationales stay in ignored `.cache/`; only sanitized aggregate accuracy and resource summaries may be published.

## Resource safeguards

The PowerShell launcher requires three sampled CPU readings averaging below 35% and GPU utilization below 60%. It starts the model hidden at BelowNormal priority, with one GPU-resident slot and four CPU threads. It samples load every 15 seconds and automatically stops both runner and server after CPU reaches 75% for three consecutive samples or GPU reaches 90% for two. The server is also stopped in `finally` on normal exit or interruption.

The 8B model is already cached at `.cache/models/Qwen3-8B-Q4_K_M.gguf`; the runner verifies its byte count, SHA-256, benchmark source revision, and benchmark data checksum before voting. No download is needed.

From the repository root:

```powershell
./experiments/hiddenbench_v0_7/run_local_capability.ps1 -PrepareOnly
./experiments/hiddenbench_v0_7/run_local_capability.ps1
```

`-PrepareOnly` checks the idle-load gate and local pinned artifacts but does not load a model. The result can license only a separate preregistered hidden-information pilot if all four full-information votes are correct.
