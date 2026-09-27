# v0.11 short-shard role-capability calibration

The preregistration and fresh 24-case task suite were committed before model inference. Read `preregistration.json` for the frozen hypotheses, metrics, checksums, runtime, and limits.

Install/runtime assets are expected in this repository's ignored `.cache/` from the earlier local pilots. Start one model server at a time from the repository root. For Qwen3-4B:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath '.cache/runtimes/llama.cpp/llama-server.exe' `
  -ArgumentList @('--model','.cache/models/Qwen3-4B-Q4_K_M.gguf', `
    '--alias','Qwen3-4B-Q4_K_M','--host','127.0.0.1','--port','8000', `
    '--n-gpu-layers','99','--ctx-size','8192','--temp','0', `
    '--n-predict','256','--reasoning','off')
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_GGUF_PATH = '.cache/models/Qwen3-4B-Q4_K_M.gguf'
python experiments/pilot_v0_11/run_short_shard.py --model Qwen3-4B
```

Stop that server, start the Qwen3-8B Q4_K_M file with the same server options and its alias `Qwen3-8B-Q4_K_M`, then run:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_GGUF_PATH = '.cache/models/Qwen3-8B-Q4_K_M.gguf'
python experiments/pilot_v0_11/run_short_shard.py --model Qwen3-8B
```

The runner verifies every task against the pinned manifest, the upstream engine commit, and the selected model file's size and SHA-256 before a warmup or scored inference. Each run creates an ignored raw trace under `.cache/pilot_v0_11/<model>/`. Once both model traces exist, create the public report and exact per-episode records without model calls:

```powershell
python experiments/pilot_v0_11/analyze_short_shard.py --runs `
  .cache/pilot_v0_11/Qwen3-4B/short_shard_runs_<run-id>.jsonl `
  .cache/pilot_v0_11/Qwen3-8B/short_shard_runs_<run-id>.jsonl
```

Do not change model settings or prompts after inference starts. If the pinned hardware settings fail to load, stop and record a preregistered amendment before scored episodes.
