# v0.16 local Qwen3-14B receiver capability

The frozen study runs 72 direct arithmetic calls plus 24 real simulator receiver episodes. It requires the official pinned Qwen3-14B Q4_K_M model (9,001,752,960 bytes) in `.cache/models/`; the runner verifies the SHA-256 before any warmup or scored call.

After download and checksum verification, start the pinned server with partial GPU offload:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath '.cache/runtimes/llama.cpp/llama-server.exe' `
  -ArgumentList @('--model','.cache/models/Qwen3-14B-Q4_K_M.gguf', `
    '--alias','Qwen3-14B-Q4_K_M','--host','127.0.0.1','--port','8000', `
    '--n-gpu-layers','24','--ctx-size','8192','--temp','0', `
    '--n-predict','256','--reasoning','off')
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_GGUF_PATH = '.cache/models/Qwen3-14B-Q4_K_M.gguf'
python experiments/pilot_v0_16/run_qwen14b_receiver.py --component all
```

The preregistration requires stopping before scored inference if this exact 24-layer/8192-context configuration cannot load. Do not change offload/context after model calls begin; preregister any feasibility amendment first.

Raw direct and hybrid traces are written to `.cache/pilot_v0_16/`. Publish the report and per-episode records with:

```powershell
python experiments/pilot_v0_16/analyze_qwen14b_receiver.py `
  --direct-runs .cache/pilot_v0_16/direct/direct_runs_<run-id>.jsonl `
  --hybrid-runs .cache/pilot_v0_16/hybrid/hybrid_runs_<run-id>.jsonl
```
