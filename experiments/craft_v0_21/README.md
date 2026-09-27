# CRAFT reduced-context feasibility v0.21

This run isolates server context allocation after v0.20's single-slot setup processed one prompt batch but stalled before completing the request. It keeps the same Qwen3-8B model, task, parser adapter, one-slot sequential call schedule, and two-turn cap. It reduces the server context from 8,192 to 4,096 tokens and raises logical/physical batches to 1,024/256. The original prompt is approximately 2.8k tokens; the run stops if actual prompt plus generated context cannot fit, so no truncation is silently introduced.

This is a local-runtime feasibility diagnostic. It cannot rank communication formats. The local model call limit remains 8 completions (two turns × three Directors plus Builder).

Start the pinned llama.cpp server:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath '.cache/runtimes/llama.cpp/llama-server.exe' `
  -ArgumentList @('--model','.cache/models/Qwen3-8B-Q4_K_M.gguf', `
    '--alias','Qwen3-8B-Q4_K_M','--host','127.0.0.1','--port','8000', `
    '--n-gpu-layers','99','--ctx-size','4096','--parallel','1', `
    '--batch-size','1024','--ubatch-size','256','--temp','0', `
    '--n-predict','256','--reasoning','off')
$env:PYTHONPATH = '.cache/python-packages'
python experiments/craft_v0_21/run_craft_feasibility.py
```

Raw traces remain in ignored `.cache/pilot_v0_21/`.
