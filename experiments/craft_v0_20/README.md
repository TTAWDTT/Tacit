# CRAFT single-slot local inference feasibility v0.20

v0.19 verified the cached Qwen3-8B Q4_K_M model and endpoint but stalled in the first 2,048-token prompt batch under llama.cpp's default four slots. v0.20 tests a resource-focused server configuration: one slot, logical batch 512, physical micro-batch 128. CRAFT calls Directors sequentially, so one server slot does not reduce runner-side concurrency. This change tests runtime feasibility only; it is not a communication protocol comparison.

The run retains the bracket-preserving Director parser and UTF-8 console adapter, the same CRAFT source revision and medium structure, two turns, three distinct Director messages per turn, and the oracle-assisted Builder. The fresh run seed is 320. At most eight model completions are counted.

Start the pinned llama.cpp server:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath '.cache/runtimes/llama.cpp/llama-server.exe' `
  -ArgumentList @('--model','.cache/models/Qwen3-8B-Q4_K_M.gguf', `
    '--alias','Qwen3-8B-Q4_K_M','--host','127.0.0.1','--port','8000', `
    '--n-gpu-layers','99','--ctx-size','8192','--parallel','1', `
    '--batch-size','512','--ubatch-size','128','--temp','0', `
    '--n-predict','256','--reasoning','off')
$env:PYTHONPATH = '.cache/python-packages'
python experiments/craft_v0_20/run_craft_feasibility.py
```

The runner verifies the exact model bytes/hash and endpoint alias. Raw inference traces stay in ignored `.cache/pilot_v0_20/`.
