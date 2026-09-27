# CRAFT Qwen3-8B parser-adapter feasibility v0.19

This is a new local feasibility run after two separate v0.17/v0.18 implementation blockers: the CRAFT parser discarded bracketed prose, and the 14B model stalled during a long request. v0.19 keeps the parser/UTF-8 adapter from v0.18 and switches to the cached Qwen3-8B Q4_K_M model, previously loaded with full GPU offload in this workspace.

The run uses the same medium structure (index 0), two turns, one message from each distinct Director per turn, and five oracle candidates for the Builder. It is not a language comparison or a held-out generalization test. It asks whether this task/model/runtime combination can deliver all three messages and make progress.

Start the pinned local server:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath '.cache/runtimes/llama.cpp/llama-server.exe' `
  -ArgumentList @('--model','.cache/models/Qwen3-8B-Q4_K_M.gguf', `
    '--alias','Qwen3-8B-Q4_K_M','--host','127.0.0.1','--port','8000', `
    '--n-gpu-layers','99','--ctx-size','8192','--temp','0', `
    '--n-predict','256','--reasoning','off')
$env:PYTHONPATH = '.cache/python-packages'
python experiments/craft_v0_19/run_craft_feasibility.py
```

The script verifies the local artifact hash and server model alias before calls. Raw traces stay under ignored `.cache/pilot_v0_19/`; publish only sanitized turn-level metrics and public messages, never private reasoning or full prompts.
