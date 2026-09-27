# HiddenBench Qwen3-14B full-information capability screen

v0.2 showed that Qwen3-8B did not approach the preregistered full-information gate. Before spending inference on a communication benchmark with this task set, v0.3 screens the cached Qwen3-14B checkpoint on the same three full-profile tasks using only the 12 independent initial votes. It calls HiddenBench's own `collect_initial_votes` implementation, preserving its prompts, fact assignment, JSON vote schema, and scorer definitions.

The run uses the previously feasible 24-GPU-layer, 8,192-context Qwen3-14B setup. Raw votes and rationales stay under ignored `.cache/pilot_hiddenbench_v0_3/`; only accuracy, token counts, and timing are eligible for publication.

Start the project-local service with the pinned artifact:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath '.cache/runtimes/llama.cpp/llama-server.exe' `
  -WorkingDirectory (Get-Location).Path `
  -ArgumentList @('--model','.cache/models/Qwen3-14B-Q4_K_M.gguf', `
    '--alias','Qwen3-14B-Q4_K_M','--host','127.0.0.1','--port','8000', `
    '--n-gpu-layers','24','--ctx-size','8192','--parallel','1', `
    '--temp','0','--n-predict','256','--reasoning','off')
```

Run `python experiments/hiddenbench_v0_3/run_capability_screen.py --prepare-only` to verify inputs and the idle local server. The default run performs 12 planned completions. A result below 0.8 means do not compare communication protocols on these tasks/model settings.
