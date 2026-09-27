# HiddenBench Qwen3-14B natural-discussion baseline

This is the first full multi-agent HiddenBench baseline after Qwen3-14B passed the full-information gate in v0.4. It uses the pinned official simulator, three verification tasks, full and hidden profiles, four agents, and the official 15 sequential discussion rounds. No extra prompt or message-format intervention is applied; the communication baseline is the benchmark's natural discussion.

The run uses Qwen3-14B Q4_K_M with the v0.4 thinking-mode settings: reasoning on, reasoning budget 1,024, temperature 0.6, top-k 20, top-p 0.95, min-p 0, presence penalty 1.5, 24 GPU layers, and 8,192 context. The seed schedule matches the official task-index offsets. Each task runs in its own upstream CLI process, with raw JSON saved before the next task.

The 408 planned responses could require roughly 8.8 hours at the v0.4 observed mean service time; actual cost depends on generated message lengths and cache behavior. Failed API requests and invalid-vote retries can increase request count. Raw traces remain under ignored `.cache/pilot_hiddenbench_v0_5/`.

Start the local service:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath '.cache/runtimes/llama.cpp/llama-server.exe' `
  -WorkingDirectory (Get-Location).Path `
  -ArgumentList @('--model','.cache/models/Qwen3-14B-Q4_K_M.gguf', `
    '--alias','Qwen3-14B-Q4_K_M','--host','127.0.0.1','--port','8000', `
    '--n-gpu-layers','24','--ctx-size','8192','--parallel','1', `
    '--temp','0.6','--top-k','20','--top-p','0.95','--min-p','0', `
    '--presence-penalty','1.5','--seed','20260928','--n-predict','1536', `
    '--reasoning','on','--reasoning-budget','1024')
```

Run `python experiments/hiddenbench_v0_5/run_hiddenbench_v0_5.py --prepare-only` for preflight or omit the flag to start the full then hidden profiles.
