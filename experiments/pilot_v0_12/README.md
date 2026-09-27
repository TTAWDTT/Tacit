# v0.12 receiver acquisition/application diagnostic

The paired diagnostic reuses the v0.11 24-case held-out suite. The injected condition appends an engine-shaped successful `receive_messages` transcript to Agent 1's initial history. It is synthetic capability instrumentation: do not count it as model tool use, a simulator receipt, or a communication protocol result.

Start one model at a time using the pinned llama.cpp settings in `preregistration.json`. For Qwen3-4B:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath '.cache/runtimes/llama.cpp/llama-server.exe' `
  -ArgumentList @('--model','.cache/models/Qwen3-4B-Q4_K_M.gguf', `
    '--alias','Qwen3-4B-Q4_K_M','--host','127.0.0.1','--port','8000', `
    '--n-gpu-layers','99','--ctx-size','8192','--temp','0', `
    '--n-predict','256','--reasoning','off')
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_GGUF_PATH = '.cache/models/Qwen3-4B-Q4_K_M.gguf'
python experiments/pilot_v0_12/run_receiver_diagnostic.py --model Qwen3-4B
```

Stop the server, start the Qwen3-8B GGUF at the same settings with alias `Qwen3-8B-Q4_K_M`, set `TLU_GGUF_PATH` to the 8B path, and run `python experiments/pilot_v0_12/run_receiver_diagnostic.py --model Qwen3-8B`.

The runner reuses v0.11 checksum verification for task manifest, upstream engine, and model file before warmup/inference. Raw traces stay in `.cache/pilot_v0_12/<model>/`. Publish aggregates and per-episode records after both runs:

```powershell
python experiments/pilot_v0_12/analyze_receiver_diagnostic.py --runs `
  .cache/pilot_v0_12/Qwen3-4B/receiver_diagnostic_runs_<run-id>.jsonl `
  .cache/pilot_v0_12/Qwen3-8B/receiver_diagnostic_runs_<run-id>.jsonl
```
