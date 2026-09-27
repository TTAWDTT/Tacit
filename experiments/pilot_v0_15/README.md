# v0.15 receiver output-contract diagnostic

The preregistered paired study compares direct JSON-array output with an XML `submit_result` wrapper. It reuses v0.3 task inputs and a synthetic successful receive transcript; it does not measure model-initiated communication.

Start the pinned Qwen3-8B llama.cpp server (same settings as v0.14), set `PYTHONPATH=.cache/python-packages` and `TLU_GGUF_PATH=.cache/models/Qwen3-8B-Q4_K_M.gguf`, then run:

```powershell
python experiments/pilot_v0_15/run_output_contract.py --model Qwen3-8B
```

The runner checks the task manifest, upstream engine, and model file before warmup. Raw traces are saved to `.cache/pilot_v0_15/Qwen3-8B/`. Publish the aggregate and per-episode results with:

```powershell
python experiments/pilot_v0_15/analyze_output_contract.py --runs `
  .cache/pilot_v0_15/Qwen3-8B/output_contract_runs_<run-id>.jsonl
```
