# v0.13 receiver arithmetic ladder

The preregistration defines three direct-call controls: local prefix, adding an offset to an already-computed prefix vector, and the combined receiver operation. No simulator tools or inter-agent messages are involved.

Start one pinned local server at a time using the settings and GGUF hashes in `preregistration.json`. Set `PYTHONPATH` to `.cache/python-packages` and `TLU_GGUF_PATH` to the selected model file, then run:

```powershell
python experiments/pilot_v0_13/run_arithmetic_ladder.py --model Qwen3-4B
```

Stop that service, start Qwen3-8B with identical server options, set the 8B `TLU_GGUF_PATH`, and run the same command with `--model Qwen3-8B`.

The runner checks the pinned task suite, upstream engine revision, and model asset before its unscored warmup. Raw model requests and responses stay under `.cache/pilot_v0_13/<model>/`. Analyze both traces with:

```powershell
python experiments/pilot_v0_13/analyze_arithmetic_ladder.py --runs `
  .cache/pilot_v0_13/Qwen3-4B/arithmetic_ladder_runs_<run-id>.jsonl `
  .cache/pilot_v0_13/Qwen3-8B/arithmetic_ladder_runs_<run-id>.jsonl
```
