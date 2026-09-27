# v0.14 agent-scaffold context diagnostic

Run Qwen3-8B with the pinned local server options from v0.13, then set `PYTHONPATH=.cache/python-packages` and `TLU_GGUF_PATH=.cache/models/Qwen3-8B-Q4_K_M.gguf`:

```powershell
python experiments/pilot_v0_14/run_scaffold_context.py --model Qwen3-8B
```

The runner verifies the v0.3 task manifest, model asset, upstream engine commit, and exact upstream prompt-source hash before the unscored warmup. One JSONL trace under `.cache/pilot_v0_14/Qwen3-8B/` contains both conditions for all cases.

Generate the public summary and per-episode outputs with:

```powershell
python experiments/pilot_v0_14/analyze_scaffold_context.py --runs `
  .cache/pilot_v0_14/Qwen3-8B/scaffold_context_runs_<run-id>.jsonl
```

The prefilled receive transcript is synthetic context in both arms. It is never counted as model tool use or as a real message delivery.
