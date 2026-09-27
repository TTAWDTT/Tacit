# HiddenBench v0.1 local capability calibration

This is a local feasibility study using the official three-task verification split from HiddenBench. It asks whether Qwen3-8B can use the pinned runner's JSON-vote and multi-agent discussion flow before any message-format comparison.

## Frozen conditions

- Profile: `hidden` and `full` on the same three tasks, one run each.
- 15 sequential communication rounds, four agents, one worker, fixed task seed `20260927`, zero sampling temperature.
- Official HiddenBench simulator, prompts, vote schema, task data, and scorer are unchanged.
- The local OpenAI-compatible endpoint is the pinned Qwen3-8B Q4_K_M llama.cpp b11202 server, one slot, 4096 context, batch 1024, micro-batch 256.
- Each condition plans 204 successful completion responses (4 initial votes + 60 messages + 4 final votes per task); the two profiles plan 408 responses total. The official client retries failed API requests up to three times, and invalid vote responses can trigger additional vote calls, so 408 is a nominal plan, not a hard request cap. Count actual attempts and completed calls from the server log, and report any retries or early stop.

The pre-discussion full-profile accuracy is the individual-reasoning gate. If it is low, diagnose task/model reasoning first; do not compare communication formats. The hidden-profile pre/post scores show whether this model benefits from discussion on these items. This one seed and three related verification tasks are calibration, not benchmark-wide evidence.

Raw outputs contain private fact assignments, prompts, rationales, and full conversations. The runner keeps them under ignored `.cache/pilot_hiddenbench_v0_1/`; publish only a sanitized aggregate.

## Run

```powershell
python experiments/hiddenbench_v0_1/run_hiddenbench_v0_1.py
```

The runner verifies the local model and server, plus the exact third-party source/data revisions, before calling the model.
