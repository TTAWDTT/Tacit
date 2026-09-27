# PrefixSum protocol replication v0.7

This experiment tests whether v0.6's compact-value behavior transfers from a scalar hidden-sum task to a distributed prefix-sum task with list-valued outputs. It uses 12 deterministic held-out tasks (four each at segment lengths 6, 15, and 30), one fixed local model setup, and seven communication conditions: shared scaffold, concise English subtotal, compact key-value subtotal, JSON subtotal, binary subtotal, full-shard transfer, and no communication.

**Implementation status:** the first execution was stopped after an upstream Silo-Bench message-content coercion bug invalidated the JSON, binary, and full-shard payloads; trace review also found role reversal. The resulting 59 cells are not usable as a protocol comparison. See the [implementation audit](../../research/PREFIXSUM_PILOT_V0_7_AUDIT.md) and the corrected [v0.8 protocol](../pilot_v0_8/README.md).

The suite is a local adaptation of Silo-Bench task II-11 Prefix Sum; the upstream project is released under the Unlicense. The task generator verifies local outputs and information insufficiency for Agent 1. Protocol conditions, task manifest, prompts, model weights, Silo engine revision, and runtime are pinned in [`policies.json`](policies.json) before model calls. The formal prediction and its bit bound are recorded there too.

## Run

Start the same local runtime/model as v0.6, using the pinned settings:

```powershell
& .cache/runtimes/llama.cpp/llama-server.exe `
  --model .cache/models/Qwen3-4B-Q4_K_M.gguf `
  --alias Qwen3-4B-Q4_K_M `
  --host 127.0.0.1 --port 8000 --n-gpu-layers 99 --ctx-size 8192 `
  --temp 0 --n-predict 256 --reasoning off
```

Then run and analyze:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_GGUF_PATH = '.cache/models/Qwen3-4B-Q4_K_M.gguf'
python experiments/pilot_v0_7/run_prefixsum.py
python experiments/pilot_v0_7/analyze_prefixsum.py
```

All model calls are local. Raw traces remain in ignored `.cache/pilot_v0_7/`. Token totals are backend-reported; no matched-token-budget claim is planned.
