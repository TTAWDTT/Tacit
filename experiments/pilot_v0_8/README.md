# PrefixSum role-explicit transport replication v0.8

The v0.7 audit found that Silo-Bench's generic XML parameter parser changes numeric and JSON `send_message.content`, while agents often reverse the intended sender/receiver roles. This frozen v0.8 protocol uses fresh PrefixSum seeds, explicit per-agent roles, and a runner-level lossless adapter for message content. Agent 0 must send before submitting; Agent 1 must receive before computing and may not send. The adapter also stores out-of-range tool integers as decimal strings in logs to avoid serialization overflow.

The experiment retains seven message conditions: shared scaffold, exact concise-English subtotal, compact `s=<sum>`, JSON subtotal, binary subtotal, full-shard JSON list, and no communication. The same local Qwen3-4B Q4_K_M and llama.cpp setup is used. All model calls are local. The protocol, seeds, prompts, parser adapter, and scoring are pinned before inference.

## Run

Start the pinned server:

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
python experiments/pilot_v0_8/run_prefixsum.py
python experiments/pilot_v0_8/analyze_prefixsum.py
```

Raw traces remain in ignored `.cache/pilot_v0_8/`. The analyzer reports direction/order compliance separately from message-format adherence and exact task success.
