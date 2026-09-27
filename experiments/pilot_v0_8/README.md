# PrefixSum role-explicit transport replication v0.8

The v0.7 audit found that Silo-Bench's generic XML parameter parser changes numeric and JSON `send_message.content`, while agents often reverse the intended sender/receiver roles. This frozen v0.8 protocol uses fresh PrefixSum seeds, explicit per-agent roles, and a runner-level lossless adapter for message content. Agent 0 must send before submitting; Agent 1 must receive before computing and may not send. The adapter also stores out-of-range tool integers as decimal strings in logs to avoid serialization overflow.

The experiment retains seven message conditions: shared scaffold, exact concise-English subtotal, compact `s=<sum>`, JSON subtotal, binary subtotal, full-shard JSON list, and no communication. The same local Qwen3-4B Q4_K_M and llama.cpp setup is used. All model calls are local. The protocol, seeds, prompts, parser adapter, and scoring are pinned before inference.

## Outcome (exploratory)

All 84 episode-condition runs completed. The adapter delivered every attempted message in the communicating conditions, and Agent 1 received a non-empty payload before submitting in 8–12 of 12 episodes depending on the condition. Subtotal syntax was generally followed, but its value matched Agent 0's true shard sum in 0/12 runs for each subtotal encoding. Full-shard JSON had valid syntax and exact content in 12/12 runs. No condition produced a fully correct episode; even correct full-shard delivery did not reliably lead Agent 1 to apply the offset and return global prefix sums. Thus this pilot diagnoses arithmetic and task-execution weaknesses and provides no protocol ranking or efficiency claim. See [the report](../../research/PREFIXSUM_PILOT_V0_8.md).

The separate deterministic oracle control completed all 12 cases with exact joint success and faithful subtotal delivery in four rounds on average. It made zero model calls and uses the same message tools and scorer. This verifies task and runtime mechanics, but is not an LLM baseline. Its summary is preserved in [`PREFIXSUM_ORACLE_CONTROL_V0_1.json`](../../research/PREFIXSUM_ORACLE_CONTROL_V0_1.json).

The analyzer distinguishes send attempts, delivered messages, and successful non-empty receives. The no-communication arm intentionally records attempted send calls that the channel intervention rejects; these are not delivered messages.

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

To validate the task, message engine, adapter, and scorer without model calls, run the deterministic control (no local server required):

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/pilot_v0_8/run_oracle_control.py
```

Raw traces remain in ignored `.cache/pilot_v0_8/`. The analyzer reports direction/order compliance separately from message-format adherence and exact task success.
