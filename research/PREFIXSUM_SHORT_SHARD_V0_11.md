# PrefixSum v0.11 short-shard role-capability calibration

This preregistered study measures sender/receiver capability at held-out segment lengths 2, 3, and 4 under one compact-KV message condition. It does not rank protocols.

Task manifest SHA-256: `c6302480780c7bea1c54ebc00436c06d57e092b285204139a20311685e3f7d33`. Runtime: llama.cpp b11202, commit fcb3074f2, Windows x64 CUDA 12.4, server at 127.0.0.1:8000.

Each cell has 8 episodes. Oracle outputs are excluded from model capability counts. In the receiver arm, the oracle sender emits the correct subtotal; a receive counts only if Agent 1 gets a nonempty payload before submitting.

## Model sender + oracle receiver

| Model | L | Qwen Agent 0 exact | Subtotal faithful | Joint exact | Message syntax |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B Q4_K_M GGUF | 2 | 5/8 | 5/8 | 5/8 | 8/8 |
| Qwen3-4B Q4_K_M GGUF | 3 | 5/8 | 2/8 | 1/8 | 8/8 |
| Qwen3-4B Q4_K_M GGUF | 4 | 6/8 | 1/8 | 1/8 | 7/8 |
| Qwen3-8B Q4_K_M GGUF | 2 | 7/8 | 8/8 | 7/8 | 8/8 |
| Qwen3-8B Q4_K_M GGUF | 3 | 5/8 | 4/8 | 3/8 | 8/8 |
| Qwen3-8B Q4_K_M GGUF | 4 | 6/8 | 3/8 | 3/8 | 8/8 |

## Model receiver + oracle sender

| Model | L | Correct oracle messages received before submit | Qwen exact given receive | Qwen local-prefix-only output |
|---|---:|---:|---:|---:|
| Qwen3-4B Q4_K_M GGUF | 2 | 0/8 | 0/0 | 2/8 |
| Qwen3-4B Q4_K_M GGUF | 3 | 0/8 | 0/0 | 0/8 |
| Qwen3-4B Q4_K_M GGUF | 4 | 0/8 | 0/0 | 0/8 |
| Qwen3-8B Q4_K_M GGUF | 2 | 8/8 | 0/8 | 1/8 |
| Qwen3-8B Q4_K_M GGUF | 3 | 6/8 | 0/6 | 2/8 |
| Qwen3-8B Q4_K_M GGUF | 4 | 1/8 | 0/1 | 1/8 |

## Interpretation

Short shards exposed a sender-side operating region, but not a two-model communication region. At L=2, the model sender plus oracle receiver completed 5/8 Qwen3-4B episodes and 7/8 Qwen3-8B episodes. Sender subtotal fidelity and Agent 0's own output are separate: a faithful message alone is not a joint success.

On the receiver side, Qwen3-4B received no payload before submission in 24 episodes. Qwen3-8B received the correct oracle message in 15/24 episodes, but returned an exact global segment in 0/15 received-message cases. Thus shorter inputs did not resolve receiver-side tool use/integration in this setup. The receiver result combines model behavior with the pinned simulator and interaction loop; it does not isolate arithmetic alone.

No tested length supports a protocol comparison yet because the model receiver never completed the task, despite correct oracle inputs in the cases where it received them. Next, isolate message acquisition from message application with a preregistered direct-context control, keeping the current task suite and exact-output scorer. Do not interpret these outcomes as evidence that compact-KV is more efficient or that a new language is needed.

Eight cases per cell and one greedy run do not support significance claims or decoding-variance estimates. This is one task family, one local engine, and two checkpoint/quantization configurations.

Per-episode submissions and exact wire messages: [JSONL run rows](data/PREFIXSUM_SHORT_SHARD_V0_11_RUNS.jsonl).
