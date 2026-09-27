# PrefixSum v0.12 receiver acquisition/application diagnostic

A paired diagnostic of ordinary receive-tool use versus an explicitly injected successful receive transcript. The injection is an artificial capability probe; it is neither model-initiated communication nor evidence of protocol superiority.

Task manifest SHA-256: `c6302480780c7bea1c54ebc00436c06d57e092b285204139a20311685e3f7d33`. Runtime: llama.cpp b11202, commit fcb3074f2, Windows x64 CUDA 12.4; RTX 4060 8 GB; 99 GPU layers requested; context 8192; temperature 0; 256-token generation cap.

Each cell has 8 episodes. `Received` counts only payloads actually returned by the simulator before submit; injected transcript rows are counted separately.

| Model | Condition | L | Exact receiver outputs | Actual receives | Injected transcript | Local-prefix-only |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B Q4_K_M GGUF | injected_successful_receive_transcript | 2 | 0/8 | 0/8 | 8/8 | 0/8 |
| Qwen3-4B Q4_K_M GGUF | injected_successful_receive_transcript | 3 | 0/8 | 0/8 | 8/8 | 0/8 |
| Qwen3-4B Q4_K_M GGUF | injected_successful_receive_transcript | 4 | 0/8 | 0/8 | 8/8 | 0/8 |
| Qwen3-4B Q4_K_M GGUF | ordinary_receive_tool | 2 | 0/8 | 0/8 | 0/8 | 1/8 |
| Qwen3-4B Q4_K_M GGUF | ordinary_receive_tool | 3 | 0/8 | 0/8 | 0/8 | 0/8 |
| Qwen3-4B Q4_K_M GGUF | ordinary_receive_tool | 4 | 0/8 | 0/8 | 0/8 | 0/8 |
| Qwen3-8B Q4_K_M GGUF | injected_successful_receive_transcript | 2 | 0/8 | 0/8 | 8/8 | 1/8 |
| Qwen3-8B Q4_K_M GGUF | injected_successful_receive_transcript | 3 | 0/8 | 0/8 | 8/8 | 4/8 |
| Qwen3-8B Q4_K_M GGUF | injected_successful_receive_transcript | 4 | 0/8 | 0/8 | 8/8 | 5/8 |
| Qwen3-8B Q4_K_M GGUF | ordinary_receive_tool | 2 | 0/8 | 8/8 | 0/8 | 1/8 |
| Qwen3-8B Q4_K_M GGUF | ordinary_receive_tool | 3 | 0/8 | 7/8 | 0/8 | 2/8 |
| Qwen3-8B Q4_K_M GGUF | ordinary_receive_tool | 4 | 0/8 | 2/8 | 0/8 | 1/8 |

Paired exact-output transitions across the 24 same-seed tasks:

| Model | Both exact | Injection only | Ordinary only | Neither exact |
|---|---:|---:|---:|---:|
| Qwen3-4B Q4_K_M GGUF | 0/24 | 0/24 | 0/24 | 24/24 |
| Qwen3-8B Q4_K_M GGUF | 0/24 | 0/24 | 0/24 | 24/24 |

## Interpretation

Neither model returned an exact receiver segment in any of the 24 ordinary-path tasks or any of the 24 injected-transcript tasks. The paired outcome table is 24/24 neither-exact for each model: adding a visible successful receive transcript did not rescue a single episode. Qwen3-8B still returned the exact local prefix without the offset in 10/24 injected cases, which is consistent with offset neglect but does not identify its only cause.

For Qwen3-8B, the ordinary path delivered a payload before submission in 17/24 new replication episodes; exact output was still 0/24. Qwen3-4B received none in the ordinary path (0/24), but exact output also remained 0/24 when the transcript was prefilled. Message acquisition alone is therefore insufficient for these models/tasks. The intervention cannot distinguish arithmetic execution, instruction interpretation, and other harness/context effects without a simpler capability control.

These are descriptive results on the 24 reused cases, not significance claims. The v0.11 and v0.12 ordinary Qwen3-8B receipt counts differ (15/24 vs 17/24), despite the same seeds and greedy setting; this indicates run variability in this local setup and should be reported rather than hidden.

The injected transcript is a synthetic prior interaction in the harness. Its output must not be counted as successful model tool use, actual message delivery, or a viable communication protocol. Token counts include this extra context and are not an efficiency comparison.

Per-episode answers, actual receive status, injection records, and cost traces: [JSONL](data/PREFIXSUM_RECEIVER_DIAGNOSTIC_V0_12_RUNS.jsonl).
