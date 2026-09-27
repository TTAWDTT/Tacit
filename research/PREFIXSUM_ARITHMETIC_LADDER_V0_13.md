# PrefixSum v0.13 receiver arithmetic ladder

Direct single-agent arithmetic controls, without tools or a multi-agent simulator. The same short-shard inputs are reused to isolate local prefix computation, scalar-offset addition to a supplied vector, and their composition.

Task manifest SHA-256: `c6302480780c7bea1c54ebc00436c06d57e092b285204139a20311685e3f7d33`. Runtime: llama.cpp b11202, commit fcb3074f2, Windows x64 CUDA 12.4; RTX 4060 8 GB; 99 GPU layers requested; context 8192; temperature 0; max 128 generated tokens.

Each cell has 8 cases. A response counts as parseable only if the full response is a JSON list of non-boolean integers.

| Model | Operation | L | Parseable | Exact | Mean input tokens | Mean output tokens |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B Q4_K_M GGUF | combined_receiver | 2 | 7/8 | 0/8 | 107.6 | 25.6 |
| Qwen3-4B Q4_K_M GGUF | combined_receiver | 3 | 8/8 | 0/8 | 111.2 | 14.9 |
| Qwen3-4B Q4_K_M GGUF | combined_receiver | 4 | 8/8 | 0/8 | 115.9 | 19.6 |
| Qwen3-4B Q4_K_M GGUF | local_prefix | 2 | 8/8 | 7/8 | 86.6 | 8.9 |
| Qwen3-4B Q4_K_M GGUF | local_prefix | 3 | 8/8 | 8/8 | 90.1 | 12.9 |
| Qwen3-4B Q4_K_M GGUF | local_prefix | 4 | 8/8 | 5/8 | 94.5 | 18.0 |
| Qwen3-4B Q4_K_M GGUF | offset_vector | 2 | 7/8 | 4/8 | 88.8 | 10.6 |
| Qwen3-4B Q4_K_M GGUF | offset_vector | 3 | 6/8 | 4/8 | 93.0 | 17.2 |
| Qwen3-4B Q4_K_M GGUF | offset_vector | 4 | 5/8 | 4/8 | 98.2 | 24.5 |
| Qwen3-8B Q4_K_M GGUF | combined_receiver | 2 | 8/8 | 4/8 | 107.6 | 9.9 |
| Qwen3-8B Q4_K_M GGUF | combined_receiver | 3 | 8/8 | 2/8 | 111.2 | 15.6 |
| Qwen3-8B Q4_K_M GGUF | combined_receiver | 4 | 8/8 | 1/8 | 115.9 | 20.9 |
| Qwen3-8B Q4_K_M GGUF | local_prefix | 2 | 8/8 | 7/8 | 86.6 | 8.9 |
| Qwen3-8B Q4_K_M GGUF | local_prefix | 3 | 8/8 | 8/8 | 90.1 | 12.9 |
| Qwen3-8B Q4_K_M GGUF | local_prefix | 4 | 8/8 | 8/8 | 94.5 | 17.9 |
| Qwen3-8B Q4_K_M GGUF | offset_vector | 2 | 8/8 | 5/8 | 88.8 | 9.6 |
| Qwen3-8B Q4_K_M GGUF | offset_vector | 3 | 8/8 | 6/8 | 93.0 | 15.2 |
| Qwen3-8B Q4_K_M GGUF | offset_vector | 4 | 8/8 | 8/8 | 98.2 | 20.8 |

## Paired totals across all 24 cases

| Model | Local prefix exact | Offset-vector exact | Combined receiver exact | Both substeps exact | Combined exact on those inputs |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B Q4_K_M GGUF | 20/24 | 12/24 | 0/24 | 10/24 | 0/10 |
| Qwen3-8B Q4_K_M GGUF | 23/24 | 19/24 | 7/24 | 18/24 | 5/18 |

## Interpretation

Qwen3-4B was exact on 20/24 local-prefix controls and 12/24 offset-vector controls, but 0/24 combined receiver controls. Qwen3-8B was exact on 23/24, 19/24, and 7/24 respectively. Among inputs where each model's separate local-prefix and offset-vector runs were both exact, the combined prompt was still exact on 0/10 (4B) and 5/18 (8B). This points to both offset arithmetic and task composition as unresolved execution bottlenecks; it does not isolate a single cause.

Unlike the simulator-based receiver conditions, the direct combined prompt produced 7/24 exact outputs for Qwen3-8B. That contrast suggests harness/context contributes to the gap, alongside the arithmetic/composition failures seen in direct calls. It is not a causal estimate because the direct prompt and simulator context differ substantially.

These controls cannot establish message-format efficiency or generalization beyond this task family.

This is a narrow diagnostic on reused task inputs with one greedy output each. Models and checkpoints are not a causal size comparison.

All prompts, raw responses, strict parses, expected arrays, and cost fields: [JSONL](data/PREFIXSUM_ARITHMETIC_LADDER_V0_13_RUNS.jsonl).
