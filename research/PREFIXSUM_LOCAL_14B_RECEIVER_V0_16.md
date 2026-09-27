# Qwen3-14B local receiver capability v0.16

A larger local checkpoint was evaluated on direct arithmetic controls and a true oracle-sender/model-receiver episode. Direct calls and simulator outcomes are reported separately.

Model revision `530227a7d994db8eca5ab5ced2fb692b614357fd`, GGUF SHA-256 `500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0`. Task manifest SHA-256 `c6302480780c7bea1c54ebc00436c06d57e092b285204139a20311685e3f7d33`. Runtime: llama.cpp b11202, commit fcb3074f2, Windows x64 CUDA 12.4; RTX 4060 Laptop GPU 8 GB; 24 GPU layers requested with CPU offload; context 8192; temperature 0; 256-token server generation cap; API direct calls capped at 128 tokens.

## Direct arithmetic ladder

| Operation | L | Exact | Parseable | Mean input tokens | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| combined_receiver | 2 | 2/8 | 3/8 | 107.6 | 30.0 |
| combined_receiver | 3 | 1/8 | 4/8 | 111.2 | 14.8 |
| combined_receiver | 4 | 4/8 | 8/8 | 115.9 | 20.8 |
| local_prefix | 2 | 8/8 | 8/8 | 86.6 | 8.8 |
| local_prefix | 3 | 8/8 | 8/8 | 90.1 | 12.9 |
| local_prefix | 4 | 8/8 | 8/8 | 94.5 | 17.9 |
| offset_vector | 2 | 8/8 | 8/8 | 88.8 | 9.8 |
| offset_vector | 3 | 8/8 | 8/8 | 93.0 | 15.2 |
| offset_vector | 4 | 8/8 | 8/8 | 98.2 | 20.8 |

## Real simulator receiver condition

| L | Correct message received before submit | Exact given receive | Joint exact |
|---|---:|---:|---:|
| 2 | 8/8 | 1/8 | 1/8 |
| 3 | 8/8 | 1/8 | 1/8 |
| 4 | 8/8 | 0/8 | 0/8 |

## Paired descriptive 8B references

| L | 8B direct combined exact | 8B simulator receive | 8B exact given receive |
|---|---:|---:|---:|
| 2 | 4/8 | 8/8 | 0/8 |
| 3 | 2/8 | 6/8 | 0/6 |
| 4 | 1/8 | 1/8 | 0/1 |

## Interpretation

Interpret only the direct operation ladder and real simulator receiver path separately. The preregistered eligibility gate was not met: 2/24 exact simulator receiver outputs, below the 20/24 threshold. Do not start the narrow message-format comparison under this task, receiver, runtime, and interface. Any model-to-model differences remain confounded by checkpoint, scale, quantization, and CPU/GPU offload.

This is one held-out case family reused from earlier pilots, one greedy sample per case, and a local mixed-offload runtime. No communication-efficiency or general-language claim follows.

Per-episode prompts, exact arrays, messages, answers, and costs: [JSONL](data/PREFIXSUM_LOCAL_14B_RECEIVER_V0_16_RUNS.jsonl).
