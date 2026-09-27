# PrefixSum v0.14 agent-scaffold context diagnostic

Paired direct receiver calls with identical data, receive transcript, user task, model, and backend. The system prompt changes from a concise receiver instruction to the pinned Silo multi-agent tool scaffold, which also supplies a full XML tool schema and example.

Task manifest SHA-256: `c6302480780c7bea1c54ebc00436c06d57e092b285204139a20311685e3f7d33`. Model: Qwen3-8B Q4_K_M GGUF. Runtime: llama.cpp b11202, commit fcb3074f2, Windows x64 CUDA 12.4; RTX 4060 8 GB; 99 GPU layers; context 8192; temperature 0; max 128 generated tokens.

Each length-condition cell has 8 cases. Exactness requires one valid XML `submit_result` tool call containing a strict JSON integer array equal to the expected segment.

| System condition | L | Valid submit | Exact | Extra receive call | Mean input tokens |
|---|---:|---:|---:|---:|---:|
| concise_receiver_system | 2 | 3/8 | 0/8 | 0/8 | 235.6 |
| concise_receiver_system | 3 | 1/8 | 0/8 | 0/8 | 239.4 |
| concise_receiver_system | 4 | 0/8 | 0/8 | 0/8 | 244.2 |
| verbose_silo_msg_system | 2 | 7/8 | 0/8 | 0/8 | 1029.6 |
| verbose_silo_msg_system | 3 | 8/8 | 0/8 | 0/8 | 1033.4 |
| verbose_silo_msg_system | 4 | 8/8 | 0/8 | 0/8 | 1038.2 |

Paired exact-output outcomes across all 24 tasks:

- Both exact: 0/24
- Concise only: 0/24
- Verbose only: 0/24
- Neither exact: 24/24

## Interpretation

Exact receiver output remained 0/24 in both conditions. However, the verbose scaffold produced a syntactically valid `submit_result` call in 23/24 cases, versus 4/24 with the concise system prompt. The verbose arm therefore improved tool-call form in this small run, while failing to improve semantic task success.

Mean input was 239.7 tokens in the concise arm and 1033.7 in the verbose arm (4.3×). The difference includes tool-schema/example context as well as system length. This is a concrete syntax-versus-semantics and context-cost trade-off, not an efficiency-frontier result.

With 24 reused tasks, treat paired counts descriptively. The study tests scaffold compatibility and tool-schema explicitness together; it does not isolate raw token length as the cause.

The successful receive transcript is synthetic context and is not counted as model tool use or an actual simulator delivery. No communication format is compared.

All prompts, raw outputs, parsed tool calls, expected arrays, and costs: [JSONL](data/PREFIXSUM_SCAFFOLD_CONTEXT_V0_14_RUNS.jsonl).
