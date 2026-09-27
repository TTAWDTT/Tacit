# PrefixSum v0.15 receiver output-contract diagnostic

Paired Qwen3-8B direct calls with the same task, inputs, successful receive transcript, and arithmetic prompt. Only the final answer serialization instruction changes between direct JSON and XML `submit_result`.

Task manifest SHA-256: `c6302480780c7bea1c54ebc00436c06d57e092b285204139a20311685e3f7d33`. Model: Qwen3-8B Q4_K_M GGUF. Runtime: llama.cpp b11202, commit fcb3074f2, Windows x64 CUDA 12.4; RTX 4060 8 GB; 99 GPU layers; context 8192; temperature 0; max 128 generated tokens.

Each length-condition cell contains 8 tasks. Contract validity and mathematical exactness are separate outcomes.

| Output contract | L | Valid contract | Exact | Mean input tokens | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| direct_json_array | 2 | 8/8 | 2/8 | 212.6 | 10.2 |
| direct_json_array | 3 | 7/8 | 3/8 | 216.4 | 16.4 |
| direct_json_array | 4 | 8/8 | 4/8 | 221.2 | 19.9 |
| xml_submit_result | 2 | 7/8 | 4/8 | 224.6 | 41.1 |
| xml_submit_result | 3 | 7/8 | 3/8 | 228.4 | 46.1 |
| xml_submit_result | 4 | 8/8 | 5/8 | 233.2 | 50.8 |

Paired exact-output transitions over all 24 tasks:

- Both exact: 6/24
- JSON only: 3/24
- XML only: 6/24
- Neither exact: 9/24

## Interpretation

Direct JSON was exact on 9/24 tasks; XML submit_result was exact on 12/24. The paired transitions were 6 XML-only versus 3 JSON-only, with 6 exact in both and 9 in neither. XML was descriptively ahead by three tasks, contrary to the preregistered direction, but 24 reused cases do not support a significance claim or stable format ranking.

Contract validity was also similar (23/24 JSON, 22/24 XML). XML used about 12 more mean input tokens and 30.5 more mean output tokens per episode than direct JSON in these calls. Thus the observed small accuracy difference came with greater generation cost; the sample does not establish a quality/cost frontier.

These results show that the XML wrapper itself does not explain the v0.14 zero-success result: under the matched concise arithmetic context, both contracts yielded exact outputs, with XML slightly higher. Prompt and interaction context still matter, and the comparison concerns final-answer serialization rather than inter-agent message representation.

The successful receive transcript is synthetic context, not a model tool call or actual simulator communication. This is a serialization diagnostic, not a message-format or language comparison.

Per-episode prompts, answers, raw responses, and costs: [JSONL](data/PREFIXSUM_OUTPUT_CONTRACT_V0_15_RUNS.jsonl).
