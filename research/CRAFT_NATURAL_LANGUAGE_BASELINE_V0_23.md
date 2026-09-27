# CRAFT v0.23 bounded-history natural-language baseline

This reference uses three difficulty levels and an explicit latest-16-line shared-history window. It is a baseline, not a protocol comparison.

| Structure | Level | Turns | Final progress | Complete | Successful messages / requested | Delivered / successful | Mean wire tokens/turn |
|---|---|---:|---:|---|---:|---:|---:|
| structure_001 | medium | 8 | 0.338 | False | 24/24 | 24/24 | 88.9 |
| structure_008 | simple | 8 | 0.310 | False | 24/24 | 24/24 | 135.4 |
| structure_003 | complex | 8 | 0.245 | False | 24/24 | 24/24 | 96.4 |

Across the three structures, descriptive mean progress was 0.298. The server completed 96 calls: 121155 prompt tokens, 7013 generated tokens, and 247.822 seconds of summed service time.
Mean prompt/generation throughput was 1810.8/38.13 tokens/s.

## Scope

This is a local capability and traffic reference under CRAFT's natural-language prompts and bounded shared history. It does not establish superiority, generalization, or an efficiency frontier. Future protocol conditions must reuse these tasks, model/runtime, history cap, scheduler, parser, and Builder oracle.

Per-turn public messages, delivery counts, transcript token counts, and inference timing records are in the companion JSON. Private reasoning and prompts are excluded.
