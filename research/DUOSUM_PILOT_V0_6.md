# DuoSum v0.6 cross-model held-out replication

**Status:** 8 held-out episodes, one greedy run per condition and episode, Qwen3-4B Q4_K_M on llama.cpp. This is exploratory; it is not a confirmatory superiority comparison.

## Pooled outcomes

Each episode has two agent submissions. *Strict success* is the Silo evaluator's exact integer check. *Semantic exactness* is a separate post-hoc diagnostic that accepts either the exact integer or a mathematically verified string `a + b = c`; it does not overwrite the benchmark's strict score.

| Condition | Strict agent success | Semantic exactness | Assigned-grammar syntax | Grammar-decoded value fidelity | Bare-decimal value matches | Integer-only submissions | Mean total model tokens | Mean payload bytes | Mean messages | Mean wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scaffold_only` | 0.438 | 0.438 | n/a | n/a | n/a | 0.500 | 7741.4 | 139.5 | 3.12 | 7.06 |
| `concise_nl` | 0.875 | 0.875 | 0/16 | 0/16 | 16/16 | 0.875 | 6211.4 | 6.0 | 2.00 | 5.16 |
| `compact_kv` | 0.938 | 0.938 | 16/16 | 16/16 | n/a | 0.938 | 6190.2 | 10.0 | 2.00 | 5.15 |
| `json_schema` | 1.000 | 1.000 | 0/16 | 0/16 | n/a | 1.000 | 6611.4 | 20.0 | 2.00 | 5.38 |
| `binary` | 0.188 | 0.188 | 15/16 | 8/16 | 4/16 | 0.562 | 6440.6 | 16.8 | 2.00 | 5.36 |
| `autoform` | 0.688 | 0.688 | n/a | n/a | 4/4 | 0.688 | 8399.6 | 210.9 | 3.38 | 7.43 |
| `no_communication` | 0.000 | 0.000 | n/a | n/a | n/a | 0.000 | 7453.8 | 0.0 | 0.00 | 7.42 |

## Per-width results

Rates are over four agent submissions from two episodes at each width. Payload bytes count only UTF-8 message content, excluding simulator JSON and repeated prompt context.

| Input width | Condition | Strict success | Semantic exactness | Total model tokens | Payload bytes | Messages |
|---:|---|---:|---:|---:|---:|---:|
| 4 | `scaffold_only` | 0.50 | 0.50 | 8548 | 185 | 4.0 |
| 4 | `concise_nl` | 1.00 | 1.00 | 6175 | 2 | 2.0 |
| 4 | `compact_kv` | 1.00 | 1.00 | 6152 | 6 | 2.0 |
| 4 | `json_schema` | 1.00 | 1.00 | 7026 | 16 | 2.0 |
| 4 | `binary` | 0.50 | 0.50 | 6373 | 4 | 2.0 |
| 4 | `autoform` | 0.25 | 0.25 | 8946 | 263 | 3.5 |
| 4 | `no_communication` | 0.00 | 0.00 | 7429 | 0 | 0.0 |
| 8 | `scaffold_only` | 0.25 | 0.25 | 8478 | 193 | 3.5 |
| 8 | `concise_nl` | 0.75 | 0.75 | 6199 | 4 | 2.0 |
| 8 | `compact_kv` | 1.00 | 1.00 | 6178 | 8 | 2.0 |
| 8 | `json_schema` | 1.00 | 1.00 | 6450 | 18 | 2.0 |
| 8 | `binary` | 0.25 | 0.25 | 6406 | 11 | 2.0 |
| 8 | `autoform` | 0.75 | 0.75 | 8437 | 241 | 3.5 |
| 8 | `no_communication` | 0.00 | 0.00 | 7436 | 0 | 0.0 |
| 12 | `scaffold_only` | 0.50 | 0.50 | 7218 | 103 | 2.5 |
| 12 | `concise_nl` | 1.00 | 1.00 | 6224 | 8 | 2.0 |
| 12 | `compact_kv` | 1.00 | 1.00 | 6204 | 12 | 2.0 |
| 12 | `json_schema` | 1.00 | 1.00 | 6474 | 22 | 2.0 |
| 12 | `binary` | 0.00 | 0.00 | 6462 | 19 | 2.0 |
| 12 | `autoform` | 1.00 | 1.00 | 8459 | 230 | 3.5 |
| 12 | `no_communication` | 0.00 | 0.00 | 7464 | 0 | 0.0 |
| 16 | `scaffold_only` | 0.50 | 0.50 | 6722 | 77 | 2.5 |
| 16 | `concise_nl` | 0.75 | 0.75 | 6247 | 10 | 2.0 |
| 16 | `compact_kv` | 0.75 | 0.75 | 6227 | 14 | 2.0 |
| 16 | `json_schema` | 1.00 | 1.00 | 6494 | 24 | 2.0 |
| 16 | `binary` | 0.00 | 0.00 | 6522 | 32 | 2.0 |
| 16 | `autoform` | 0.75 | 0.75 | 7757 | 110 | 3.0 |
| 16 | `no_communication` | 0.00 | 0.00 | 7486 | 0 | 0.0 |

## Paired comparison with v0.5

The eight held-out task files and seven conditions are the same as v0.5. Each column counts correct agent outputs out of 16; this is a descriptive paired replication across two model setups, not an isolated model-scale effect.

| Condition | v0.5 strict | v0.6 strict | Difference | v0.5 semantic | v0.6 semantic |
|---|---:|---:|---:|---:|---:|
| `scaffold_only` | 4/16 | 7/16 | +3 | 15/16 | 7/16 |
| `concise_nl` | 3/16 | 14/16 | +11 | 11/16 | 14/16 |
| `compact_kv` | 0/16 | 15/16 | +15 | 10/16 | 15/16 |
| `json_schema` | 3/16 | 16/16 | +13 | 4/16 | 16/16 |
| `binary` | 5/16 | 3/16 | -2 | 13/16 | 3/16 |
| `autoform` | 4/16 | 11/16 | +7 | 12/16 | 11/16 |
| `no_communication` | 1/16 | 0/16 | -1 | 1/16 | 0/16 |

## Interpretation

- Compare communicating conditions with the observed no-communication control; do not assume the intervention worked from its label.
- Assigned-grammar syntax and grammar-decoded value fidelity apply to fixed-format arms. The separate bare-decimal column records whether a decimal-only message equals its sender's private value; it does not count as compliance with a sentence or schema instruction.
- Strict success uses the benchmark's exact integer tool contract. Semantic exactness separately accepts only a verified integer or a simple arithmetic string whose stated operands and result are mutually consistent. It does not change the benchmark score.
- No-communication semantic success was 0/16 agent outputs. Communicating arms ranged from 3/16 to 16/16. The zero control score is consistent with communication being necessary on these cases, but eight episodes are not a proof for the full task family.
- The concise-NL instruction specified a plain-English sentence, but the raw audit shows decimal-only payloads in 16/16 messages. Those numerals matched their senders' private values in 16/16; this is effective decimal shorthand, not adherence to the assigned NL form.
- Compact-KV messages decoded to the sender's value in 16/16 messages, yet semantic task success was 15/16. This separates reliable serialization from downstream reasoning/submission failures.
- The example-assisted binary arm encoded the sender's value correctly in only 8/16 messages; JSON syntax was valid in 0/16 messages. Raw audit found all 16 JSON messages were single-quoted Python-style mappings, so the 16/16 strict task success reflects receiver tolerance of that payload rather than valid-JSON adherence.
- Within this run, concise-NL's actual decimal-only payload averaged 6 bytes and reached 14/16 strict successes; compact-KV averaged 10 bytes and reached 15/16; the invalid-JSON mapping averaged 20 bytes and reached 16/16. This is a small descriptive trade-off, not a matched-budget frontier, and the first and third arms did not follow their assigned grammars.
- AutoForm semantic success was 11/16, compared with 7/16 for the unformatted scaffold. This small run shows no evidence that format self-selection improves task success for this model/task.
- Payload bytes, model tokens, repeated context, tool calls, and end-to-end latency are separate measures. This run does not impose equal-byte or equal-token budgets and cannot define a communication-efficiency frontier.
- Relative to v0.5, strict success increased for scaffold (+3 agents), concise NL (+11), compact KV (+15), JSON (+13), and AutoForm (+7); binary fell by 2 and no-communication fell by 1. This paired pattern is exploratory: model size, quantization, backend, and chat template all change together, so it cannot isolate scale or establish a general condition effect.
- Token totals are reported as returned by each backend. Do not compare v0.5 and v0.6 model-token totals as a cross-model efficiency result because the model tokenizer and inference backend changed.
- The lower bound in `docs/THEORY.md` is in binary wire bits; it is not directly comparable with model tokens or UTF-8 bytes.

## Next revision

Next, evaluate a typed parser/decoder as an explicit system condition against prompt-only formatting, then replicate across a richer task family and additional sender/receiver pairs. Account for parser cost, malformed-message recovery, and wire size. Do not call the invalid JSON arm JSON adherence or compare backend token totals directly across v0.5/v0.6.

Raw prompts, tool traces, and model responses remain in the ignored `.cache/pilot_v0_6/`; this public summary contains aggregate measurements and selected anonymized outcomes only.
