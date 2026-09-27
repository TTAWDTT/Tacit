# DuoSum v0.4 held-out calibration

**Status:** four held-out episodes, one greedy run per condition and width. This is an exploratory held-out check, not a confirmatory comparison.

## Pooled outcomes

Each episode has two agent submissions. *Strict success* is the Silo evaluator's exact integer check. *Semantic exactness* is a separate post-hoc diagnostic that accepts either the exact integer or a mathematically verified string `a + b = c`; it does not overwrite the benchmark's strict score.

| Condition | Strict agent success | Semantic exactness | Format adherence | Integer-only submissions | Mean total model tokens | Mean payload bytes | Mean messages | Mean wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scaffold_only` | 0.000 | 0.875 | n/a | 0.000 | 6082.0 | 46.5 | 2.00 | 13.19 |
| `concise_nl` | 0.000 | 0.500 | 9/13 | 0.000 | 7778.8 | 75.5 | 3.25 | 14.92 |
| `compact_kv` | 0.000 | 0.625 | 9/10 | 0.125 | 6655.0 | 13.5 | 2.50 | 14.34 |
| `json_schema` | 0.000 | 0.625 | 0/8 | 0.000 | 6319.2 | 20.5 | 2.00 | 13.11 |
| `binary` | 0.125 | 0.750 | 0/8 | 0.250 | 6296.2 | 6.5 | 2.00 | 13.90 |
| `no_communication` | 0.000 | 0.000 | n/a | 0.750 | 7395.5 | 0.0 | 0.00 | 31.10 |

## Per-width results

Rates are over the two agent submissions in the single episode at that width. Payload bytes count only UTF-8 message content, excluding simulator JSON and repeated prompt context.

| Input width | Condition | Strict success | Semantic exactness | Total model tokens | Payload bytes | Messages |
|---:|---|---:|---:|---:|---:|---:|
| 4 | `scaffold_only` | 0.00 | 1.00 | 6056 | 44 | 2.0 |
| 4 | `concise_nl` | 0.00 | 0.50 | 7444 | 66 | 3.0 |
| 4 | `compact_kv` | 0.00 | 0.50 | 5380 | 8 | 2.0 |
| 4 | `json_schema` | 0.00 | 1.00 | 6298 | 18 | 2.0 |
| 4 | `binary` | 0.00 | 1.00 | 6256 | 4 | 2.0 |
| 4 | `no_communication` | 0.00 | 0.00 | 6954 | 0 | 0.0 |
| 8 | `scaffold_only` | 0.00 | 1.00 | 6076 | 46 | 2.0 |
| 8 | `concise_nl` | 0.00 | 1.00 | 7484 | 69 | 3.0 |
| 8 | `compact_kv` | 0.00 | 1.00 | 6310 | 10 | 2.0 |
| 8 | `json_schema` | 0.00 | 1.00 | 6318 | 20 | 2.0 |
| 8 | `binary` | 0.00 | 0.50 | 6268 | 6 | 2.0 |
| 8 | `no_communication` | 0.00 | 0.00 | 7309 | 0 | 0.0 |
| 12 | `scaffold_only` | 0.00 | 0.50 | 6087 | 47 | 2.0 |
| 12 | `concise_nl` | 0.00 | 0.00 | 8686 | 94 | 4.0 |
| 12 | `compact_kv` | 0.00 | 0.50 | 7513 | 17 | 3.0 |
| 12 | `json_schema` | 0.00 | 0.50 | 6325 | 21 | 2.0 |
| 12 | `binary` | 0.00 | 0.50 | 6291 | 7 | 2.0 |
| 12 | `no_communication` | 0.00 | 0.00 | 7455 | 0 | 0.0 |
| 16 | `scaffold_only` | 0.00 | 1.00 | 6109 | 49 | 2.0 |
| 16 | `concise_nl` | 0.00 | 0.50 | 7501 | 73 | 3.0 |
| 16 | `compact_kv` | 0.00 | 0.50 | 7417 | 19 | 3.0 |
| 16 | `json_schema` | 0.00 | 0.00 | 6336 | 23 | 2.0 |
| 16 | `binary` | 0.50 | 1.00 | 6370 | 9 | 2.0 |
| 16 | `no_communication` | 0.00 | 0.00 | 7864 | 0 | 0.0 |

## Interpretation

- The no-communication condition is evaluated on all four episodes; compare its observed strict and semantic rates with each communicating condition rather than assuming the intervention worked.
- Format adherence is the number of transmitted messages matching the assigned representation exactly. It is not applicable to the unformatted scaffold and no-communication controls.
- Strict success uses the benchmark's exact integer tool contract. Semantic exactness separately accepts only a verified integer or a simple arithmetic string whose stated operands and result are mutually consistent. It does not change the benchmark score.
- The shared bare-integer instruction did not resolve terminal answer formatting: strict success was 0/8 for scaffold, concise-NL, compact-KV, JSON, and no-communication, and 1/8 for binary. This is an observed interface/model failure, not evidence that the communication messages themselves caused the score gap.
- Semantic exactness was 7/8 for scaffold, 4/8 concise-NL, 5/8 compact-KV, 5/8 JSON, 6/8 binary, and 0/8 no-communication. The one-run-per-cell sample is too small to rank formats; it does show that communication was necessary in these four episodes and that the exact-sum task can reveal information transfer.
- The binary condition transmitted decimal strings in all four episodes (0/8 messages matched base-2 encoding); the JSON condition used single-quoted Python-dict strings (0/8 valid JSON messages). Their outcomes and payload lengths cannot be attributed to successful binary or JSON encoding. Compact key-value exactly matched its format/value in 9/10 messages; concise-NL did so in 9/13. Some repeated messages carried the other agent's value, so syntax adherence alone is insufficient: semantic fidelity must also be tracked.
- The binary arm's 6.5-byte mean payload is therefore a misleading format label: it reflects short decimal messages, not binary coding. Format adherence is a prerequisite for interpreting a representation comparison.
- Payload bytes, model tokens, repeated context, tool calls, and end-to-end latency are separate measures. This run does not impose equal-byte or equal-token budgets and cannot define a communication-efficiency frontier.
- The four episodes provide only one observation per input width. They are insufficient to estimate a scaling law or stable condition effect.
- The lower bound in `docs/THEORY.md` is in binary wire bits; it is not directly comparable with model tokens or UTF-8 bytes.

## Next revision

Before scaling samples or sweeping budgets, separate two effects: (1) message decoding and answer inference, and (2) final answer serialization. Freeze a format-neutral, deterministic answer normalizer/evaluator that accepts an integer or a strictly verified arithmetic expression, report its score alongside Silo's strict score, and preserve the submitted raw answer. Then repeat paired held-out episodes across more seeds and model sizes. Only after reliable task success should equal-byte/equal-token sweeps estimate a communication-efficiency frontier.

Raw prompts, tool traces, and model responses remain in the ignored `.cache/pilot_v0_4/`; this public summary contains aggregate measurements and selected anonymized outcomes only.
