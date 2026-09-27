# DuoSum v0.5 held-out comparison

**Status:** 8 held-out episodes, one greedy run per condition and episode, one local model. This is exploratory; it is not a confirmatory superiority comparison.

## Pooled outcomes

Each episode has two agent submissions. *Strict success* is the Silo evaluator's exact integer check. *Semantic exactness* is a separate post-hoc diagnostic that accepts either the exact integer or a mathematically verified string `a + b = c`; it does not overwrite the benchmark's strict score.

| Condition | Strict agent success | Semantic exactness | Syntax-valid messages | Sender-value fidelity | Integer-only submissions | Mean total model tokens | Mean payload bytes | Mean messages | Mean wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scaffold_only` | 0.250 | 0.938 | n/a | n/a | 0.250 | 6264.5 | 48.8 | 2.12 | 13.38 |
| `concise_nl` | 0.188 | 0.688 | 23/23 | 21/23 | 0.188 | 7314.9 | 65.1 | 2.88 | 14.54 |
| `compact_kv` | 0.000 | 0.625 | 17/17 | 17/17 | 0.312 | 5995.9 | 10.5 | 2.12 | 14.38 |
| `json_schema` | 0.188 | 0.250 | 0/24 | 0/24 | 0.250 | 7710.4 | 28.1 | 3.00 | 14.74 |
| `binary` | 0.312 | 0.812 | 4/17 | 4/17 | 0.438 | 6564.2 | 6.1 | 2.12 | 12.55 |
| `autoform` | 0.250 | 0.750 | n/a | n/a | 0.438 | 6193.2 | 51.8 | 2.25 | 15.12 |
| `no_communication` | 0.062 | 0.062 | n/a | n/a | 0.750 | 6547.1 | 0.0 | 0.00 | 29.15 |

## Per-width results

Rates are over four agent submissions from two episodes at each width. Payload bytes count only UTF-8 message content, excluding simulator JSON and repeated prompt context.

| Input width | Condition | Strict success | Semantic exactness | Total model tokens | Payload bytes | Messages |
|---:|---|---:|---:|---:|---:|---:|
| 4 | `scaffold_only` | 1.00 | 1.00 | 6029 | 42 | 2.0 |
| 4 | `concise_nl` | 0.75 | 0.75 | 8029 | 74 | 3.5 |
| 4 | `compact_kv` | 0.00 | 0.00 | 4866 | 6 | 2.0 |
| 4 | `json_schema` | 0.50 | 0.50 | 7720 | 26 | 3.0 |
| 4 | `binary` | 0.50 | 0.50 | 7022 | 2 | 2.5 |
| 4 | `autoform` | 1.00 | 1.00 | 5903 | 42 | 2.0 |
| 4 | `no_communication` | 0.25 | 0.25 | 5965 | 0 | 0.0 |
| 8 | `scaffold_only` | 0.00 | 0.75 | 6650 | 56 | 2.5 |
| 8 | `concise_nl` | 0.00 | 0.25 | 8052 | 78 | 3.5 |
| 8 | `compact_kv` | 0.00 | 0.50 | 6554 | 10 | 2.5 |
| 8 | `json_schema` | 0.00 | 0.00 | 7688 | 28 | 3.0 |
| 8 | `binary` | 0.50 | 1.00 | 6382 | 4 | 2.0 |
| 8 | `autoform` | 0.00 | 0.25 | 6076 | 56 | 2.5 |
| 8 | `no_communication` | 0.00 | 0.00 | 7184 | 0 | 0.0 |
| 12 | `scaffold_only` | 0.00 | 1.00 | 6148 | 48 | 2.0 |
| 12 | `concise_nl` | 0.00 | 0.75 | 6880 | 60 | 2.5 |
| 12 | `compact_kv` | 0.00 | 1.00 | 6328 | 12 | 2.0 |
| 12 | `json_schema` | 0.25 | 0.50 | 7090 | 25 | 2.5 |
| 12 | `binary` | 0.25 | 1.00 | 6415 | 8 | 2.0 |
| 12 | `autoform` | 0.00 | 0.75 | 6222 | 60 | 2.5 |
| 12 | `no_communication` | 0.00 | 0.00 | 5562 | 0 | 0.0 |
| 16 | `scaffold_only` | 0.00 | 1.00 | 6230 | 50 | 2.0 |
| 16 | `concise_nl` | 0.00 | 1.00 | 6298 | 50 | 2.0 |
| 16 | `compact_kv` | 0.00 | 1.00 | 6236 | 14 | 2.0 |
| 16 | `json_schema` | 0.00 | 0.00 | 8344 | 34 | 3.5 |
| 16 | `binary` | 0.00 | 0.75 | 6437 | 10 | 2.0 |
| 16 | `autoform` | 0.00 | 1.00 | 6572 | 50 | 2.0 |
| 16 | `no_communication` | 0.00 | 0.00 | 7478 | 0 | 0.0 |

## Interpretation

- Compare communicating conditions with the observed no-communication control; do not assume the intervention worked from its label.
- Syntax validity and sender-value fidelity are separate: the first asks whether a fixed-format message parses; the second asks whether it decodes to that sender's private value. They are not applicable to the unformatted scaffold, adaptive AutoForm, and no-communication controls.
- Strict success uses the benchmark's exact integer tool contract. Semantic exactness separately accepts only a verified integer or a simple arithmetic string whose stated operands and result are mutually consistent. It does not change the benchmark score.
- No-communication semantic success was 1/16 agent outputs. Communicating arms ranged from 4/16 to 15/16; the control failed on most cases but one local guess was correct, so it is not a logical proof by itself.
- Compact-KV messages decoded to the sender's value in 17/17 messages, yet semantic task success was 10/16. This separates reliable serialization from downstream reasoning/submission failures.
- The example-assisted binary arm encoded the sender's value correctly in only 4/17 messages; JSON syntax was valid in 0/24 messages. Outcomes from those arms therefore mix intended-format use with fallback/nonconforming messages.
- AutoForm semantic success was 12/16, compared with 15/16 for the unformatted scaffold. This small run shows no evidence that format self-selection improves task success for this model/task.
- Payload bytes, model tokens, repeated context, tool calls, and end-to-end latency are separate measures. This run does not impose equal-byte or equal-token budgets and cannot define a communication-efficiency frontier.
- This single-model, one-run-per-cell study is insufficient to estimate cross-model transfer, a scaling law, or a robust condition effect.
- The lower bound in `docs/THEORY.md` is in binary wire bits; it is not directly comparable with model tokens or UTF-8 bytes.

## Next revision

Interpret outcomes jointly with format adherence. If fixed formats are followed and communication improves semantic task success, expand to a second receiver model and richer task family before matched-budget sweeps. If adherence remains poor, treat instruction-following as a bottleneck and avoid attributing outcomes to the intended representation.

Raw prompts, tool traces, and model responses remain in the ignored `.cache/pilot_v0_5/`; this public summary contains aggregate measurements and selected anonymized outcomes only.
