# DuoSum v0.3 calibration pilot

**Status:** four calibration episodes, one greedy run per condition and width. This is a feasibility study, not a confirmatory comparison.

## Pooled outcomes

Each episode has two agent submissions. *Strict success* is the Silo evaluator's exact integer check. *Semantic exactness* is a separate post-hoc diagnostic that accepts either the exact integer or a mathematically verified string `a + b = c`; it does not overwrite the benchmark's strict score.

| Condition | Strict agent success | Semantic exactness | Integer-only submissions | Mean total model tokens | Mean message payload bytes | Mean messages | Mean wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scaffold_only` | 0.250 | 0.875 | 0.250 | 6358.5 | 52.0 | 2.25 | 13.26 |
| `concise_nl` | 0.125 | 0.750 | 0.125 | 7021.0 | 63.0 | 2.75 | 13.55 |
| `compact_kv` | 0.125 | 0.750 | 0.125 | 6707.0 | 12.5 | 2.50 | 12.54 |
| `json_schema` | 0.250 | 1.000 | 0.250 | 6187.2 | 20.2 | 2.00 | 11.72 |
| `no_communication` | 0.000 | 0.000 | 0.875 | 5803.2 | 0.0 | 0.00 | 23.55 |

## Per-width results

Rates are over the two agent submissions in the single episode at that width. Payload bytes count only UTF-8 message content, excluding simulator JSON and repeated prompt context.

| Input width | Condition | Strict success | Semantic exactness | Total model tokens | Payload bytes | Messages |
|---:|---|---:|---:|---:|---:|---:|
| 4 | `scaffold_only` | 1.00 | 1.00 | 6172 | 43 | 2.0 |
| 4 | `concise_nl` | 0.50 | 1.00 | 7289 | 65 | 3.0 |
| 4 | `compact_kv` | 0.50 | 0.50 | 7308 | 11 | 3.0 |
| 4 | `json_schema` | 1.00 | 1.00 | 6145 | 17 | 2.0 |
| 4 | `no_communication` | 0.00 | 0.00 | 6950 | 0 | 0.0 |
| 8 | `scaffold_only` | 0.00 | 0.50 | 7207 | 68 | 3.0 |
| 8 | `concise_nl` | 0.00 | 0.00 | 8488 | 90 | 4.0 |
| 8 | `compact_kv` | 0.00 | 0.50 | 7335 | 14 | 3.0 |
| 8 | `json_schema` | 0.00 | 1.00 | 6177 | 19 | 2.0 |
| 8 | `no_communication` | 0.00 | 0.00 | 4791 | 0 | 0.0 |
| 12 | `scaffold_only` | 0.00 | 1.00 | 5955 | 47 | 2.0 |
| 12 | `concise_nl` | 0.00 | 1.00 | 6137 | 47 | 2.0 |
| 12 | `compact_kv` | 0.00 | 1.00 | 6077 | 11 | 2.0 |
| 12 | `json_schema` | 0.00 | 1.00 | 6199 | 21 | 2.0 |
| 12 | `no_communication` | 0.00 | 0.00 | 6305 | 0 | 0.0 |
| 16 | `scaffold_only` | 0.00 | 1.00 | 6100 | 50 | 2.0 |
| 16 | `concise_nl` | 0.00 | 1.00 | 6170 | 50 | 2.0 |
| 16 | `compact_kv` | 0.00 | 1.00 | 6108 | 14 | 2.0 |
| 16 | `json_schema` | 0.00 | 1.00 | 6228 | 24 | 2.0 |
| 16 | `no_communication` | 0.00 | 0.00 | 5167 | 0 | 0.0 |

## Interpretation

- The no-communication arm had zero strict and semantic correct submissions across the four calibration episodes. The task construction therefore passed its communication-necessity check for this model and split.
- All JSON-arm submissions were semantically exact under the post-hoc arithmetic parser, while most were rejected by the strict integer tool interface as expression strings. This is a receiver/output-contract failure, not missing information. The parser is diagnostic only; changing the evaluator would require a new benchmark version.
- Compact key-value messages were shortest by UTF-8 payload bytes, but their total model-token cost was not the lowest. Repeated context, format-instruction tokens, extra tool calls, and terminal answer failures mean payload size alone is not an efficiency frontier.
- At the 4-bit calibration episode, JSON and the scaffold-only arm both achieved strict success 1.0; at larger widths, agents generally computed the right arithmetic but emitted expressions instead of an integer. This single seed per width cannot establish a width trend.
- No bandwidth cap was swept, and model-token counts are not information bits. The communication-complexity bound in `docs/THEORY.md` is a wire-bit reference, not a token target.

## Next revision

Add a common, format-neutral final-answer contract requiring a bare integer in `submit_result`, then rerun held-out seeds. Keep strict tool success and semantic exactness as separate outcomes. Only after that calibration should message-byte/token budgets be swept and the Pareto frontier compared.

Raw prompts, tool traces, and model responses remain in the ignored `.cache/pilot_v0_3/`; this public summary contains aggregate measurements and selected anonymized outcomes only.
