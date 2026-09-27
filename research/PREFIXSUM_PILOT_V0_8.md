# PrefixSum v0.8 role-explicit protocol comparison

**Status:** 12 fresh seeded episodes, one greedy run per condition and episode, Qwen3-4B Q4_K_M on llama.cpp with a lossless message-content adapter. This is exploratory, not a confirmatory ranking.

Strict agent success is exact equality of the returned integer list to that agent's pinned prefix-sum segment. Episode success requires both agents' lists to be exact. Message syntax and sender-value fidelity are separately audited from raw traces.

## Pooled results

| Condition | Strict agent success | Fully correct episodes | A0→A1 send attempts | Delivered messages | A1 received payload before submit | A1 send attempts | Message syntax | Sender-value fidelity | Mean payload bytes | Mean backend tokens | Mean wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scaffold_only` | 2/24 | 0/12 | 12 | 12 | 12/12 | 0 | n/a | n/a | 34.4 | 7500 | 7.54 |
| `concise_nl` | 4/24 | 0/12 | 12 | 12 | 8/12 | 0 | 12/12 | 0/12 | 20.2 | 7794 | 7.71 |
| `compact_kv` | 3/24 | 0/12 | 12 | 12 | 11/12 | 0 | 12/12 | 0/12 | 5.0 | 7784 | 7.53 |
| `json_schema` | 4/24 | 0/12 | 12 | 12 | 10/12 | 0 | 11/12 | 0/12 | 11.1 | 7797 | 7.43 |
| `binary` | 2/24 | 0/12 | 11 | 11 | 11/12 | 0 | 11/11 | 0/11 | 15.8 | 8009 | 8.19 |
| `full_shard` | 3/24 | 0/12 | 12 | 12 | 12/12 | 0 | 12/12 | 12/12 | 64.2 | 7796 | 7.63 |
| `no_communication` | 2/24 | 0/12 | 26 | 0 | 0/12 | 0 | n/a | n/a | 0.0 | 8413 | 7.65 |

## Results by segment length

| Segment length | Condition | Strict agent success | Fully correct episodes | Mean payload bytes | Mean backend tokens |
|---:|---|---:|---:|---:|---:|
| 6 | `scaffold_only` | 1/8 | 0/4 | 8.2 | 7065 |
| 6 | `concise_nl` | 3/8 | 0/4 | 20.0 | 7377 |
| 6 | `compact_kv` | 2/8 | 0/4 | 5.0 | 7258 |
| 6 | `json_schema` | 3/8 | 0/4 | 13.2 | 7430 |
| 6 | `binary` | 2/8 | 0/4 | 8.5 | 6819 |
| 6 | `full_shard` | 2/8 | 0/4 | 22.8 | 7318 |
| 6 | `no_communication` | 1/8 | 0/4 | 0.0 | 7690 |
| 15 | `scaffold_only` | 1/8 | 0/4 | 16.5 | 7381 |
| 15 | `concise_nl` | 1/8 | 0/4 | 20.0 | 7758 |
| 15 | `compact_kv` | 1/8 | 0/4 | 5.0 | 7544 |
| 15 | `json_schema` | 1/8 | 0/4 | 10.0 | 7696 |
| 15 | `binary` | 0/8 | 0/4 | 15.0 | 7745 |
| 15 | `full_shard` | 1/8 | 0/4 | 56.0 | 7702 |
| 15 | `no_communication` | 1/8 | 0/4 | 0.0 | 8406 |
| 30 | `scaffold_only` | 0/8 | 0/4 | 78.5 | 8053 |
| 30 | `concise_nl` | 0/8 | 0/4 | 20.5 | 8249 |
| 30 | `compact_kv` | 0/8 | 0/4 | 5.0 | 8551 |
| 30 | `json_schema` | 0/8 | 0/4 | 10.0 | 8264 |
| 30 | `binary` | 0/8 | 0/4 | 24.0 | 9462 |
| 30 | `full_shard` | 0/8 | 0/4 | 114.0 | 8370 |
| 30 | `no_communication` | 0/8 | 0/4 | 0.0 | 9142 |

## Deterministic engine and scorer control (not an LLM result)

A deterministic two-agent oracle completed 12/12 episodes, delivered 12/12 messages, and sent the correct subtotal in 12/12 cases. Mean completion was 4.0 rounds. It made zero model-backend calls and used the same pinned task manifest, Silo engine commit, message tools, parser adapter, and scorer. This shows the benchmark mechanics can support exact success; it does not measure an LLM protocol.


## Interpretation

- Compare each communicating arm with the no-communication control. Agent 0 can compute its local prefix segment without receiving Agent 1's values, but Agent 1 cannot recover its global offset from its own segment alone; therefore episode-level unanimity is the task-necessity outcome.
- Send attempts, successful delivery, and an Agent 1 receive call that returned a non-empty payload are distinct. `no_communication` still has send attempts because the intervention rejects the channel; it delivered zero messages. Agent 1's received-payload measure requires a non-empty receive result before its submit call.
- Agent 0 sent a message with the required syntax in most format arms, but subtotal value fidelity was 0/12 for concise English, compact-KV, JSON, and binary. In contrast, full-shard syntax and value fidelity were both 12/12. This localizes a major failure to subtotal calculation or value encoding, before attributing downstream failures to receiver decoding.
- Despite successful full-shard transmission, no episode was fully correct in any condition. The task therefore exposed a second bottleneck: correct receipt did not reliably produce Agent 1's offset-adjusted prefix list. In the audited compact-KV example, Agent 1 received `s=161` (the true sender shard sum was 168) and submitted only its local cumulative sums; the raw context shows the message was present. This is a sender aggregation and receiver execution failure, not transport loss.
- The local adapter preserves the exact raw `send_message.content` string before the pinned Silo tool layer stores it. This avoids the upstream generic XML parser's integer/JSON coercion; it does not rewrite or normalize any model message.
- For each segment length L and values in {1,...,50}, the exact subtotal has 49L+1 possible values. Any zero-error fixed-length encoding therefore needs at least ceil(log2(49L+1)) bits: 9, 10, and 11 bits at lengths 6, 15, and 30. This bound concerns the subtotal wire code, not model tokens, prompt cost, or the computation of the prefix arrays.
- Compact-KV payloads averaged 5 bytes versus 64.2 bytes for full-shard payloads, but compact totals were never faithful in this run and did not yield episode success. This is only a wire-size comparison, not evidence of lower total cost or an efficiency-frontier gain.
- One episode-condition run and twelve seeds provide exploratory evidence only. The two agent outputs in an episode are paired observations, not independent trials. No total-token budget is enforced and no efficiency frontier or scaling law is claimed.
- Token counts are backend-reported. They are descriptive within this fixed setup and should not be compared directly across different model tokenizers or runtimes.

## Next revision

The deterministic oracle has now verified the task, message transport, and scorer end to end. The next diagnostic is a paired hybrid: oracle Agent 0 with an LLM receiver isolates offset decoding/application; LLM Agent 0 with an oracle receiver isolates sender aggregation and message formation. This can localize the model-side failure before a new task or protocol is proposed. Continue to treat the current run as no protocol ranking.

Raw traces remain in ignored `.cache/pilot_v0_8/`; task files and checksums are public in `benchmarks/prefixsum_v0_2/tasks/`.
