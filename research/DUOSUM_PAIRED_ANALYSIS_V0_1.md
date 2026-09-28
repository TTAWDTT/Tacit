# DuoSum v0.5/v0.6 paired reanalysis

This analysis uses the eight shared task IDs per run and resamples at the task level, keeping the two submissions in an episode together. It is descriptive: eight tasks provide very low precision, and the two model runs differ in size, quantization, backend, and chat template. No format-superiority claim follows.

Rebuild the sanitized episode ledger and this report from the local ignored run traces with `python research/analyze_duosum_paired.py`; the script makes no model requests.

Differences are `compact_kv − comparator`; cost differences below zero favor compact-KV on that measure. `payload_bytes` are message-content UTF-8 bytes as recorded by the existing pilot; `serialized_message_file_bytes` include the simulator's per-message JSON files. Neither includes the full repeated prompt/context, so these are channel-content diagnostics, not total wire-cost frontiers.

Paired percentile bootstrap: 20,000 episode-cluster resamples, seed 1729; intervals are exploratory with n=8.

## pilot_v0_5

| Paired metric | Comparator | Mean difference | 95% percentile interval |
|---|---|---:|---:|
| `strict_agent_rate` | `scaffold_only` | -0.2500 | [-0.6250, 0.0000] |
| `joint_strict_success` | `scaffold_only` | -0.2500 | [-0.6250, 0.0000] |
| `semantic_agent_rate` | `scaffold_only` | -0.3125 | [-0.6250, -0.0625] |
| `joint_semantic_success` | `scaffold_only` | -0.2500 | [-0.6250, 0.0000] |
| `payload_bytes` | `scaffold_only` | -38.2500 | [-42.7500, -36.0000] |
| `serialized_message_file_bytes` | `scaffold_only` | -37.7500 | [-42.2500, -35.2500] |
| `model_tokens` | `scaffold_only` | -268.6250 | [-744.8781, 109.5000] |
| `message_count` | `scaffold_only` | 0.0000 | [0.0000, 0.0000] |
| `elapsed_seconds` | `scaffold_only` | 0.9925 | [-0.3262, 2.3575] |
| `strict_agent_rate` | `concise_nl` | -0.1875 | [-0.4375, 0.0000] |
| `joint_strict_success` | `concise_nl` | -0.1250 | [-0.3750, 0.0000] |
| `semantic_agent_rate` | `concise_nl` | -0.0625 | [-0.3750, 0.2500] |
| `joint_semantic_success` | `concise_nl` | 0.1250 | [-0.2500, 0.5000] |
| `payload_bytes` | `concise_nl` | -54.6250 | [-65.5000, -43.8750] |
| `serialized_message_file_bytes` | `concise_nl` | -124.2500 | [-180.6250, -68.5000] |
| `model_tokens` | `concise_nl` | -1319.0000 | [-2217.6250, -515.3750] |
| `message_count` | `concise_nl` | -0.7500 | [-1.2500, -0.2500] |
| `elapsed_seconds` | `concise_nl` | -0.1653 | [-0.9654, 0.6464] |
| `strict_agent_rate` | `json_schema` | -0.1875 | [-0.4375, 0.0000] |
| `joint_strict_success` | `json_schema` | -0.1250 | [-0.3750, 0.0000] |
| `semantic_agent_rate` | `json_schema` | 0.3750 | [-0.1250, 0.8750] |
| `joint_semantic_success` | `json_schema` | 0.3750 | [-0.1250, 0.8750] |
| `payload_bytes` | `json_schema` | -17.6250 | [-21.7500, -13.7500] |
| `serialized_message_file_bytes` | `json_schema` | -99.0000 | [-150.0000, -48.2500] |
| `model_tokens` | `json_schema` | -1714.5000 | [-2408.1281, -1059.8750] |
| `message_count` | `json_schema` | -0.8750 | [-1.3750, -0.3750] |
| `elapsed_seconds` | `json_schema` | -0.3593 | [-2.0190, 1.3722] |
| `strict_agent_rate` | `binary` | -0.3125 | [-0.6250, -0.0625] |
| `joint_strict_success` | `binary` | -0.2500 | [-0.6250, 0.0000] |
| `semantic_agent_rate` | `binary` | -0.1875 | [-0.5000, 0.1250] |
| `joint_semantic_success` | `binary` | -0.1250 | [-0.5000, 0.2500] |
| `payload_bytes` | `binary` | 4.3750 | [3.6250, 5.5000] |
| `serialized_message_file_bytes` | `binary` | 4.8750 | [-31.0000, 41.1250] |
| `model_tokens` | `binary` | -568.3750 | [-1310.5000, 0.2500] |
| `message_count` | `binary` | 0.0000 | [-0.3750, 0.3750] |
| `elapsed_seconds` | `binary` | 1.8259 | [0.2751, 3.6896] |
| `strict_agent_rate` | `autoform` | -0.2500 | [-0.6250, 0.0000] |
| `joint_strict_success` | `autoform` | -0.2500 | [-0.6250, 0.0000] |
| `semantic_agent_rate` | `autoform` | -0.1250 | [-0.5000, 0.2500] |
| `joint_semantic_success` | `autoform` | 0.0000 | [-0.5000, 0.5000] |
| `payload_bytes` | `autoform` | -41.2500 | [-48.7500, -36.0000] |
| `serialized_message_file_bytes` | `autoform` | -53.0000 | [-82.6250, -35.7500] |
| `model_tokens` | `autoform` | -197.3750 | [-828.7500, 362.8750] |
| `message_count` | `autoform` | -0.1250 | [-0.3750, 0.0000] |
| `elapsed_seconds` | `autoform` | -0.7429 | [-3.0870, 1.4066] |
| `strict_agent_rate` | `no_communication` | -0.0625 | [-0.1875, 0.0000] |
| `joint_strict_success` | `no_communication` | 0.0000 | [0.0000, 0.0000] |
| `semantic_agent_rate` | `no_communication` | 0.5625 | [0.1250, 0.8750] |
| `joint_semantic_success` | `no_communication` | 0.6250 | [0.2500, 0.8750] |
| `payload_bytes` | `no_communication` | 10.5000 | [8.3750, 12.3750] |
| `serialized_message_file_bytes` | `no_communication` | 208.7500 | [195.5000, 233.1250] |
| `model_tokens` | `no_communication` | -551.2500 | [-1285.8750, 204.8750] |
| `message_count` | `no_communication` | 2.1250 | [2.0000, 2.3750] |
| `elapsed_seconds` | `no_communication` | -14.7749 | [-21.6110, -9.4099] |

## pilot_v0_6

| Paired metric | Comparator | Mean difference | 95% percentile interval |
|---|---|---:|---:|
| `strict_agent_rate` | `scaffold_only` | 0.5000 | [0.2500, 0.7500] |
| `joint_strict_success` | `scaffold_only` | 0.7500 | [0.3750, 1.0000] |
| `semantic_agent_rate` | `scaffold_only` | 0.5000 | [0.2500, 0.7500] |
| `joint_semantic_success` | `scaffold_only` | 0.7500 | [0.3750, 1.0000] |
| `payload_bytes` | `scaffold_only` | -129.5000 | [-173.0000, -84.0000] |
| `serialized_message_file_bytes` | `scaffold_only` | -235.2500 | [-325.2500, -139.3750] |
| `model_tokens` | `scaffold_only` | -1551.1250 | [-2213.7500, -791.3750] |
| `message_count` | `scaffold_only` | -1.1250 | [-1.6250, -0.6250] |
| `elapsed_seconds` | `scaffold_only` | -1.9109 | [-2.8236, -0.9840] |
| `strict_agent_rate` | `concise_nl` | 0.0625 | [0.0000, 0.1875] |
| `joint_strict_success` | `concise_nl` | 0.1250 | [0.0000, 0.3750] |
| `semantic_agent_rate` | `concise_nl` | 0.0625 | [0.0000, 0.1875] |
| `joint_semantic_success` | `concise_nl` | 0.1250 | [0.0000, 0.3750] |
| `payload_bytes` | `concise_nl` | 4.0000 | [4.0000, 4.0000] |
| `serialized_message_file_bytes` | `concise_nl` | 4.0000 | [4.0000, 4.0000] |
| `model_tokens` | `concise_nl` | -21.1250 | [-22.6250, -20.0000] |
| `message_count` | `concise_nl` | 0.0000 | [0.0000, 0.0000] |
| `elapsed_seconds` | `concise_nl` | -0.0043 | [-0.0513, 0.0479] |
| `strict_agent_rate` | `json_schema` | -0.0625 | [-0.1875, 0.0000] |
| `joint_strict_success` | `json_schema` | -0.1250 | [-0.3750, 0.0000] |
| `semantic_agent_rate` | `json_schema` | -0.0625 | [-0.1875, 0.0000] |
| `joint_semantic_success` | `json_schema` | -0.1250 | [-0.3750, 0.0000] |
| `payload_bytes` | `json_schema` | -10.0000 | [-10.0000, -10.0000] |
| `serialized_message_file_bytes` | `json_schema` | -10.0000 | [-10.0000, -10.0000] |
| `model_tokens` | `json_schema` | -421.1250 | [-724.1250, -268.5000] |
| `message_count` | `json_schema` | 0.0000 | [0.0000, 0.0000] |
| `elapsed_seconds` | `json_schema` | -0.2236 | [-0.4096, -0.1150] |
| `strict_agent_rate` | `binary` | 0.7500 | [0.5000, 0.9375] |
| `joint_strict_success` | `binary` | 0.7500 | [0.3750, 1.0000] |
| `semantic_agent_rate` | `binary` | 0.7500 | [0.5000, 0.9375] |
| `joint_semantic_success` | `binary` | 0.7500 | [0.3750, 1.0000] |
| `payload_bytes` | `binary` | -6.7500 | [-12.5000, -1.3750] |
| `serialized_message_file_bytes` | `binary` | -6.7500 | [-12.5000, -1.3750] |
| `model_tokens` | `binary` | -250.3750 | [-273.0000, -229.6250] |
| `message_count` | `binary` | 0.0000 | [0.0000, 0.0000] |
| `elapsed_seconds` | `binary` | -0.2113 | [-0.3113, -0.1311] |
| `strict_agent_rate` | `autoform` | 0.2500 | [0.0625, 0.5000] |
| `joint_strict_success` | `autoform` | 0.3750 | [0.1250, 0.7500] |
| `semantic_agent_rate` | `autoform` | 0.2500 | [0.0625, 0.5000] |
| `joint_semantic_success` | `autoform` | 0.3750 | [0.1250, 0.7500] |
| `payload_bytes` | `autoform` | -200.8750 | [-282.1250, -125.1250] |
| `serialized_message_file_bytes` | `autoform` | -330.1250 | [-441.7500, -227.2500] |
| `model_tokens` | `autoform` | -2209.3750 | [-2709.8750, -1707.0000] |
| `message_count` | `autoform` | -1.3750 | [-1.7500, -1.1250] |
| `elapsed_seconds` | `autoform` | -2.2811 | [-3.0834, -1.5459] |
| `strict_agent_rate` | `no_communication` | 0.9375 | [0.8125, 1.0000] |
| `joint_strict_success` | `no_communication` | 0.8750 | [0.6250, 1.0000] |
| `semantic_agent_rate` | `no_communication` | 0.9375 | [0.8125, 1.0000] |
| `joint_semantic_success` | `no_communication` | 0.8750 | [0.6250, 1.0000] |
| `payload_bytes` | `no_communication` | 10.0000 | [7.8750, 12.1250] |
| `serialized_message_file_bytes` | `no_communication` | 196.0000 | [193.8750, 198.1250] |
| `model_tokens` | `no_communication` | -1263.5000 | [-1270.7500, -1258.5000] |
| `message_count` | `no_communication` | 2.0000 | [2.0000, 2.0000] |
| `elapsed_seconds` | `no_communication` | -2.2699 | [-2.3588, -2.1881] |

## Interpretation boundary

This reanalysis can quantify paired exploratory differences in the existing runs. It cannot repair instruction non-adherence (the v0.6 JSON arm emitted invalid JSON), separate model/backend changes across v0.5/v0.6, infer population-level superiority from eight tasks, or create a matched-budget frontier. Message format was not the only varying causal factor across conditions. The public JSONL contains sanitized per-task outcomes and cost summaries only; source traces remain ignored under `.cache/`.
