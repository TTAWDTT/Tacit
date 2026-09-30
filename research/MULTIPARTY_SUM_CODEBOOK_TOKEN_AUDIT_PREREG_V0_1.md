# Preregistration: Qwen3-4B content-token audit of finite sum codebooks (P23)

**Status:** frozen before running the deterministic tokenizer count. This is a cost-only audit, not a model experiment.

## Question and prediction

How much model-facing Qwen3-4B content-token overhead does an explicit exhaustive codebook protocol add relative to existing decimal, JSON, and fixed-width binary text senders on the same private-sum task? Do ideal payload bits predict actual prompt/output token cost?

Prediction P23a: explicit codebook instructions will make its content-token count larger than the bare binary payload count, because every active sender receives an encoder table and the receiver receives a MAP lookup table. P23b is deliberately unsigned: codebook versus decimal/JSON total content-token ranking is not predicted from the bit frontier. Either outcome is informative. These predictions concern tokenizer counts, not task quality.

## Frozen design

- Pin `Qwen/Qwen3-4B` tokenizer revision `eb971e9fb1f41c13b5e5a56e56886305c5ad94a0`, SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
- Reuse the exact v0.1 task string, role contexts, final-answer instruction, and runtime JSON user-content serialization.
- Exhaustively enumerate every private-value vector for `m=2,3,4` and every budget with at least one active frontier sender. Compare the frozen codebook against existing `decimal`, `json`, and `binary` text instructions. Each condition sees the same vector; no gold vector is included in any role context beyond each sender's own private value.
- Count sender and receiver input content tokens plus ideal sender and receiver completion content tokens. Report role input, output, receiver echo, codebook serialized-card tokens, ideal payload bits, and oracle success separately.
- Use the frozen MAP output for codebook receiver completion; baseline encoders transmit their exact source value and receiver outputs the exact sum.

## Exclusions and decision rule

Tokenizer counts exclude chat templates, special tokens added by serving, hidden reasoning, provider cache/billing behavior, wall-clock latency, inference compute, and actual model errors. Serialized-card tokens are a separate artifact-size diagnostic and are not silently added to prompts. P23a is supported if every nonempty codebook setting has mean role-instruction input tokens greater than its ideal payload bits; it is falsified by any setting that does not. P23b is resolved descriptively per `(m,budget)`; no threshold will be moved after seeing results. No superiority claim follows from this audit.
