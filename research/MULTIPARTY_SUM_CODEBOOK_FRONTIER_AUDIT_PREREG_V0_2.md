# Preregistration: exact-runner prompt and no-message frontier audit (P24)

**Status:** frozen before v0.2 counting. This amends and supersedes v0.1 for baseline-comparison claims.

**Scope note:** the no-message arm in this audit is the existing operational runner instruction, not a shortest possible hand-written decoder prompt. That keeps the comparison reproducible against the actual SDK implementation but does not claim prompt optimality.

## Reason for amendment

Review found v0.1 reconstructed only format-specific receiver instructions for decimal/JSON/binary and omitted the common private-context, missing-input, and Bayes-prior guardrail present in `examples/multiparty_private_sum.py`. This made baseline prompt cost artificially low. v0.1 remains as the historical first audit, but its cross-protocol cost ranking is not used as a calibrated comparison. v0.2 reconstructs the existing runner's complete content strings and adds the operational no-message arm.

## Question and estimands

Under the exact same IID private-sum task, what model-free exact-success frontier is attainable by no-message, operational decimal/JSON/binary text controls, and the frozen exhaustive lossy codebooks when cost includes Qwen3-4B content tokens, SDK application-wire bytes, and planned model-call count?

Primary row estimands are the exact prior-averaged success, total known content tokens (input content at all roles plus ideal completions), application bytes for delivered sender messages, and calls per episode. Role instructions, user contents, message echoes, sender completions, and final completions remain separate components. The task vectors are paired exhaustively within each `m`; no across-`m` pairing is used.

## Frozen method

- Tokenizer: pinned Qwen3-4B revision `eb971e9fb1f41c13b5e5a56e56886305c5ad94a0`, SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
- Enumerate every `{0,1,2,3}^m` vector for `m=2,3,4`.
- Reconstruct the actual baseline system text, sender/receiver contexts, final-answer instruction, user JSON, exact transmitted messages, and protocol IDs in `run_sum_episode`; its no-message schedule is empty and makes one receiver call.
- The operational baseline system prompts come from the shared `_role_instructions` constructor used by both `run_sum_episode` and the audit; the audit does not maintain a copied receiver prompt.
- Compare no-message, decimal, JSON, two-character binary text, and every nonempty exhaustive frontier codebook point. Codebook receiver outputs use the exact frozen MAP decoder. Baseline communication arms use ideal faithful messages and exact sums, so they are capability upper references rather than observed model behavior.
- Measure application-byte framing with the SDK's pure `LocalTCPMessageChannel.measure` function; count only delivered sender transmissions and state that network headers/inference are excluded.

## Decision and limits

Report the complete cost-success frontier and identify only within-`m` Pareto points; do not claim a language winner merely because a lossy protocol is cheaper than a 100%-success arm. P24 measures deterministic tokenizer/accounting costs, not generated model behavior, quality under errors, provider billing, latency, or compute. Chat templates and provider-added tokens are excluded. Results apply to this narrow finite aggregation task. Do not revise prompts or task definitions after counting.
