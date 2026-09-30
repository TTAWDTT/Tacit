# Protocol prompt/token scaling sweep v0.1

**Status:** prespecified deterministic, tokenizer-only measurement; no model outputs, task-utility comparison, or inference scaling claim.

## Motivation

The fixed four-axis Qwen3 audit found that the runner's digit-symbol arm emits a shorter message than JSON but spends many more input tokens on two explicit value/code maps. The current runner visibly enumerates every value for every attribute in both sender and receiver instructions. Before considering any learned or hard-coded codebook, measure how this acquisition/instruction cost grows with task dimensions.

This follows the cost decomposition in [Theory §4](../docs/THEORY.md#4-a-checkable-setup-cost-crossover): report setup/instruction, per-episode context, and message costs separately. It uses the modular higher-order task geometry from [Theory §17](../docs/THEORY.md#17-exact-support-and-communication-scaling-for-the-modular-ood-split), but it does not score agents or simulate success.

## Frozen sweep

- Tokenizer: Qwen/Qwen3-4B `tokenizer.json`, revision `eb971e9fb1f41c13b5e5a56e56886305c5ad94a0`, SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
- Representations: built-in compact JSON, frozen compact labeled-fields v0.3, and the runner's fixed per-axis digit map. Keep exact runner `_Protocol` instructions.
- Dimensions `d ∈ {3,4,6,8,10}`; cardinalities `V ∈ {2,4,8,10}`; 16 deterministic episode-shaped prompts per `(d,V)`; four receiver candidates per episode. Restrict `V ≤ 10` so the current unseparated digit message has one decimal character per axis.
- Meanings/candidates come from the identity-rank modular holdout `sum(indices) mod V = 0`. Candidate tables are public-input fixtures for cost construction, not held-out model outcomes. Target positions rotate evenly. Use the same private meaning and candidate table for all representations in each episode.
- Reconstruct exact sender and receiver system/user content as the v0.4 runner does. Count UTF-8 bytes and Qwen content tokens with added special tokens disabled. Include both request inputs, the ideal sender message output, and the delivered message echoed inside the receiver's JSON transcript. Exclude receiver output, endpoint templates/special tokens, protocol installation/cache effects, and all model compute.

## Predictions

**P19a — explicit codebook growth.** For bounded label lengths, the digit arm's raw value-map instruction bytes grow with `dV`; JSON and v0.3 instructions do not enumerate all values and should have much smaller growth in `V`. Qwen BPE token counts need not be strictly monotone at every adjacent point, so UTF-8 instruction bytes are the primary deterministic check and tokenizer counts are the deployment-specific secondary check.

**P19b — output/context trade-off.** The digit payload uses exactly `d` UTF-8 digit bytes for `V≤10`, while its explicit codebook cost grows with the vocabulary. Its known total-cost gap against JSON should therefore grow with `V` at fixed `d`, unless tokenizer segmentation or the shared task context dominates. This is a cost-accounting prediction only; it predicts no receiver-accuracy behavior.

## Decision rule and limits

If the fixed codebook's recurring prompt cost dominates its message reduction throughout this small sweep, do not call it a communication-efficient language under the runner's actual prompt-distribution policy. A one-time stored decoder, trained token, or provider-side prefix cache is a different runtime condition and must state and amortize its setup/cache costs under a declared reuse horizon. Any tokenizer-cost win still needs a gated model experiment with task success and provider-reported input/output usage.
