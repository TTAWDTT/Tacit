# Dense protocol token crossover confirmation v0.1

**Status:** preregistered deterministic confirmation of a tokenizer-cost pattern; not a model-performance or statistical-inference study.

## Prior evidence and question

The original frozen sweep ended at `d=10`. A post-sweep exploratory extension measured `d∈{20,40,80}` and found compact-fields v0.3 minus JSON known-content-token differences of `+7`, `−33`, and `−113` at those three dimensions, identically for `V∈{2,4,8,10}`. This motivated the present dense confirmation range. The prior extension is exploratory; this file freezes the next sweep before generating any intermediate-dimension measurements.

Question: where does the sign change occur on the integer dimensionality grid, and does it replicate across the four cardinalities under the exact same runner prompts and pinned tokenizer?

## Frozen design

- Dimensions: every integer `d ∈ {20,21,…,40}`.
- Cardinalities: `V ∈ {2,4,8,10}`.
- Sixteen deterministic prompt records per `(d,V)`, four receiver candidates each. Use the same identity-rank modular holdout fixture generator as v0.1 and the exploratory extension.
- Compare built-in compact JSON, the unchanged compact labeled-fields v0.3 card, and the unchanged per-axis digit-symbol protocol. Reconstruct both roles' exact v0.4 instruction and user-content strings.
- Tokenizer: Qwen/Qwen3-4B `tokenizer.json`, revision `eb971e9fb1f41c13b5e5a56e56886305c5ad94a0`, SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`; `add_special_tokens=False`.
- Primary measure: mean per-episode known content-token difference `compact_fields_v0.3 − JSON`, including both request inputs, the receiver-side message echo, and ideal sender output. Report all three arms, the components already emitted by the auditor, and every `(d,V)` setting.
- Exclude receiver output, endpoint templates/special tokens, provider caching, actual model outcomes, inference compute, billing, latency, and all claims about task utility.

## Predictions and decision rules

**P20a — bracket replication.** At each cardinality, compact-fields minus JSON is positive at `d=20` and negative at `d=40`, reproducing the previously observed bracket. If either endpoint changes sign, flag a reproducibility discrepancy and do not claim a dense crossover location until the prompt/tokenizer hash chain is reconciled.

**P20b — dense location.** At every tested cardinality, the first integer `d` whose compact-fields minus JSON mean is `≤0` will be in `{23,24,25}`. This narrow prediction follows the exploratory endpoint slope; it is intentionally falsifiable. Report the actual first non-positive `d` separately for each `V`. If the locations differ by cardinality, report the heterogeneity and reject a single cardinality-invariant crossover claim.

**P20c — symbolic cost.** The fixed-symbol arm will remain more expensive than JSON at every setting, and its mean known-token excess will increase with `V` at each fixed `d`, due to the explicit axis/value maps. Report violations without changing the design.

No regression line will be presented as an inferential scaling law. A sign change in this dense deterministic grid only locates a tokenizer-and-prompt-specific arithmetic crossover. No model superiority follows. Any later task experiment requires an independent frozen outcome plan and all existing resource/capability gates.

## Reproducibility boundary

The sweep is deterministic prompt construction and tokenizer counting. It loads no model weights and makes no inference requests. Bind the generated JSON to this preregistration hash, both preceding sweep JSON hashes, the pinned tokenizer, runner source, compact-fields codec, and card. Preserve every setting, including falsifications and discrepancies.
