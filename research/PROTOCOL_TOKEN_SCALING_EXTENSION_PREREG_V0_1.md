# Protocol token scaling extension v0.1 (exploratory preregistration)

**Status:** post-sweep exploratory falsification; not part of the frozen v0.1 sweep and not confirmatory inference.

## Why extend the range

The frozen sweep ends at `d=10`. It found that compact fields' known-token premium over JSON declines from +42 tokens at `(d,V)=(3,2)` to +27 at `(10,10)`, while the digit-code premium grows. That trend creates a real possible reversal: the compact card has a mostly fixed instruction cost, while its per-field message can save tokens repeatedly as dimensionality grows. Stopping at the task's current four fields would miss that crossover if it exists.

## Frozen exploratory extension

- Dimensions `d ∈ {20,40,80}` and cardinalities `V ∈ {2,4,8,10}`.
- Sixteen deterministic synthetic prompt records per setting, four candidates per receiver table, and the same identity-rank modular holdout construction and content-token accounting as the frozen sweep.
- Same pinned Qwen3-4B tokenizer, JSON/v0.3/symbolic arms, exact runner prompt constructors, and exclusion of model outcomes, chat templates, receiver answers, and inference compute.
- Preserve and report every setting, including negative results. Do not infer model utility or natural-task scaling from these prompt-cost fixtures.

## Exploratory prediction P19c

The compact-fields known-token premium against JSON will reach zero and become negative at a sufficiently large `d` for at least one tested `V`, because its per-episode output saving grows with the number of fields while the generic card text is fixed. If it stays positive through `d=80`, that falsifies the proposed crossover over this tested range. The explicit symbolic codebook is predicted to remain more expensive, with its instruction burden increasing roughly with `dV` under these bounded-length labels.

This extension is motivated by the preregistered v0.1 trend after it was observed. It must be labeled exploratory and cannot be used to rewrite the original sweep's prediction or to establish an inferential scaling law. Any discovered crossover needs a fresh frozen confirmation range and then a model task-success/cost evaluation.
