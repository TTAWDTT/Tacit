# Protocol onboarding transfer design v0.1

**Status:** runnable interface and model-free leak checks exist; no messages have been generated and no model has been evaluated. This is a design record, not an empirical result.

## Question

Can an unfamiliar receiver acquire a sender's frozen communication convention from examples of use, and when does the resulting task-quality gain justify acquisition and repeated-context costs?

This isolates receiver-side in-context transfer. The sender receives the sender instruction from a frozen protocol card. The receiver receives only a candidate table and meaning/message examples drawn from the training partition; it never receives the card or its decoder instruction. The evaluated target meanings belong to a held-out composition partition. It is not online adaptation: no receiver weights change and no scored answer is added to later context.

## Hypotheses and falsifiable predictions

1. **Holistic no-overlap null.** For a random holistic bijection over meanings, if none of a receiver's candidate meanings occurs among its examples and the prior over mappings is uniform, the message label for the target is independent of the target identity. Bayes accuracy remains `1/k`, regardless of the number of examples elsewhere in the domain. Any higher measured score requires a structural prior, leakage, nonuniform sampling/mapping, or non-holistic structure.
2. **Compositional transfer.** When training examples ground reusable primitives and the sender protocol composes them, a new receiver can exceed `1/k` on unseen combinations of those primitives. The gain should depend on primitive coverage and should survive independent receiver seeds/model families.
3. **Cost break-even.** Exemplar onboarding can improve end-task success but need not improve efficiency. Its frontier position depends on sender-side example-generation cost, receiver prompt tokens/bytes for every reuse, fixed decoder distribution cost for card baselines, and reuse horizon. The arm should lose for some short horizons even if its accuracy is higher.
4. **Bandwidth interaction.** If learned structure saves message bytes, its quality/cost advantage should grow under tighter message caps, but exemplar/setup context must remain a separate cost axis rather than disappearing outside the budget.

## Conditions

Use identical held-out episodes, candidate order, sender/receiver model population, decoding settings, and sender convention where applicable:

- no message and full information;
- optimized concise natural-language and JSON baselines;
- fixed symbolic/compositional code with shared decoder card;
- same sender card with usage-only receiver onboarding at a preregistered example-count ladder (for example `1, 2, 4, 8, 16, 32`), plus zero-shot transfer with no examples;
- a holistic codebook control with examples whose meanings are disjoint from the scored candidate meanings.

Select exemplar rows using a seed fixed before validation, stratify coverage by primitive values, and keep the selected example IDs/content frozen across receiver comparisons. Protocol/card development and any induction costs use training only; validation selects sample-count/frontier operating points, and the final test protocol and count are frozen before test access. Do not use a full-card arm as a substitute for the usage-only receiver: it measures shared specification transfer, a different onboarding channel.

## Measurement

Primary outcome is exact candidate selection. Report the full quality-versus-cost frontier, not only accuracy. Record separately:

- wire bytes and receiver-native tokens for each scored message;
- receiver prompt input tokens including repeated examples, all calls, output tokens, failures, retries, and service/wall latency;
- example artifact bytes per receiver request;
- one-time example-generation calls/tokens and protocol-induction/setup costs;
- fixed-card distribution bytes and any decoder/tool installation cost;
- total cost at declared reuse horizons, with setup amortized as `setup_cost / H` only when the artifact is reused for exactly the declared `H` tasks.

The runner's `communication_budget_bytes` caps the task message boundary, not onboarding context. Therefore plot message-cap frontiers and complete onboarding cost as distinct axes; do not imply equal total bytes when usage-example bytes differ. Each result now carries the deduplicated exemplar artifact in the standard `setup` array, and `--usage-reuse-horizon` declares `H`; the existing end-to-end frontier and paired tools amortize setup bytes, model calls, tokenizer-indexed tokens, and time over `H`. `usage_example_bytes_per_receiver_request` separately reports the repeated prompt context, which is already included in inference input tokens.

## Controls and validity checks

- Hash-bind every example artifact to the frozen sender card, split, and exact episode manifest. Verify each meaning ID resolves to a sender training tuple; never load or expose evaluator gold to the receiver.
- Keep semantic labels and provenance IDs out of the prompt. Preserve source traces because hashes cannot prove that a declared message was actually generated from the declared tuple/card.
- Include shuffled meaning/message pairs, shuffled message labels, and candidate-disjoint examples as negative controls. A correct-pair advantage over shuffled pairs is necessary evidence of use of the examples.
- Score the same receiver independently with full information before any message run. Freeze prompts and sample selection before test.
- Treat split/seed as the uncertainty cluster. The small current fixture is plumbing only; power/sample-size planning must precede confirmatory claims.
- Run only after a fresh passing machine preflight and independent request ceiling. No current model evidence or resource authorization is implied by this design.

## Current artifact

The runnable `usage_only_transfer` arm and JSON schema are in [`experiments/emergent_ood_v0_4/README.md`](../experiments/emergent_ood_v0_4/README.md). The runner validates schema, support membership, hashes, and declared acquisition accounting; authorship and semantic leakage still require an auditable generation process. Focused model-free tests exercise the receiver prompt boundary and CLI plumbing. No claim of transfer, superiority, or language emergence follows until real model results and the negative controls are complete.
