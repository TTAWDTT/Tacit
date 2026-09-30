# Emergent OOD v0.4: matched communication-budget frontier design

**Status:** preregistration addendum for future validation/test work; no cap values are frozen here because no model-backed message-length pilot exists. No test outcomes or model traces were inspected to choose this procedure.

## Why an explicit budget design is needed

The v0.4 runner enforces `--wire-budget-bytes` against the complete serialized application message. A sender completion that exceeds the cap is not delivered, while its generation and inference cost remain charged. This is a valid hard channel constraint. However, the default 4,096-byte ceiling is only a configured limit: if all candidate messages fit, it creates no observed bandwidth pressure. Such a run can compare measured cost and utility, but it cannot test the budget-tightness prediction H3.

## Estimand

Let `p` be a frozen protocol/format policy, `b` a cap on the complete serialized application message, and `s` an independently generated composition split. Define `Y(s,p,b)` as exact candidate-selection rate over the complete preregistered test block. At a common cap `b`, compare paired split effects `D(s,b)=Y(s,p₁,b)-Y(s,p₀,b)`. For predeclared tight and loose caps `b_t < b_l`, the budget interaction is

`I_B = E[D(s,b_t) - D(s,b_l)]`.

This difference-in-differences is an interaction on task success. Compute the same contrast separately for delivery rate, exact sender-message fidelity, application bytes, recipient-native input tokens, all inference tokens/calls, and latency. Do not merge these into a scalar score. A lower cap can reject a generated message after incurring its full generation cost; report generated, rejected, and delivered outcomes separately.

## Outcome-independent cap selection

1. **Freeze the application boundary.** Use the exact runtime serializer and its complete envelope fields. Do not substitute payload UTF-8 bytes or ideal code bits for `communication_budget_bytes`. Measure the minimum legal application frame and the exact no-message/full-information implementation paths without opening test ledgers.
2. **Use training/validation only.** On development episodes, collect raw completion lengths and serialized application lengths for every eligible strong baseline under a permissive cap. Retain malformed messages and failed calls; they remain part of the length distribution and cost ledger.
3. **Choose a shared cap grid from the pooled baseline distribution.** Freeze a small grid before test that contains (a) a loose cap above nearly all development messages and (b) at least two lower caps at which development messages from the pooled baseline set are actually rejected. Use common cap values for every representation, selected from pooled lengths rather than one protocol's preferred operating points. Include the exact minimum frame boundary only if it remains a semantically valid task condition. Record the calculation, source hashes, and full cap list in the preregistration; each per-cap run manifest records its selected cap and source episode-manifest hash.
4. **Check task relevance before test.** If no lower shared cap produces both delivered and rejected messages for the applicable baseline set, report that the pilot did not create a binding bandwidth regime. Revise the development task or cap ladder using training/validation only; do not claim to have tested H3. If every arm collapses to no delivery at the tightest point, treat that point as the no-communication endpoint rather than evidence of a useful language.
5. **Freeze before test access.** Freeze protocol cards/prompts, candidate episodes, model/tokenizer population, cap grid, primary contrast(s), multiplicity allocation, endpoint, request budget, and stopping rule. Apply every cap to identical episode IDs, split seeds, and model settings. Retain all preregistered results, including failures and missing token usage.

## Interpretation rules

- A configured but nonbinding ceiling supports only an unconstrained-by-channel comparison at that operating point.
- At a binding cap, a success gain is meaningful only with matched opportunity and fully charged inference/setup. Show both the task frontier and delivery/fidelity diagnostics; do not call a format “more information-dense” merely because it emits fewer bytes.
- The primary evidence for H3 is the predeclared paired `I_B` on independent split clusters, with its interval and multiplicity correction. A plot where one cap appears favorable is descriptive until that contrast is tested.
- If the cap changes the number of delivered senders or effective schedule, the estimand is a joint representation-plus-delivery-policy effect. Use the same sender schedule and receiver context rules across protocols, then label any cap-induced absence as part of the channel intervention.
- Report same-quality cost comparisons separately from equal-cap comparisons. A Pareto frontier over observed points is descriptive unless the frontier-selection procedure and uncertainty analysis are frozen in advance.

## Operational limit

The runner's hard batch request ceiling and fresh local resource preflight remain binding. The cap ladder must be sized to the resource-feasible pilot and complete paired design; do not increase the request ceiling, lower the CPU/GPU thresholds, or open test data to obtain more favorable points. Until a resource-feasible validation pilot provides length distributions and split-effect nuisance information, the cap grid and confirmatory sample size remain unfrozen.
