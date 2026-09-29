# Emergent OOD v0.4 clustered analysis audit

**Status:** design audit only; no model outcomes were used. This note specifies what current reports can support and what a confirmatory design still needs.

## Experimental unit and estimand

The v0.4 runner evaluates several balanced candidate sets under one generated composition split. Rows within a split reuse the same ontology, split digest, episode key, and task construction. The runner records `inference_cluster_id = split=<seed>`; candidate sets and episodes are nested within that split. They are not independent replications of the task family.

For protocol `p`, byte cap `b`, and split seed `s`, define the split-level task-success summary

`Y(s,p,b) = mean exact-selection success over the preregistered, complete set of test episodes for that split`.

For two protocols at the same byte cap, the paired split effect is

`D(s,b) = Y(s,A,b) - Y(s,B,b)`.

The population estimand is the mean of `D(s,b)` over independently generated evaluation split seeds under the declared task-generation process. Report each wire, inference-token, call, service-time, and latency dimension separately; do not scalarize them into one cost unless an explicit conversion rule is independently justified. A frontier contrast must keep the byte cap fixed. Different caps are different task strata and cannot be treated as paired protocol arms.

This estimand is conditional on the frozen model population, receiver calibration rule, candidate-set generator, protocol prompts/cards, candidate coverage, and budget definition. The 12/12 train-only receiver gate is an execution-eligibility screen for a model, not a population-level capability estimate. Run it once on a separate calibration split seed and freeze the qualified model before validation/test; do not screen each evaluation split and omit it on a failure. Report the receiver's independent calibration result separately, and retain every preregistered evaluation split.

The capability gate requires different calibration and evaluation split seeds and the same task key, task seed, vocabulary, and candidate count. Exact meanings or candidate tuples can recur across split seeds in this finite ontology. This is not treated as test leakage because the completion adapter makes stateless independent requests: calibration prompts/results are not placed in model-visible evaluation context, no model parameters are updated, and no calibration response is reused as a cached task answer. The local evaluation bundle and gold ledger remain inaccessible to the model.

## What the existing analysis supports

`tools.paired_report` v2 pairs matching episode IDs and resamples complete `inference_cluster_id` groups. It reports the mean paired episode difference, cluster-bootstrap percentile intervals, independent-cluster count, and each cluster's left/right means and paired difference. With balanced complete episode coverage per split, the pooled mean equals the equally weighted mean of split means. If batches are incomplete or split cluster sizes differ, that equivalence no longer holds; combine every preregistered offset for each condition and verify equal episode coverage before interpretation.

`tools.frontier_report` groups by the complete task/model stratum, including `communication_budget_bytes`. It describes Pareto membership at each matched cap and makes no population-dominance inference. Use paired uncertainty separately for each same-cap contrast. A green frontier point from one split is descriptive only.

## Few-cluster limits from related methods research

In a simulation study of binary cluster-randomized trials with 8–30 clusters, Thompson et al. found that unweighted cluster-level analysis, restricted-pseudolikelihood GLMM, and selected corrected GEE methods controlled type-I error in most studied scenarios; their recommendation for designs with at most 30 clusters includes cluster-count degrees-of-freedom corrections. The study also found that power depended on cluster count, cluster size, prevalence, and analysis choice ([Thompson et al., 2022](https://doi.org/10.1186/s12874-022-01699-2)).

That evidence is methodological context, not a plug-in analysis prescription: their clusters were randomized to arms, while v0.4 applies every protocol to episodes within each split and compares the arms in a paired design. The transferable points are to preserve the true independent-unit level, estimate power under the planned clustered analysis, and avoid treating large within-split episode counts as a substitute for independent splits.

## Requirements before a confirmatory test

1. Use a separate calibration split only for the fixed receiver eligibility screen. Use validation splits for prompt/card choice and budget-grid debugging. Freeze the protocol artifacts, candidate-set coverage, model/tokenizer population, wire caps, comparison family, endpoint, and stopping rule before opening test outcomes.
2. Generate evaluation split seeds independently of calibration and validation seeds. Freeze the complete seed list or its public derivation before test. Retain every eligible evaluation split and every preregistered episode regardless of test outcomes.
3. Estimate power by simulating the actual paired split-level analysis over plausible distributions of `D(s,b)` informed by a separately labeled pilot. Do not convert the independent-episode McNemar table in `PAIRED_PROTOCOL_SAMPLE_SIZE.md` into required v0.4 episode counts. Until the split-level variance and practical target effect are defensibly specified, the confirmatory split count remains unresolved.
4. For each budget and contrast, report the mean split-level success difference, all split-level values, cluster count, and an uncertainty interval whose small-sample behavior is validated under the planned data-generating scenarios. Keep absolute per-condition success and each cost dimension visible beside paired contrasts.
5. Treat fewer than 20 independent splits as a strong descriptive warning, not a universal pass/fail threshold. More candidate sets inside an existing split can improve its split summary but do not raise the independent split count.

## Operational feasibility

The default fixture has 16 candidate sets per validation/test stage. A single message arm on one balanced set plans eight calls and stays within the current 12-call cap. The new `--set-offset` lets separate batches cover later sets without duplicating episode IDs. For a complete same-cap comparison, every condition must use identical offsets and the combined result ledgers must have matching episode IDs. This is only a collection recipe; it does not authorize model execution. Every inference batch still requires a passing fresh local resource report and a valid independent receiver-calibration ledger.

## Decision

The executable frontier pipeline is now suitable for matched-budget descriptive studies and cluster-aware paired reporting when complete balanced data are supplied. It is not yet a confirmatory study plan: sample size, multiplicity family, independently generated test-split count, and evidence-based variance assumptions must be fixed after a resource-feasible validation pilot and before any test-stage outcomes.
