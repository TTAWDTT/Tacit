# GEPA as a reflective optimized-English baseline (v0.1)

**Status:** literature and design audit only. GEPA was not installed, run, or queried against a model. This audit does not add a performance result.

## Why audit it now

Tacit now has a training-only, exact-score OPRO-style English prompt-search controller. That closes the gap between a one-shot candidate generator and an iterative optimized-English baseline, but it does not exhaust strong prompt optimizers. GEPA (Genetic-Pareto) is a directly relevant newer method: it reflects on execution trajectories, proposes text mutations, and uses Pareto-aware candidate selection. Its paper reports results on six other tasks, including improvements over MIPROv2; those results establish GEPA as a serious candidate baseline, not as evidence about Tacit's tasks ([Agrawal et al., arXiv v2, accepted to ICLR 2026](https://arxiv.org/abs/2507.19457)).

The official [`gepa-ai/gepa` API](https://gepa-ai.github.io/gepa/api/core/optimize/) supports task and reflection models separately, explicit train/validation sets, per-instance Pareto tracking, `max_metric_calls`, and `max_reflection_cost`. Its official guide describes iterations with task-model minibatch evaluation, a reflection-model call, candidate validation, and possible full validation; this makes total provider requests depend on the adapter and validation policy ([budget guide](https://gepa-ai.github.io/gepa/guides/faq/#how-do-i-control-gepas-runtime-and-budget)). The repository's active controller is therefore a useful low-dependency OPRO-style baseline, but a GEPA comparison would test a materially stronger reflection-and-search mechanism.

## Fit and important mismatches

| Design property | Tacit's current search controller | GEPA capability | Required Tacit treatment |
|---|---|---|---|
| Candidate improvement | Exact-score failures and capped error examples are fed to later English proposals | Natural-language reflection over task trajectories/feedback | Include only training-stage examples, exact outcomes, and permitted public traces |
| Search state | Fixed small number of proposal rounds; each round has a complete candidate manifest | Candidate pool, reflective mutations, optional merges, Pareto-aware parent selection | Freeze a hard global task/reflection request budget and save every candidate/trace |
| Cost cap | Runner caps each task batch at 12 model calls | `max_metric_calls` counts evaluator/metric work; reflection is a separate model path, and one metric call may contain multiple endpoint requests | Enforce a request counter at the actual `ChatModel.complete` boundary for task and reflection calls; never treat metric-call count as a provider-call cap |
| Paired data | All prompt candidates use the same candidate-set episodes | Per-instance Pareto can track candidate strengths across examples | Make one balanced candidate-set cluster the adapter instance and retain identical IDs/order across candidates |
| Held-out selection | Validation selects a frozen Pareto set once after training search | GEPA can use a `valset` repeatedly for Pareto tracking | Run search with training data only (`valset=None` or a train-only inner split); keep the project's held-out validation untouched until the final freeze, and test untouched once |
| Optimizer architecture | Local exact-score prompt lineage, OPRO-style | Reflective trajectory-driven mutation and evolutionary Pareto pool | Compare the algorithms under the same task-model population, English-only constraint, training instances, and total request/cost budget |

## Recommended controlled adaptation

1. **Keep the current search as the low-dependency comparator.** The implemented method is an OPRO-style adaptation, not a reproduction of OPRO, MIPRO, or GEPA. Do not silently relabel it.
2. **Constrain GEPA to ordinary English.** Optimize sender and receiver instruction text only. Keep the sender output rule explicit: canonical attribute names and values in grammatical English; no JSON, code, symbolic tokens, tuple lookup tables, or candidate IDs. Freeze the receiver answer contract and exact task metric outside the optimizer.
3. **Make one candidate-set cluster one adapter example.** Its score is the exact mean selection success over the four balanced target episodes, accompanied by per-episode outcomes and allowed training-only traces. Do not present four targets from one set as four independent clusters. All candidates must use the same preregistered cluster sequence.
4. **Protect the true held-outs.** Use `valset=None` so GEPA's internal Pareto state is derived from training data, or predeclare an inner training-only cluster split. After search ends, open the existing validation partition once for the same frozen candidate-selection rule used by the local OPRO arm. Open test only after both arms are frozen. Because validation is small, disclose candidate multiplicity and keep test split seeds as the inferential unit.
5. **Meter real calls, not optimizer abstractions.** `max_metric_calls` alone is insufficient for this task: a cluster evaluation invokes sender/receiver inference for multiple target episodes, while reflection uses another model endpoint. Wrap both endpoint clients with one persisted global request ledger and hard cap; include retries and failed requests, token use, bytes, wall/service time, reflection prompts/completions, and cache hits. Do not reuse cached outcomes across nominally distinct conditions without identical content hashes.
6. **Respect resource gates and checkpoints.** The existing host report fails the frozen envelope, so no model call is currently eligible. If a future fresh report passes, run at most the preregistered capped batch, checkpoint before continuing, and require a new passing report wherever the local runner's batch policy requires it. Pin the GEPA release and source commit before implementation or reproduction; the live documentation and defaults move.
7. **Keep adaptation separate from demonstration search.** This first comparison optimizes instructions only. Few-shot examples/onboarding are a separate system condition with their acquisition, prompt, and reuse costs charged.

## Falsifiable decision rule

Under equal task-model and reflection-model request budgets, identical training clusters, a fixed English constraint, and a single held-out test evaluation, GEPA should improve at least one task-quality/cost operating point over the current OPRO-style controller if reflective trajectory feedback adds useful search information. If it only raises training scores, consumes more reflection/inference cost, or loses on the test frontier, Tacit has no evidence that GEPA's stronger optimizer buys useful generalization for this workload. A representation claim must then compare against whichever English optimizer has the better held-out frontier, or explain the preregistered deployment/budget condition that excludes it.

## Scope and source limits

GEPA optimizes textual components of a prompted system; it is not itself a communication language, codec, or learned-symbol baseline. Its reported gains and sample-efficiency claims come from other tasks and budgets. This note relies on the GEPA paper and official documentation, including moving documentation accessed on 2026-09-30; pin a released artifact and re-audit defaults before any experiment. No dependency was added and no request was sent.
