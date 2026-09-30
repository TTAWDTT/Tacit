# Iterative prompt optimization as an optimized-natural-language baseline v0.1

**Status:** baseline-method audit and preregistration requirements; no model-based optimizer experiment was run.

## Primary sources

- Opsahl-Ong et al., [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPRO)](https://arxiv.org/abs/2406.11695). MIPRO proposes task/program-aware instructions, bootstrapped demonstrations, minibatch evaluation, and meta-optimization of proposal generation for a downstream metric.
- Stanford NLP's [official MIPROv2 documentation](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md) and [implementation](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/mipro_optimizer_v2.py). MIPROv2 can optimize instructions alone by setting both demonstration limits to zero; its standard method otherwise searches instruction/demo combinations with Bayesian optimization and candidate evaluation.
- Wu et al., [LLM Prompt Duel Optimizer (PDO)](https://aclanthology.org/2026.findings-acl.490/), Findings of ACL 2026. PDO treats pairwise LLM-judge preferences as a dueling-bandit signal and mutates top performers under a bounded comparison budget.
- Gupta et al., [OPTiMACS](https://aclanthology.org/2026.findings-acl.1441/) and this repository's [source audit](OPTIMACS_AUDIT_V0_1.md). This is a learned task-conditioned message-format policy and remains a separate, stronger format-selection baseline.

## What this changes

The v0.4 natural-language condition is an unoptimized plain-English control. `induce_protocol_cards.py --protocol-family plain_english` makes one training-only call that proposes several English instruction cards; `select_protocol_frontier.py` then retains a Pareto set from validation runs. This is useful candidate generation and validation selection, but it does not feed measured task failures back into later proposals. The repository's existing OPRO discussion correctly says that this is not iterative score-feedback optimization. Therefore neither the plain condition nor the one-shot card family may be called the project's **optimized natural-language baseline**.

MIPRO is not itself a message codec or an LLM-to-LLM language. It is a strong prompt optimizer that can tune the two endpoint instructions against the exact task metric. The instructions can be constrained so the sender must emit ordinary English with the canonical field names and values, while the receiver must use ordinary English semantics. This yields a direct test of whether prompt search alone can close the task-success/cost gap. MIPRO's few-shot demonstration search is a distinct adaptive-system condition: examples add setup and repeated prompt cost and could encode a convention, so report it separately from zero-shot instruction optimization.

PDO's judge-based preferences are not the preferred primary optimizer signal here. The fixture has exact task labels and an exact candidate-ID scorer. Use the ground-truth task score for prompt search, avoid an extra LLM judge and its cost/variance, and keep validation selection separate from final test scoring.

## Required comparison before a representation-superiority claim

1. **Freeze the search boundary.** The optimizer may inspect only training-stage role data and exact training outcomes. It must never read validation/test gold files while proposing or revising instructions. Preserve every proposal, prompt, completion, rollout, and source hash.
2. **Constrain the NL family.** Require complete grammatical English using all canonical attribute names and exact values; prohibit abbreviations, symbolic codewords, JSON, tables, equations, candidate IDs, and whole-tuple lookup tables. Fix the receiver answer contract. Audit actual message adherence separately from task success.
3. **Use three distinct stages.** Optimize on the existing train partition; use validation only to select the nondominated frozen candidates; open test results once after the freeze. Do not iteratively tune on validation then report that same validation score as evidence.
4. **Separate zero-shot prompts from demonstrations.** First optimize sender/receiver instructions with demonstrations disabled. If a MIPRO-style few-shot condition is added, report it as an adaptive system, retain its example artifact, charge acquisition and distribution setup plus repeated prompt tokens, and include a shuffled-example control.
5. **Charge the optimizer.** Include every generator, bootstrap, candidate rollout, validation call, retry, prompt/completion token, measured byte, wall/service time, and decoder/setup artifact. State the declared reuse horizon and compare complete frontiers; do not hide selection costs as free preprocessing.
6. **Respect local resource limits.** The official implementation inspected for this audit sets its automatic `light` validation size to 100 examples and a minimum minibatch size of 50; those defaults do not fit the project's 12-call-per-batch harness contract for a two-call sender/receiver episode. Do not run the optimizer unchanged on the project's 8-GiB local GPU. Pin the upstream version, freeze a smaller explicit candidate/trial/query budget, use the exact scorer, batch within the existing request ceiling, and disclose that local-budget adaptation. Never weaken the existing host resource gate.
7. **Keep the full baseline family.** Compare optimized NL with the unoptimized English control, AutoForm-style open format, JSON, the fixed symbolic control, and any frozen reusable card under the same task/model/schedule and channel/inference budgets. OPTiMACS remains a separate adaptive-format policy comparison; a prompt optimizer does not substitute for it.

## Falsifiable decision

If zero-shot optimized English or its costed few-shot variant matches or dominates a proposed compositional representation on the held-out task frontier, the evidence does not justify a new representation for that task/model condition. If a new representation wins only against the unoptimized English instruction, the result is a prompt-search gap, not a language gain. A representation claim requires the frozen optimized-NL comparator to be beaten at equal task success or equal total cost, with uncertainty over independent task clusters.

## Current decision

The one-shot-induced English condition remains a lower-tier feasibility control. Since this audit was written, Tacit added an iterative exact-score OPRO-style controller, complete call-capped candidate rounds, and train-only batch aggregation; see the updates in the research log. That controller has not been run against a model. GEPA is a further reflective-Pareto candidate, audited separately in [`PROMPT_OPTIMIZATION_GEPA_AUDIT_V0_1.md`](PROMPT_OPTIMIZATION_GEPA_AUDIT_V0_1.md). Before any representation-superiority claim, compare against the strongest feasible frozen English optimizer on untouched test clusters and charge its full search cost. No optimized-English model result exists, and no model was loaded during the source audit.
