# Pending experiment specification — not authorized to execute models

## Main identification question

Within each qualified receiver, compare receiver-adapted vs receiver-independent
selection with the **same family inventory and search budget**. Cross this with
family: optimized concise NL, JSON/key-value, compact fields, flat arbitrary
codebook, randomly relabeled codebook, explicit composition, and reviewed route A.
Include plain original NL, FULL, no-message and missing-source controls. FULL is a
full-information reference, not a proven optimum; infeasible tight points stay
infeasible. Keep content, schedule, number of turns, framing, priors, candidate
tables, temperature, completion limits, and task seeds paired. Deterministic sender
and model sender are separate experiments. Content selection, if later introduced,
requires the same selected facts expressed in NL and JSON.

The primary adaptation contrast is within a family at matched budget; a protocol
contrast compares families after equal search. Factorial interaction checks whether
adaptation helps codes beyond NL. A receiver-independent arm uses predeclared
source receivers for selection, spending the same proposal/evaluation budget;
target feedback is inaccessible. Include zero-search controls, and equal additional
inference/example budgets given to fixed baselines. Record actual used budgets and
all unsuccessful candidates, not just ceilings. Vary search budget within each
family as a secondary curve; do not attribute a larger search budget to language.

Predictions: at matched effort the adapted card improves strict target success at
fixed feasible complete cost over independent selection; correct mappings improve
over shuffled associations at identical messages and meanings; transfer to an
unseen receiver can attenuate this gain; costs may only recover at sufficiently
large N. Failure of the first contrast narrows the claim to search rather than
adaptation. Failure of the mapping contrast undermines association use. If equal
extra examples/inference removes gains, reject a protocol-specific explanation.
No synthetic receiver is allowed as supporting evidence for these predictions.

## Transfer arms and boundaries

| Arm | New receiver data permitted | Accounting and interpretation |
|---|---|---|
| Zero onboarding | No card, examples or feedback; public ontology only | Original discovery retained; actual requests charged; zero-shot |
| Fixed card | Source-frozen card, no target feedback | Card distribution plus every repeated context; instruction-assisted transfer |
| Bounded examples | Source-frozen training meaning/message examples; proposed counts 1,2,4,8,12 | Generation, selection and repetitions charged; few-shot, not zero-shot |
| Re-adaptation | Target receiver train feedback, separate validation selection | New discovery and deployment costs; adaptation, not zero-shot |

Freeze source protocol before naming held-out receiver outcomes. Hold source
artifact and exemplar selection constant across receivers. Mapping control preserves
meaning and label multisets, but shuffles their association with a frozen seed;
charge control construction. Lexical relabeling and mapping corruption are distinct
controls. No test feedback is permitted in any arm. Changing ontology requires a
separate task definition; new combinations within the same ontology are not that.

## Ledger schema required before model execution

One event row per proposal/evaluation/call/deployment/tool/scoring operation:
`event_id, parent_id, candidate_hash, family, receiver/config_hash, sender_hash,
model/tokenizer_hash, stage, split_seed, episode_seed, model_seed, prompt_hash,
generator_version, artifact_hash, phase, attempted/success/invalid/rejected,
semantic_correct, strict_valid, message_faithful, truncated, generated/delivered,
utf8_input/output/envelope_bytes, input/output_tokens_by_role_and_tokenizer,
call_count, observable_reasoning_usage, wall/service_latency, actual_fee_currency,
cache_hit, reset_policy, session_id, fixed_or_recurring, raw_source_hash`.
Unobserved quantities are null with reason, not zero. Separate R&D history,
reproducible discovery, deployment, recurring events; assign each charge exactly
once. Store the original request and outcome, including failed candidates and
undelivered oversize messages. Original runner result schemas remain unchanged;
an audited adapter is future work, not silently assumed present.

Report per-axis raw totals and amortized costs at proposed N=1,10,100,1000 and the
long-run limit, plus every derived threshold (including never/unknown/temporary).
Count card/example context every stateless request. Claim cache discounts only with
observed cache evidence and explicit resets. Do not sum tokens across model axes.

## Freeze and resource checklist (unresolved = blocker)

Existing tracks only: INDEX_m is capability/plumbing (4 strict FULL passes before
8 messages, total 12), Emergent OOD independent 12/12 qualification ledger, Private
Match independent calibration. None substitutes for the other. Interface repairs
require fresh qualification; diagnostic suffix-ID extraction cannot change strict
scoring. No real receiver is qualified by this extension.

Before confirmation, lock actual receiver/model/tokenizer IDs and hashes, all
prompts/cards, role-view audit, generator/split/episode/model seeds, train support,
validation support and sealed test hashes; independent splits must preserve atomic
coverage and keep test combinations out of training examples. Use repository keyed
role ledgers; filenames/metadata/IDs cannot enter prompts. Reference unit fixtures
are not a substitute for this audit.

Primary outcome: paired split-level strict task success at a preselected complete
cost budget. Primary complete cost axis and any conversions remain **unresolved**;
choose from validation feasibility and the intended deployment, not test outcomes.
Wire bytes remain a separate existing runner cap. Choose one loose and at least two
validated binding common budget points; record actual refusals/overflows, never
interpolate inaccessible points. Freeze main contrast, family/multiplicity set,
reuse horizons, independent split count and precision/power plan from validation
variance and approved budget. Use paired split-level differences and a declared
cluster interval method; specify receiver strata separately. Repeated episodes in
one split are not independent splits. No arbitrary five-point success threshold.
Noninferiority or practical margins need deployment justification before testing.

Missing, rejected and invalid outcomes count as strict failures in the planned
intention-to-evaluate denominator; missing costs block the affected frontier axis.
Retain failure types and all seeds/receivers/budget points. Stop at fixed registered
sample count or approved cost ceiling, not significance. Gate/hash/leakage/cost
failures halt the affected batch. Sequential stopping needs a separately frozen
corrected design. Uncertain result plus exhausted resources remains uncertain.

Resource request must list model storage/download bytes, local vs external compute,
call ceilings for qualification, proposals, all candidates/validation/transfer/test,
maximum input/output/reasoning budgets, fee ceiling and shutdown conditions. Current
approved scope is **zero model requests, zero large downloads, zero external GPU**.
No real experiment is executable from this document until these blanks are filled,
reviewed and new expenditure authorized. Existing PRs #1/#2/#3 remain optional future
infrastructure dependencies, not dependencies of these offline tests.
