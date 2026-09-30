# Confirmatory OOD study: preregistration draft v0.1

**2026-09-30 — design only, NOT frozen or authorized for inference.** Work package C.
Repository: `TTAWDTT/Tacit`; inspected HEAD
`5dd260c5824b280b5323a1707ab132524afa8e4c` (13:10:32 UTC). The checkout had
only local branch `work`, no local `main`, and no existing changes. No repository
`AGENTS.md` or `.agents/skills` was present; `/workspace/.agents` was empty.
This adds a proposal and isolated synthetic checks; it changes no experiment,
scorer, launcher, threshold, or existing preregistration. No private task/test
ledgers, model outputs, weights, endpoints, or external GPUs were accessed.

## Question and primary family

Does a frozen reusable compositional card A improve held-out exact candidate-ID
selection over validation-selected optimized ordinary-English card B, under a
fixed sender→receiver schedule and a shared application-byte cap? This is a
conditional representation-policy effect for one frozen model pair and the
default v0.4 ontology, not broad task or cross-model transfer. Receiver instructions
needed to teach each card count as part of the intervention and its cost.

Use the existing strict scorer unchanged. For each independent split s, average
all 16 balanced test sets (k=4; 64 episodes) within each arm/cap:
`Y(s,p,b) = exact successes / 64`. Equal split weights define the estimand.

| Registered claim | Split contrast | Null | Practical claim requires |
| --- | --- | --- | --- |
| H1, primary: tight-cap advantage | D_s = Y(s,A,b_t) − Y(s,B,b_t), in [−1,1] | E[D] ≤ 0.05 | simultaneous lower CI > 0.05 |
| H2, key secondary: bandwidth interaction | J_s = D_s(b_t) − D_s(b_l), in [−2,2] | E[J] ≤ 0.05 | simultaneous lower CI > 0.05 |

**Proposed smallest practically important effect: 5 absolute percentage points**
(one additional exact success per 20 episodes in expectation), for both the
success difference and difference-in-differences. This is a design judgment,
not an empirically validated utility threshold; approve/freeze its rationale
before test access. An interval above zero but not above 0.05 is insufficient.
Failure to reject is inconclusive, not equivalence. H2 is calculated and published
regardless of H1, but an advantageous tight-budget protocol claim requires H1;
an interaction alone can arise from harm at the loose cap.

Only these two hypotheses form the confirmatory family (FWER ≤ .05).
Each gets a two-sided interval with alpha=.025 (Bonferroni; no independence
between contrasts required). JSON, symbolic, compact-fields, AutoForm, built-in
plain NL, no-message, full-information, and the middle cap are descriptive
controls. No additional pairwise significance claims, pooled ontology claims,
cost-dominance claims, or post-hoc winning-cap claims. Additional confirmatory
model pairs/ontologies/arms require a new family allocation before test access.

## Development selection and freeze

1. Freeze a bounded training search plan separately for A and optimized English B:
   candidate counts, rounds, total request attempts, feedback scorer, and equal
   search opportunity. A must have train-only provenance; no card is yet selected
   by this document. Failure to produce a qualified A or strong B stops this study.
2. Use a fixed, seed-disjoint validation pilot for model-free artifact checks and,
   only under separate future authorization, message lengths and split nuisance
   estimates. Never open test role/gold files or results for design choices. Keep
   training feedback inside training; disclose all candidates and search costs.
3. Freeze the exact application serializer/envelope and minimal legal frame.
   At a common permissive cap, pool complete serialized generated lengths from
   the prelisted eligible baseline set with balanced per-arm episode coverage.
   Include malformed completions that have a serializable message. Failed calls
   without a length stay missing, charged and reported; never substitute zero.
   If length coverage is incomplete, block cap freezing until a predefined
   development remediation is documented; do not silently drop failed calls.
4. Proposed deterministic grid: nearest-rank pooled quantiles Q25, Q50, Q99,
   with `Qp = sorted_lengths[ceil(p*M)-1]`, yield `b_t`, `b_m`, `b_l`.
   Equality fits the cap; rejection means length > cap. Require three distinct
   legal byte values; at least 99% fit at b_l and both delivered and rejected
   baseline messages occur at each lower cap. Otherwise mark bandwidth design
   infeasible and amend on development data only. No invented numeric caps.
   Cap acceptance is based on delivery/length, never on which arm wins.
5. Evaluate each candidate on identical complete validation splits at this grid.
   Select within each family by highest equal-split tight-cap success, then lowest
   mean generated application bytes, then lexical card SHA-256. Freeze one A and
   one B across all caps. `select_protocol_frontier.py` only returns a shortlist;
   it does not implement this winner rule. Preserve a reviewable selection table
   including every candidate, raw lengths, failures, and source hashes.
6. Freeze n, seed derivation/list, collection order, all artifacts and budgets
   in a signed-off manifest before generating/accessing test roles. Draw test
   split seeds independently from a declared PRNG/entropy source with independent
   calibration/validation namespaces. Check seed separation without looking at
   outcomes. Record randomization seed and generator version. Permuted splits or
   meanings may recur in this finite ontology: do not filter on their contents.
   Randomize paired arm/cap batch order within split using a frozen independent
   schedule seed; send stateless requests and pin model/tokenizer revisions.

## Inference, sample planning, and stopping

Independent unit = generated composition **split**, not episode, candidate set,
batch, receiver answer, or ontology vocabulary. Claims condition on frozen
training/validation-selected cards, model population, task key/generator and
ontology. Independence is a design assumption about split draws and execution;
reused context, adaptive prompts or cross-split model state invalidate it.

Primary method is the bounded-mean Hoeffding interval, clipped to the known
range. For x_s in [a,b], n splits and alpha=.025:

`r = (b-a) * sqrt(log(2/alpha)/(2*n)); CI = [max(a,mean-r), min(b,mean+r)]`.

Use width 2 for D and width 4 for J. This gives simultaneous coverage at least
95% for the two fixed contrasts under independent bounded sampling, without
normality or symmetric paired differences. It is intentionally conservative.
Cluster-bootstrap reports and the audit's empirical-Bernstein interval may be
labeled sensitivity analyses only; never switch the decision method after test.
Do not use episode McNemar counts, sign-flip tests without symmetry, or select the
most favorable interval. All per-split values and absolute arm rates are published.

**n remains unresolved.** Obtain separately labeled validation/pilot split-level
distributions and simulate this exact paired analysis over plausible variance,
skew and budget-correlation scenarios. State a target rejection probability
(proposed 80%) at an explicitly justified alternative strictly above .05;
power at the .05 null boundary is not 80%. Include sampling uncertainty in pilot
nuisance inputs and complete request feasibility. If the feasible n cannot meet
the target, declare descriptive feasibility work or redesign before test.
Twenty splits is a warning convention, not a power guarantee. Pure precision
arithmetic in the synthetic report is not a sample-size recommendation.

Stop at the frozen n complete splits and fixed request/time ceiling; no interim
efficacy/futility peeking or sample-size extension. Freeze those numerical ceilings
before authorization. Existing fresh local resource gates and 12-call batch cap
remain binding. No extra retries outside frozen policy; all attempts, interrupted
requests and resumes are charged. A resource stop pauses collection; only an
identical, integrity-checked continuation within the original ceilings is allowed.
An exhausted ceiling, hash mismatch or unrecoverable incomplete block ends the
study as incomplete, with no confirmatory claim from a complete-case subset.

Preserve failed, over-budget, truncated and malformed outcomes using existing
runner/scorer semantics. A scored failure stays in the 64-episode denominator.
An absent outcome is missing, not an invented failure or success: block primary
analysis until complete or report incomplete study. Verify exact expected IDs,
no duplicates, matching offsets and all arm/cap rows for every registered split.
Do not exclude evaluation splits based on full-information performance.

## Three separate gates and task families

| Track | Eligibility / purpose | Cannot establish |
| --- | --- | --- |
| INDEX_m v0.3 minimum four-task pipeline screen | Episodes 0,1 at m=4,8; four exact, untruncated full-information answers; at most 12 total requests, no retries | v0.4 receiver eligibility, stable success, transfer or protocol advantage |
| Emergent OOD v0.4 | Independent calibration split, training partition only; 3 balanced sets × 4 = **12/12** full-information; verified ledger, same key/task seed/k/vocabulary/model/tokenizer; different evaluation seed | Per-test-split selection or population capability estimate |
| Private Match | Separate v0.2 two-role and v0.3 three-agent studies; v0.3 has q=4, 16 rows, complementary senders, own 4-episode perfect disjoint calibration, 8 development and 8 evaluation episodes | Substitute v0.4 ledger, pool outcomes, or treat its eight episodes as OOD split replication |

Run v0.4 eligibility once for the frozen receiver; its ledger remains separate
from all validation/test contrasts. No gate may be borrowed from another task.
These numbers describe existing contracts; this draft does not amend them.

PACT/Proxifield applicability follows the [factorial audit](PACT_PROXIFIELD_FACTORIAL_AUDIT_V0_1.md):
one-shot v0.4 has neither evolving action-state history nor multiple eligible
recipients, so neither method is a valid plug-in experimental arm. Record that
exclusion now; retain strong task-appropriate English and structured controls.
Any later routing factorial requires invariant router metadata, fixed content
and routing factors, and a separately preregistered interaction. If content
changes routing metadata, label it a coupled system effect.

## Complete cost and release contract

Use [costs v3](../docs/COST_ACCOUNTING.md) and keep a vector, without exchange rates:
generated/rejected/delivered payload and complete application-envelope bytes;
recipient-native payload tokens; every sender/receiver prompt and output token
by model/tokenizer; all attempted calls including failures/resumes; service,
end-to-end wall and critical-path time; parsing/decoding; observed CPU/GPU and
peak memory with measurement scope. Unknown usage is null with coverage counts,
never zero. Application bytes are not actual network traffic or ideal code bits.

Charge training/search, discarded candidates, validation selection, budget pilot,
capability calibration, onboarding/examples, shared dictionary/card transmission,
repairs, and preprocessing separately as setup. Count repeated card prompts in
per-call cost. Report total research expenditure plus per-arm attributable and
shared setup (shared costs once in total; disclose allocation rather than hiding
them). Report operational cost/episode and setup/H at H=64*n evaluation episodes
**per arm/cap**, with raw setup totals; this is a study horizon, not a deployment
lifetime. Additional H curves are descriptive. Cost per solved episode uses all
attempt costs and is undefined at zero successes. Lower bytes alone is no claim
of overall savings or inferential Pareto dominance.

The two core message arms × 3 caps × 16 sets × 8 calls/set plan **768*n** test
calls (384*n scored arm/cap episodes), before descriptive controls, pilot, search,
12 calibration calls, failures or resumes. Every message batch uses one set,
8 planned calls, under the existing 12-call ceiling. Before freeze, add each
chosen control's actual call plan and all setup attempts to the total ceiling;
do not launch an underbudgeted partial paired design. Heterogeneous tokens stay
in separate units. No compute, time, dollar or GPU feasibility has been established.

## Review gate and offline evidence

Before a future test: fill model/tokenizer/endpoint identifiers; A/B and scorer,
serializer, code, ontology and manifest hashes; independent calibration ledger;
training-search and validation seed lists/budgets; cap values and selection table;
test seed derivation/list and batch order; justified n/alternative/power report;
total attempts/time/resource ceilings; missingness/resume plan; and an immutable
freeze timestamp. **All are currently pending where no real pilot exists.**
Offline success below provides no permission for model execution.

From the repository root, with only Python's standard library:

```sh
python research/confirmatory_ood_synthetic_v0_1.py --self-test
python research/confirmatory_ood_synthetic_v0_1.py > /tmp/tacit-confirmatory-synthetic.json
diff -u research/data/CONFIRMATORY_OOD_SYNTHETIC_V0_1.json /tmp/tacit-confirmatory-synthetic.json
```

The [synthetic helper](confirmatory_ood_synthetic_v0_1.py) checks pairing, coverage,
range scaling, duplication/missingness rejection, and exact binomial operating
characteristics for assumed paired split distributions (including perfectly
correlated contrasts). It opens no experimental files. The committed JSON is
deterministic synthetic evidence only; it does not estimate model effects,
validate split independence, or establish sufficient sample size.

Observed offline verification: **9/9 checks passed** on Python 3.12.14; regenerated
JSON matched byte-for-byte. Across eight assumed distributions and n=4,12,20,100,1000,
maximum simultaneous noncoverage was 0.003518 and maximum probability of rejecting
any true practical null was 0.001834 (rounded upward). These are finite scenario
checks, not a proof over all populations. For X in {−1,+1}, P(X=+1)=.6, H1 rejection
probability was .003611 at n=20 and .966744 at n=1000. Making the *unclipped radius*
at most .05 requires 3,506 splits for D and 14,023 for J by this conservative
formula; these precision calculations establish neither power nor feasibility.
No model-backed cap, effect variance, independence, budget or sample adequacy was
verified. No offline check failed in the final run.

Design inputs: [v0.4 contract](../experiments/emergent_ood_v0_4/README.md),
[clustered audit](EMERGENT_OOD_CLUSTERED_ANALYSIS_AUDIT_V0_1.md),
[budget design](EMERGENT_OOD_BUDGET_FRONTIER_DESIGN_V0_1.md),
[INDEX screen](../experiments/index_v0_3/README.md), and
[Private Match registration](../experiments/private_match_v0_3/preregistration.json).
