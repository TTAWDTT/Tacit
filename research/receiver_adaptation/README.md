# Receiver adaptation: bounded selection and complete-cost break-even

Status: first-round research candidate, offline only, 2026-09-30. No LLM evidence.
[Execution contract](https://chatgpt.com/space/page_bd3d1298bd5c81919e4101089077ae2e).
Actual base/main at inspection: `5dd260c5824b280b5323a1707ab132524afa8e4c`.
Head is recorded in the PR, avoiding a self-referential commit hash here.

## Findings and decision

Continue only as a receiver-conditioned **selection hypothesis**, not a new language
claim. Existing `experiments/emergent_ood_v0_4/induce_protocol_cards.py`,
`nl_feedback.py`, `select_nl_search_parent.py`, `select_protocol_frontier.py`, and
`generate_usage_examples.py` already cover train-only proposals, exact-feedback NL
search, validation selection, and acquisition. Reimplementing these is not novelty.
This extension supplies a small inspectable selector and exact cost inequalities;
it neither replaces those production provenance checks nor adds a runner condition.
No dependency on unmerged infrastructure PRs #1/#2/#3 or route A is required.

The strongest counterexamples are (1) a receiver already decodes all representations
perfectly, so adaptation cannot improve success and incurs extra cost; (2) receiver
R2 interprets every selected label through a derangement, so a perfect R1 mapping
fails on R2; (3) repeated onboarding context exceeds payload savings, so no reuse
horizon recovers discovery cost. These are existence proofs, not population rates.

## Mechanism and allowed feedback

Let m be sender-visible task information, p a frozen card plus primitive bijection,
e_p(m) the message, and R_r the fixed receiver/configuration. The hypothesis is that
choosing p conditional on **development** feedback from r reduces strict decoding
errors relative to an equally searched receiver-independent choice. It is not a
claim of reduced information-theoretic payload requirements. Private Match v0.3
still requires each source to distinguish q coordinates for zero error; renaming
coordinates cannot defeat its 2 log2(q) fixed-width payload bound. Emergent OOD v0.4
still uses its 192 training / 64 held-out combinations and keyed validation/test
subdivision, with full candidate tables visible only to the receiver.

Candidate inventory, proposal rule, public ontology/priors, maximum query/call/token
budgets, explanation/example budgets, and seeds must be declared before validation.
Allow training strict success/failure, syntax validity, observed cost and generated
messages. Invalid, rejected and truncated responses remain failures. Never supply
gold candidate IDs, candidate-table order, hidden receiver information, evaluator
keys, episode names, or complete held-out tuples to the proposing sender/card.
Training feedback may refine proposals within the registered bound; validation
only ranks a completed inventory, never creates another candidate. The minimal
reference deliberately uses a fixed inventory with no adaptive proposal loop.

`select` requires complete paired train and validation observation grids for one
receiver, charges all candidates (including failed ones), and freezes the card,
primitive mapping, inventory/feedback/support hashes, selection rule and costs.
Ranking is **unconstrained** validation strict-success count, lower validation cost
on one declared axis, then stable name. The v2 freeze explicitly records
`selection_scope=unconstrained_inventory_ranking`, `budget_feasibility=not_evaluated`
and `deployment_authorized=false`. This is not a budget-constrained selector: a
100-unit successful candidate still outranks a 1-unit failed candidate, even when
a future deployment cap would be 10. The regression preserves this limitation
rather than silently calling the output feasible. `max_candidates` caps inventory
size only, not actual queries/tokens. Candidate-specific deployment/repeated-context
costs, reuse horizon and complete-cost feasibility require a separately reviewed
stage before M2; this PR does not supply or authorize that stage. Train scores are audited/charged, not counted as evaluation.
A failure must be an explicit false row, not an omitted candidate. Deployment uses
the frozen artifact unchanged; `verify_freeze` detects edits. Reuse does not mutate
it. A hash proves integrity, not truthful provenance or semantic correctness.

The reference accepts trusted in-memory observations and split membership. It does
not read files or prove that callers labeled supports honestly. Before real use,
reuse the existing role-ledger/manifests and selector audit; no model-facing adapter
is delivered here. A card can contain arbitrary text; the code does not certify
that it is leak-free. There is no simulated positive adaptation experiment: supplied
scores test bookkeeping only. The mapping decoder tests are mathematical controls,
not learned receivers. Thus designing a decoder to like our code cannot count as
confirmatory evidence.

## Exact cost result and limits

For one comparable, predeclared axis, let D include all reproducible discovery,
failed proposals, train/validation evaluation, selection, and card generation; U
include deployment acquisition/installation; c include each actual execution,
repeated context, inference, retries, invalid responses, framing and final scoring.
Then C(N)=D+U+Nc and average C(N)=(D+U)/N+c for positive integer N. Record historical
research exploration separately; additionally report discovery-inclusive and
previously-frozen-artifact deployment views without silently dropping D.

For adaptive A vs comparator B, set F=(D_A+U_A)-(D_B+U_B), s=c_B-c_A. A is no more
costly exactly when F <= Ns. If s>0, first N is max(1,ceil(F/s)); if s=0 it is all
N when F<=0 and never otherwise. If s<0 it holds only through floor(F/s), provided
that bound is at least 1: a temporary advantage, not eventual amortization. The
implementation uses rational arithmetic and includes equality, so strict advantage
at an exact equality starts one use later. Unknown fields produce unknown results;
incompatible token axes raise errors. No scalar conversion of different tokenizers
is implied. Apply inequalities separately per axis; vector dominance requires their
intersection AND appropriate success evidence.

Example, explicitly invented units: D_A=100,U_A=20,c_A=6 vs D_B=U_B=0,c_B=10.
At N=1,10,30,100,1000 A averages 126,18,10,7.2,6.12 vs B=10. Equality at 30,
strict saving from 31, limit 6. Add 5 units repeated context to A: c_A=11 and no
break-even exists. A new receiver with U_A increased by 80 moves equality to 50
if per-use costs remain unchanged; with accuracy collapse, that cost equality is
irrelevant to a success frontier. Numbers are demonstrative, never measured.

This affine result assumes stable workload, cost per use, cache/reset policy and
receiver. Drifting costs, failures, rate limits or tiered prices require direct
cumulative ledgers. Cost-per-success ratios are secondary and require positive
success probability; they do not replace strict success under a budget. Success
improvement alone cannot prove domination. For uncertain F and s, propagate paired
split estimates; if saving intervals include zero, there may be no finite supported
threshold. Never report only a favorable point estimate of N.

## Reproduction and deliverables

From repository root, standard library only:

```sh
python -m unittest experiments.receiver_adaptation.test_reference -v
python -m experiments.receiver_adaptation.demo
```

`reference.py` supplies immutable primitive candidates, deterministic freezing,
mapping derangement, full selected-axis costs and integer break-even intervals.
`test_reference.py` includes 21 tests and 8,000 direct inequality checks.
`demo.py` prints the illustrative reuse table and counterexample, without writing
artifacts or contacting any endpoint. `EXPERIMENT_SPEC.md` states the unresolved
real-experiment freeze requirements. No existing scorer/runner/protocol is edited.
The unit suite is explicitly invoked because it lives inside this isolated module,
outside the repository's default tests directory. Cross-review awaits route A head.

## Fixed-head cross-review response

Agent A reviewed original head `c6e0e3e80b43523b78aa273e0a6c6f9005827042`
in [PR comment](https://github.com/TTAWDTT/Tacit/pull/4#issuecomment-5912927099).
Both findings reproduce. Red→a, blue→bbbb over [red,red,red,blue] costs 7 bytes;
changing the sender mapping costs 13. Dictionary label preservation is not episode
message preservation. `decoder_corruption_control` now constructs identical sender
message sequences and changes only receiver reference associations. It guarantees
payload identity, not equal receiver-card tokens, reasoning or complete cost.
The second finding is an explicitly retained design limitation: inventory ranking
is not constrained optimization. Versioned freeze metadata and a counterexample
test prevent the current artifact being described as budget qualified. No external
consumer of this experimental schema is registered in the repository.

Independent review of route A at `b438b8d07aae5e148b223141203599ccd6eef54c`
reproduced 8 new plus 38 existing tests and the stored falsification report. All
768 tuples match compact fields, so the narrowed validator interpretation stands.
No route A changes are included in this branch. Review comments pin exact heads;
revised B needs affected-item re-review before M2.
