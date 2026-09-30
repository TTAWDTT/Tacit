# Pending experiment specification: instructions and validation, not a new code

Status: design candidate for review; **not execution-ready or confirmatory**.
No paid call, model loading, download or external GPU is authorized by this file.
The minimal model-free experiment is `python -m experiments.compositional_protocol.falsify`;
its prediction is exact equality to the existing compact baseline and its
falsifier is any mismatch. Equality rejects the new-expressivity/wire-saving
claim. Do not expand the framework to recover a positive result.

## Remaining hypotheses and identifying contrasts

A1 (narrowed): on valid held-out v0.4 tuples, an explicit public domain/scope
instruction card reduces strict task error versus the generic compact card at
matched *complete* budgets for a qualified fixed receiver. Primary direction:
paired split-mean strict success difference > 0 under a common binding budget.
Require comparison against compact fields with the **identical** domain/scope
instruction. No additional benefit against that arm means the effect belongs to
instructions, not a new representation. Nonpositive differences contradict the
directional hypothesis; intervals spanning zero remain inconclusive. No arbitrary
five-percentage-point success criterion is adopted.

A2 (diagnostic): public-domain validation rejects out-of-domain/wrong-scope
messages more often than syntax-only parsing on a frozen, equal corruption set.
This is a software claim, already deterministic here. Invalid but well-typed
substitutions should survive it. Do not promote this diagnostic to task accuracy;
rejection counts as a failure and costs remain charged. Apply the same validator
and same rejection policy to compact, NL and JSON through equivalent parsers.
No retries/repairs without a separately budgeted and frozen condition.

A3 (secondary): factorized conventions may yield more successful unseen-tuple
transfer than arbitrary whole-tuple lookup learned only from training examples.
This is already a compact/symbolic baseline property. Report separately (a)
training-only usage onboarding and (b) a full schema/codebook card supplied at
setup. Under (a) unseen holistic assignments are unidentifiable without a prior;
its failure cannot identify a new mechanism. Under (b) charge the full lookup
card and use the same explanation/search limits; unsupported oversized cards are
infeasible rather than silently truncated. A compact-equivalent result or loss
under equal complete cost is an acceptable negative result.

## Arms, budgets and factor separation

| Arm/control | Information and purpose |
| --- | --- |
| FULL and no message | All facts/none; Private Match also x-only and y-only; independent qualification is not evaluation filtering |
| Raw NL, short NL, development-selected NL | Same exact facts; equal train data, validation selection count, instruction and inference budget; preserve rejected search candidates and costs |
| JSON, compact generic-v3 / compact_kv | Same records and transport envelope; strong existing formats |
| Typed reference and compact + same typed card | Identical valid payloads; isolate domain instructions/checking rather than rename a format |
| Fixed positional code and explicit compositional codebook | Factorization with and without overt labels; mapping and order disclosed equally |
| Arbitrary whole-tuple lookup and random relabeling | Equal meaning coverage; separate train-only usage from full-map setup; freeze random mapping seed before outcomes |
| Remove domain checking / remove labels but keep order | Type-validation and positional-binding ablations; neither secretly loses information |
| Remove labels and order; shuffle correct mappings | Explicitly lossy or incorrect diagnostic controls, not competitive baselines |
| Route B cross if independently reviewed | A/B × adapted/unadapted with the same search budget; do not conflate adaptation and serialization |

Deterministic sender and LLM sender are separate experiments. This reference
uses only public schema plus own facts, no candidate IDs, other sender facts,
metadata or test-specific rank. Any later information selector requires matched
NL/JSON serialization of exactly its selected facts. Keep recipient topology,
message count/order, output contract, temperature, completion caps, examples and
receiver reasoning budget fixed. Charge explanations even when public schema
is shared; compare the same schema access in every arm. Do not give A free
executable decoding while requiring baselines to decode using an LLM.

## Data, qualification and freezing

Use existing v0.4 schema/split and evaluator-keyed role ledgers. Within every
independent split seed, use training only for discovery; keyed held-out validation
selects candidates/budgets; test stays sealed. Existing default train/held-out
sizes are 192/64, with validation/test 16/48. Assert atom coverage and absence of
complete test tuples from training examples. Private Match v0.3 keeps its own
uniform-coordinate generator and disjoint calibration/development/evaluation;
do not transplant the v0.4 split claim to it.

Before inference: independently qualify INDEX_m (4/4 FULL before 8 communication
calls, total <=12), v0.4 receiver (separate 12/12 FULL ledger), and Private Match
under its frozen calibration. Retain each track's exact condition and scorer;
none is a universal effect threshold. Strict malformed/truncated outputs fail;
trailing-ID extraction is diagnostic only. Fixing an interface requires new
independent qualification, never retrospective conversion of failures to passes.

Freeze protocol/card and prompt hashes; actual base/head; generator version;
model revision, tokenizer hash, endpoint configuration; split/episode/model and
mapping seeds; all arms and candidate counts; examples; public schema; trusted
scope routing; cache/session/reset policy; role-visible fields; output limits;
cost axes; resource/call ceiling; statistical specification. Programmatic audits
must show only sender-private fields reach its encoder and no keys, gold, stage,
file names, episode IDs, candidate IDs or manifest details reach sender prompts.
The current codec's exact-scope checks are only one part of that audit.

Do not run the old runner with an unregistered new condition. Any integration
requires a separately reviewed extension and hash checks of frozen sources.
No unmerged infrastructure PR is required for this offline package. Real future
work may require #1's launcher and #3's statistical design after review; that
would be an explicit pinned dependency, not an implicit copy.

## Complete cost ledger (per request and deployment)

Preserve raw records, with unknown observations represented as null plus a reason:

- identity: deployment/run/request IDs, stage/role/arm, timestamp, protocol/card/
  prompt/content hashes, generator and model/tokenizer hashes, split/episode/model
  seeds, candidate-selection provenance, generation/delivery status;
- fixed research discovery and reproducible deployment costs in separate columns:
  searches, every failed candidate, card/codebook generation, calibration,
  onboarding/examples, tool calls, retries and rejected/invalid completions;
- per-call input/output UTF-8 bytes, generated and delivered payload bytes,
  serialized application framing and acknowledgement bytes, final answer/scoring
  costs, sender encoding and receiver reasoning; per-endpoint native input/output
  tokens, cached tokens, observable reasoning usage, inference/tool call counts,
  monotonic end-to-end and service latency, actual fee/currency when available;
- repetitions: setup artifact hash, actual distribution count, context repeated
  each call, cache hit/miss and reset policy, reuse horizon N, rejected delivery,
  truncation, semantic correctness, strict legality and fidelity separately.

Avoid double counting repeated setup bytes as both a one-time inference charge
and per-call input tokens. Distribution bytes and repeated prompt bytes remain
different dimensions. Unknown reasoning is not zero. Different tokenizer tokens
are not interchangeable compute. Primary feasibility axis proposed: complete
application wire bytes with all other cost dimensions reported; final choice
must be frozen following validation, not selected on test results. No weighted
single cost score without justified preregistered conversion rates.

Report C_j(N)=F_j/N+E_j for each cost dimension j at N=1,10,100,1000 and the
long-run limit, with raw totals. Report no break-even where E_A >= E_compact and
F_A > F_compact in that dimension. This N grid is a proposed fixed display grid,
not evidence of a real deployment horizon. The offline equality check supplies
only E_payload equality, not F or complete E. Repeated instruction overhead does
not vanish at large N under stateless resets.

Validation must establish one loose cap and at least two genuinely binding common
caps from full serialized lengths (not payload lengths). Freeze cap values and
selection rule before test. Mark FULL or other arms infeasible where they exceed
caps; retain all generation costs and failures. Never interpolate infeasible
points or compare only average successful-message length.

## Inference, transfer and stopping

Primary paired contrast: typed instruction arm versus strongest matched compact
instruction arm, per receiver and budget, averaged within split then across
independent split seeds. Candidate-set/episode repetition is nested within split.
Use paired cluster bootstrap resampling whole independent splits for intervals;
choose split count, bootstrap replication count and precision/power target from
validation plus approved budget before test. Do not treat the three offline
fixture seeds as a powered sample. Predeclare the primary receiver/budget and
use Holm correction for any family of additional confirmatory contrasts;
otherwise label them exploratory. Missing, refusal and malformed responses are
failures with observed costs retained, not dropped pairs. Freeze handling of
transport-only missingness and unknown cost dimensions before launch.

Separate held-out combinations from a held-out receiver not used in discovery.
For each receiver report zero-onboarding, a fixed card, and budgeted train-only
examples separately; adaptation after seeing new receiver feedback is not
zero-shot. Freeze the protocol before testing the new receiver; report its own
qualification and costs. Cross-ontology claims require new task definitions and
independent evidence; this reference supplies none.

Stop upon qualification, role-isolation, hash, ledger or resource failure; stop
at approved call/cost/sample limits, not significance. No sequential early stop
without a separately preregistered corrected procedure. If the compact-equivalence
control accounts for all benefit, stop the new-language claim and report the
instruction/validation result only. If intervals are inconclusive and budget
ends, report uncertainty. Never extend samples until significance.

Remaining blockers: unselected qualified receiver/tokenizer, no measured budget
ladder, no validation variance/sample-size justification, no approved real-call
budget, no frozen prompts/model manifests or integration, and outstanding fixed-
head cross-review. These are explicit M2/M3 gates, not completed experiments.
