# Development runner cost adapter

Status: offline tested connection candidate, 2026-09-30. No real model requests,
model qualification, protocol advantage or confirmation release. Preserve the
session's manually configured model; this package does not manage ChatGPT models,
reset cards, GPUs or services.

## Current state and smallest gap

Main is `5dd260c5824b280b5323a1707ab132524afa8e4c`. This branch intentionally stacks
on PR #7 fixed head `e840eb7e392f16fb5f5932625fe871ae41274463`; target PR base is
`codex/tacit-m2-offline-gates`, not main. PR #7's two original P1 races were lifted
by [independent affected-head review](https://github.com/TTAWDTT/Tacit/pull/7#issuecomment-5914215750)
within offline ledger/snapshot scope. That review did not qualify any model.
No AGENTS.md or .agents/skills exists in the repository or workspace instructions
paths checked. Existing runner/scorer/protocol source is unchanged.

The missing connection was between per-call complete prompts/native usage and
Ledger reservations. `GuardedChatModel.complete` now implements the existing
ChatModel interface; `run_development_episode` invokes the unchanged OOD
`run_condition`. Each sender and receiver request gets its own durable reservation
and receipt; no hidden two-call callback. Full sender/receiver instructions, card,
examples if present, candidate context and transcript travel through the tokenizer
and serializer before dispatch. `NativeCounters` uses an already-loaded local
tokenizer's actual chat template and generation prefix, with thinking disabled;
it never downloads or loads a model. It rejects a template hash mismatch.

This is an additive bridge, not a new experiment runner/platform. It does not
implement endpoint discovery, automatic optimization or session management. Full
paired candidate scheduling remains the existing experiment design/controller's
responsibility. No execution CLI is provided. It deliberately cannot read test
files and rejects test-stage provenance. Trusted setup must already validate role
ledgers, keys, candidate tables and development split manifest; the bridge checks
sender row membership and sender/receiver/stage/contract binding. Split support
checks reject attaching seed23 metadata to seed17 rows. It is not a sandbox for
forged trusted manifests or arbitrary card contents.

## Accounting and preserved failures

`Contract` requires literal-loopback HTTP with no auth/redirect/remote destination,
pinned model identity, tokenizer/template hashes, output cap, response read cap,
socket timeout, and explicit verified claims of free local operation, statelessness,
disabled reasoning and server completion-cap enforcement. False/unknown claims
block client construction. A caller checking boxes is not evidence: an independent
server/configuration audit is still necessary before a real call.

The exact compact JSON request is persisted before dispatch. The same immutable
bytes are sent to the transport. Input native tokens and the complete HTTP JSON
request body are recurring context; output tokens, response JSON bytes and one
model request are work. Native token axes include role + full endpoint contract
hash; sender/receiver/model token counts are not converted into equivalent compute.
Every policy axis must be supported: current-role native input/output, shared
request/response-body bytes and model_requests, plus only the opposite role's
native axes from the same Contract or an explicit peer_contract. Undeclared peer
contracts and any other axes (including provider_generation_seconds,
local_compute_cost, fee or arbitrary token axes) reject before reservation/POST.
The other role's zeros mean this single POST makes no call to that role, not that
its endpoint is free. Observational timing and unknown compute costs cannot be
promoted to enforced budget dimensions. All supported policy axes appear in each Charge. Ledger upper bounds remain conservative
and non-refundable; actual usage is separate. Raw HTTP responses are stored as
bytes, including malformed/error responses. Incomplete chunked reads preserve bounded
partial response-body bytes, HTTP status and an explicit truncation flag, while
usage stays unknown and the ledger halts. Response headers/credentials are not
stored. New observational columns migrate an existing receipt table without
resetting reservations or old receipts; missing historical metadata stays null. There is no retry. Missing usage,
model/template mismatch, hidden reasoning, redirects, response read overflow,
transport errors or output overrun halt the ledger with its reservation intact.
An observed token overrun retains observed actual values; invalid/untrusted usage
has unknown actual cost, with raw evidence retained.

SQLite `model_receipts` stores full request, request hash, endpoint contract,
raw response, observed usage/timing/finish reason and errors. `episode_results`
stores started/completed/error state and the unchanged strict runner result.
Both are evaluator-private: their prompts/results can contain private meanings,
candidate tables and gold. Never expose the database or exported receipts to models.
A failed partial episode cannot silently resume or rerun. Ledger status success
means **valid cost receipt** in this adapter; it is not strict task success. The
runner's outcome is the authoritative strict result, including valid wrong IDs
and malformed answers. Do not pass adapter receipt rows to Scope.freeze as if they
were its one-strict-outcome-per-development-row records; their shapes differ.
A reviewed candidate-level join/controller remains needed for optimized selection.

Protocol cards and onboarding examples must be measured on EVERY actual request.
Discovery, failed candidate search and deployment remain separate charged events
in the same fixed ledger, inherited from #7; they are not inferred zero because
this bridge supplies only inference receipts. The result explicitly marks setup
accounting unresolved. Report discovery-inclusive and artifact-reuse views; never
claim a complete-cost frontier from inference receipts alone.

UTF-8 metrics cover HTTP JSON bodies and the existing runner's complete application
message envelope. HTTP headers/TCP bytes are excluded, not asserted zero. Wall
latency is observed; provider generation timing is retained if supplied. External
API fee is zero only under the separately confirmed free-loopback premise. Local
energy/compute opportunity cost and unobserved cache/reasoning details remain null.
Socket timeout bounds each I/O wait; it is not a wall-time or server GPU hard limit.
The response read cap protects client memory, not generation. Server generation
must enforce max_tokens and disable thinking. A timed-out server may still work;
operator shutdown/resource controls are required. No total time/GPU/fee safety
certification is claimed. Token counting agreement is audited against provider
usage; discovering a discrepancy after a first call cannot undo that spending.

## Reproduction

```sh
python -m unittest tests.test_runner_cost_adapter -v
python -m unittest tests.test_runner_cost_adapter tests.test_m2_gates tests.test_m2_snapshot_races tests.test_emergent_ood_v04_runner -q
python -m compileall -q experiments/runner_cost_adapter
```

Tests use invented counters and fake completions, including an actual local HTTP
server that serves JSON without loading a model. They verify reserve-before-POST,
full prompt/card context, native receipt mismatch, strict format/valid-wrong-ID
failures, generated-but-undelivered message costs, disabled-reasoning violations,
HTTP redirect/error handling, unknown usage, output overrun, durable exceptions,
role visibility, split/test rejection and integration with the original scorer.
No simulated success is evidence for TACIT versus NL.

## Available resources and proposed next sample

Scoped check in this Tacit environment: TLU base/model variables are unset; `ss -ltn`
shows only port1040 and no usual model-server port. No model endpoint was queried,
model weights inspected or other project/session accessed. This establishes no
known reusable model service here; it does not establish absence on the user's
local computer. User/parent will designate any free endpoint and existing cache.
No download, paid API, rental compute or reset card is needed for this delivery.

Proposed staged preparation, not execution permission: first INDEX_m maximum12
requests under its existing 4-FULL→8-message gate; then independent OOD12/12
qualification on a separate calibration split. After server/template/usage and
resource checks, an initial single **training-only** balanced four-target set
crossing three predeclared cards (concise NL, compact fields, reviewed TACIT/card
candidate) uses 12 episodes / at most24 requests when both roles are model-driven.
Split that into two batches ≤12 and preserve one fixed cumulative ledger.
No sender-oracle saving may be called a learned-protocol gain. FULL/no-message
qualification/reference requests are separate and charged. This plumbing pilot is
not confirmation and does not select a protocol using test feedback.

A subsequent paired train+validation comparison over two independently generated
and isolated development splits would be 2 splits × 2 stages × 4 balanced targets
× 3 cards × 2 role calls =96 requests before qualification/failed setup/search.
This larger number is disclosed, not silently authorized. Treat split as the
uncertainty unit; two splits still cannot establish a reliable effect. Keep cards
and model/template versions frozen, seed schedules paired, and all failures.
Optimized concise NL, JSON and stronger compact cards need equal development/search,
example and inference budgets before a superiority comparison. The existing route A
compact-equivalence negative result stands: relabeling its validator as TACIT is
not a distinct representation arm. Candidate content and meaning coverage must
be reviewed before even the tiny plumbing pilot; no favorable invented language
is supplied here.

## Remaining blockers before a credible real pilot

1. Parent/user-designated free endpoint, actual model/config and tokenizer files,
   verified server template/thinking/output limits, stateless behavior and fresh
   resource/independent receiver qualification. Contract flags are not that proof.
2. NativeCounters tested on that already-installed tokenizer/server with exact
   hashes; reviewed full input/output ceilings and server/process cancellation.
3. Trusted keyed role-file setup/provenance, fixed ledger path, all discovery/setup
   charges and candidate-level strict-result join; search controller must count
   actual per-batch ≤12 model calls and enforce approved cumulative limits.
4. Fair distinct candidate/card arms and matched train/search/onboarding budgets,
   primary complete-cost axis, binding budget ladder, reuse N and statistical plan.

The shipped bridge closes the concrete HTTP/request/receipt/result connection.
It does not approve the remaining scientific, resource or controller decisions.
No main merge, force push, model experiment or service launch is part of this PR.

## Publication evidence and library boundary

Author validation: 18/18 new adapter tests; combined adapter + PR7 gates/races +
unchanged OOD runner tests 81/81 (18+37+26). These counts overlap, are software tests,
and are not independent scientific observations. compileall and diff whitespace
checks passed. Full repository suite was not run. A real HTTP fake service was
used only in a test; no endpoint serving a model was contacted.

`episode_prefix` includes the card and wire-budget hashes so paired cards/cap points
on the same row have distinct identities; a repeat of the same intervention is
rejected. `export_evidence` exports all ledger attempts, raw receipts, and partial/
completed/error episodes, preserving failures. Unknown costs stay explicitly
unresolved. NativeCounters itself is tested with a tokenizer stub only; no installed
real tokenizer/server equality has been demonstrated.

The library bridge does not replace the frozen runner CLI's resource/capability
checks or assert they were invoked: its tests call the low-level run_condition,
just as existing unit tests do. Live use requires the parent to validate a fresh
resource report and independent qualified receiver ledger under the exact contract
before entering this library, with an approved per-batch/cumulative schedule.
The contract currently binds the named model and tokenizer/template plus endpoint
and verified server assumptions; it does not independently prove server weight
revision, max input context, server cancellation, statelessness or free billing.
No passing booleans or HTTP mock results count as that verification. Keep draft
until independent adapter review; there is no model-execution release here.

## Independent review corrections

Original published head `79db169bce30f22707f7d42319b3f44d5e3c37a1` accepted arbitrary
Policy axes and filled unhandled dimensions with zero. A zero cap on provider
seconds accepted an observed five-second response; unknown compute cost likewise
became a false ledger zero. The revision rejects these unsupported axes before
POST rather than claiming to implement timing/compute hard bounds. Only declared
peer-native token axes may be zero for a call exclusively to the current role.

The original chunked-HTTP failure halted but lost IncompleteRead.partial. The
revision retains bounded response-body evidence, status200 and truncation for
`abc` without a final chunk, while actual usage remains unknown and no retry
POST occurs. A mocked overlong partial is clipped to the response limit. Endpoint
headers/credentials are excluded from partial evidence. Existing receipts migrate
in place without a ledger reset.

Both targeted tests failed at the original code before fixing it (all three
unsupported-axis subcases entered the fake transport; incomplete response was
null). Added five test methods cover these failures, explicit distinct peers,
partial-byte bounds and legacy receipt migration. Revised author validation:
23/23 adapter tests, 86/86 combined (23+37+26), compileall and whitespace checks.
This is author evidence pending fixed-head independent re-review. The review also
confirmed six related cumulative/failure/no-retry/template/receipt checks; existing
tests continue to pass. No real model call or complete endpoint qualification is
claimed. CI status is reported separately in the PR, not inferred from tests.
