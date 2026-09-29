# Emergent OOD transfer fixture v0.4

This standard-library-only artifact provides a higher-order meaning split and keyed, role-separated communication episodes. It is not a model runner, protocol, benchmark result, or language-performance claim.

## Meaning split

The default universe is `shape × color × quantity × texture`, with four values per attribute (`4^4 = 256` meanings). For each seed, the generator deterministically orders each axis using domain-separated SHA-256 hashes. A meaning is held out when the sum of its four permuted ranks is `0 mod 4`.

This yields 64 held-out four-way combinations and 192 training combinations. Every value combination of one, two, or three attributes appears in training. Exhaustive enumeration confirms 16 unary assignments, 96 pair assignments, and 256 triple assignments. The split digest and lower-order coverage are recomputed by the validator.

```powershell
python experiments/emergent_ood_v0_4/split.py --seed 17 --output .cache/emergent_ood_v0_4/seed_17.json
```

Omitting `--output` prints JSON to stdout. File output is rejected unless it stays inside the project. The split seed is public metadata, not a secret or a task key.

## Keyed communication episodes

Create a local evaluator-only 256-bit key once, then generate role ledgers:

```powershell
python -m experiments.emergent_ood_v0_4.episodes keygen --key .cache/emergent_ood_v0_4/evaluator.key
python -m experiments.emergent_ood_v0_4.episodes generate --key .cache/emergent_ood_v0_4/evaluator.key --seed 17 --task-seed 0 --k 4 --sets-per-stage 16 --output-dir .cache/emergent_ood_v0_4/episodes
```

The private key and generated episodes stay under ignored `.cache/`. HMAC domain-separated streams allocate 16 of the 64 held-out meanings to validation and the other 48 to test; training targets come only from the 192 training meanings. Each candidate set has exactly `k` episodes, one per candidate target, with a target-independent candidate table and ordering. Therefore the no-message Bayes accuracy is exactly `1/k` for every set. The manifest reports candidate co-occurrence coverage both on observed meanings and against the stage's full target support.

Output consists of separate `sender_{train,validation,test}.jsonl`, `receiver_...`, and evaluator-only `gold_...` ledgers. Sender rows contain only a private attribute tuple; receiver rows contain only candidate IDs and tuples. Model-facing rows have no episode, target, meaning, candidate-set, or stage identity. A trusted runner must preserve row order to align the ledgers. Never give a model access to the output directory, manifest, key, or gold ledger. Manifest file hashes support artifact integrity checks; they do not prove that evaluation was conducted correctly.

The default `sets-per-stage=16` is a small plumbing fixture (64 targets per stage), not a sample-size recommendation. Candidate sets and meanings are nested/repeated observations. Any inferential run needs a preregistered sample plan, more clusters, tool-boundary leakage checks, and a passing model-resource gate. Key and output paths outside the project are rejected.

## Evaluator runner

The runner uses the repository's protocol-neutral runtime and leaves model server startup to the operator. It verifies the split digest, all nine ledger hashes, role separation, target membership, and balanced candidate sets before any model request. It selects complete balanced candidate-set clusters, omits all evaluator IDs/stage labels from prompts, uses one sender-to-receiver turn followed by a sealed final answer, and writes raw results plus an evaluator-only manifest under the project.

Available conditions are `no_message`, `full_information`, `natural_language`, `json`, `symbolic`, and `shared_protocol_card`. The English arm is a plain baseline, not development-optimized natural language. The symbolic condition is a fixed four-axis digit code and a strong handcrafted control; it is not Tacit's learned language. `shared_protocol_card` evaluates a previously frozen sender/receiver instruction artifact using this JSON schema: `{"schema":"tlu.shared_protocol_card.v1","protocol_id":"...","sender_instruction":"...","receiver_instruction":"..."}`. The runner hashes and applies the card unchanged, but does not yet discover or evolve cards. The receiver must return an exact candidate ID. Malformed/truncated answers remain failures. Reported token totals are `null` when any call lacks provider usage, with per-call missingness preserved. The run records tokenizer/model IDs, generated versus delivered bytes, the full loopback application envelope, call/service/wall costs, sender-format audits, and an exact small-batch conflict-graph coloring floor. Temperature is fixed at zero; sender/receiver completion caps are 160/48 tokens.

Dry-run is the default and makes no model request:

```powershell
python -m experiments.emergent_ood_v0_4.runner --input-dir .cache/emergent_ood_v0_4/episodes --stage validation --conditions no_message natural_language
```

After a protocol has been developed on training episodes and frozen, evaluate it with `--conditions shared_protocol_card --protocol-card .cache/emergent_ood_v0_4/frozen-card.json`. Validation can select development choices; freeze the card before touching the test stage.

Execution requires a passing resource report no older than five minutes, obtained before starting the configured local endpoints. The runner only accepts loopback URLs and does not start the service:

```powershell
$env:TLU_SENDER_MODEL = "local-sender-model"
$env:TLU_RECEIVER_MODEL = "local-receiver-model"
$env:TLU_SENDER_TOKENIZER_ID = "sender-tokenizer-revision"
$env:TLU_RECEIVER_TOKENIZER_ID = "receiver-tokenizer-revision"
python experiments/emergent_ood_v0_3/resource_preflight.ps1 -Output .cache/emergent_ood_v0_4/resource_preflight.json -Ports 8000,8001
# Start the local model endpoint only after a passing report, then:
python -m experiments.emergent_ood_v0_4.runner --input-dir .cache/emergent_ood_v0_4/episodes --stage validation --conditions no_message natural_language --sender-base-url http://127.0.0.1:8000/v1 --receiver-base-url http://127.0.0.1:8001/v1 --execute --resource-preflight .cache/emergent_ood_v0_4/resource_preflight.json
```

The runner computes the exact chromatic number only for batch graphs with at most 20 observed meanings; larger graphs are marked non-exact rather than approximated as a theorem. The full-information arm is currently descriptive; there is no disjoint capability-calibration artifact, so it must not be used to filter evaluation episodes. Validation results are development data and test-stage protocols must be frozen before they are run. The 12-call hard batch cap is a feasibility safeguard, not a confirmatory-study design. No protocol superiority claim follows from this runner.

Run focused model-free checks:

```powershell
python -m unittest discover -s tests -p "test_emergent_ood_v04_*.py" -v
```

## Scope and research role

The task asks a sender to communicate a private meaning so a receiver can select its match from a balanced candidate set. Training exposes lower-order combinations while validation/test targets are disjoint held-out compositions. The proposed CLSR-inspired method may induce a reusable dialect from training episodes, select/profile only on validation, then freeze before final evaluation.

The artifact provides a first strict task scorer and local runner, but no learned dialect, protocol-induction workflow, disjoint receiver-capability calibration, cross-model transfer experiment, or performance evidence. Canonical-label fidelity for English is a conservative string audit, not a general semantic judge. See the [CLSR transfer experiment design](../../research/CLSR_TRANSFER_EXPERIMENT_DESIGN_V0_1.md) and [CLSR prior audit](../../research/CLSR_AUDIT_V0_1.md).
