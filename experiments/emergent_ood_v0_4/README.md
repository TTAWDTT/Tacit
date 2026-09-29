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

The private key and generated episodes stay under ignored `.cache/`. HMAC domain-separated streams allocate 16 of the 64 held-out meanings to validation and the other 48 to test; training targets come only from the 192 training meanings. Each candidate set has exactly `k` episodes, one per candidate target, with a target-independent candidate table and ordering. Therefore the no-message Bayes accuracy is exactly `1/k` for every set.

Output consists of separate `sender_{train,validation,test}.jsonl`, `receiver_...`, and evaluator-only `gold_...` ledgers. Sender rows contain only a private attribute tuple; receiver rows contain only candidate IDs and tuples. Model-facing rows have no episode, target, meaning, candidate-set, or stage identity. A trusted runner must preserve row order to align the ledgers. Never give a model access to the output directory, manifest, key, or gold ledger. Manifest file hashes support artifact integrity checks; they do not prove that evaluation was conducted correctly.

The default `sets-per-stage=16` is a small plumbing fixture (64 targets per stage), not a sample-size recommendation. Candidate sets and meanings are nested/repeated observations. Any inferential run needs a preregistered sample plan, more clusters, tool-boundary leakage checks, and a passing model-resource gate. Key and output paths outside the project are rejected.

Run focused model-free checks:

```powershell
python -m unittest discover -s tests -p "test_emergent_ood_v04_*.py" -v
```

## Scope and research role

The task asks a sender to communicate a private meaning so a receiver can select its match from a balanced candidate set. Training exposes lower-order combinations while validation/test targets are disjoint held-out compositions. The proposed CLSR-inspired method may induce a reusable dialect from training episodes, select/profile only on validation, then freeze before final evaluation.

The artifact does not provide a receiver scorer or model adapter, validate model isolation, or establish task success, language quality, communication necessity, or superiority. A runner still needs frozen prompts and call schedule, semantic-fidelity metrics, complete channel and inference cost accounting, cross-model transfer tests, and a resource-gate pass. See the [CLSR transfer experiment design](../../research/CLSR_TRANSFER_EXPERIMENT_DESIGN_V0_1.md) and [CLSR prior audit](../../research/CLSR_AUDIT_V0_1.md).
