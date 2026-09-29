# Emergent OOD split v0.4

This is a standard-library-only **meaning split generator**, not a task runner, model experiment, or benchmark claim. It creates a higher-order composition split for planning a future receiver-side LLM communication study.

## Default split

The default meaning universe is `shape × color × quantity × texture`, with four values per attribute (`4^4 = 256` meanings). For each seed, the generator obtains an independent deterministic ordering of each axis's values using domain-separated SHA-256 hashes. A meaning is held out when the sum of its four permuted ranks is `0 mod 4`.

This yields 64 held-out four-way combinations and 192 training combinations. Fixing any assignment to one, two, or three attributes leaves every value combination represented in training. A direct exhaustive check of all 256 meanings found the expected split sizes and coverage counts: 16 unary assignments, 96 pair assignments, and 256 triple assignments. The generator recomputes these coverage conditions and a canonical SHA-256 split digest before returning a split.

Generate a split to a project-local cache path:

```powershell
python experiments/emergent_ood_v0_4/split.py --seed 17 --output .cache/emergent_ood_v0_4/seed_17.json
```

Omitting `--output` prints JSON to stdout. Any file output is resolved and rejected unless it stays inside the project directory. The split seed is public metadata, not a secret and not suitable as a task key for tool-enabled model agents.

Run the focused model-free checks:

```powershell
python -m unittest discover -s tests -p test_emergent_ood_v04_split.py -v
```

## Scope and research role

The split exists because v0.2 has only nine held-out meanings, too few to cleanly separate protocol induction, selection, and evaluation. The proposed use is a one-message sender/receiver task: sender sees one private held-out meaning; receiver sees a target-balanced candidate set drawn from a sealed evaluation subset. The proposed CLSR-style adaptation should induce/evolve its reusable dialect on training-only communication episodes, select/profile on held-out validation meanings, then freeze it before final evaluation.

This implementation only creates and validates the tuple partition. It does not generate role-separated communication episodes, supply a receiver scorer, implement any protocol, or establish task success, language quality, communication necessity, or superiority. Candidate sampling, leakage control, clustered inference, message accounting, and model-resource gates remain future work. See the [CLSR transfer experiment design](../../research/CLSR_TRANSFER_EXPERIMENT_DESIGN_V0_1.md) and [CLSR prior audit](../../research/CLSR_AUDIT_V0_1.md).
