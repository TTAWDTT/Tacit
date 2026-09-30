# Message-association batch feasibility v0.1

**Status:** model-free structural screen on one generated evaluation bundle. No model outputs were inspected or generated. This is an exploratory batch-feasibility result, not confirmatory evidence and not a communication result.

## Question

Can the default v0.4 test episodes be partitioned into complete three-candidate-set batches such that a donor message can be assigned to every recipient while (a) every message is used exactly once, (b) no message stays with its source episode, and (c) a donor's private meaning is absent from the recipient's entire candidate table?

This matters because the receiver-only replay is capped at 12 calls per batch, with `k=4`; three candidate sets make 12 receiver episodes. If feasible offsets were selected after viewing model outcomes, the replay would be a selected control. The structural screen below uses only role-separated task inputs and is frozen before any sender or receiver model result for this bundle.

## Exact procedure

For each receiver episode `r` and donor episode `d`, add an edge iff `r != d` and `meaning(d)` is absent from all candidates shown to `r`. A complete compatible assignment is a perfect matching in this graph. Count such matchings exactly with the replay implementation's subset dynamic program; a zero count means the window is infeasible.

The public generator can create a 12-set bundle under a local evaluator key:

```powershell
python -m experiments.emergent_ood_v0_4.episodes generate `
  --key .cache/emergent_ood_v0_4/evaluator.key `
  --seed 23 --task-seed 0 --k 4 --sets-per-stage 12 `
  --output-dir .cache/emergent_ood_v0_4/evaluation-23-12sets
```

Keep the key, gold ledger, and generated bundle local and evaluator-only. Bundle generation reads the local key; the subsequent feasibility analysis does not reread it. The bundle loader validates all role ledgers for integrity and alignment; the compatibility graph itself uses only sender tuples and receiver candidate tables, with evaluator IDs solely to join records. It does not use target labels, inspect messages, or access model scores.

Recompute and print the structural scan with:

```powershell
python research/derangement_batch_feasibility.py `
  --input-dir .cache/emergent_ood_v0_4/evaluation-23-12sets `
  --split-seed 23 --stage test --batch-sets 3
```

The [scanner](derangement_batch_feasibility.py) emits the input manifest digest, every rolling-window count, and the canonical non-overlapping schedule. It uses no model endpoint and writes no task artifacts.

## Frozen bundle and result

- Split seed: `23`; task seed: `0`; `k=4`; test support: 48 meanings / 12 candidate sets / 48 episodes.
- Split SHA-256: `4a1211075ce60664f7e83df4a6a1cfcf2ce82f3f4663518dd89f0aa0329b83a1`.
- Local episode-manifest SHA-256: `AE4451ADC982905C8FBF458FC583BA23D29811318FABA167C89F24229047603C`.
- All 10 rolling windows of three consecutive candidate sets (offsets 0 through 9) admit a perfect matching. Minimum receiver degree is 7 or 8.

| Candidate-set offset | Episodes | Compatible perfect matchings | Minimum degree |
|---:|---:|---:|---:|
| 0 | 12 | 4,783,104 | 8 |
| 1 | 12 | 1,078,272 | 7 |
| 2 | 12 | 1,078,272 | 7 |
| 3 | 12 | 4,783,104 | 8 |
| 4 | 12 | 1,078,272 | 7 |
| 5 | 12 | 1,078,272 | 7 |
| 6 | 12 | 4,783,104 | 8 |
| 7 | 12 | 4,783,104 | 8 |
| 8 | 12 | 4,783,104 | 8 |
| 9 | 12 | 4,783,104 | 8 |

For complete, non-overlapping coverage, freeze replay blocks at offsets **0, 3, 6, and 9**. Each block contains three sets and 12 receiver calls; together they cover the 48-episode test stage exactly once. Each block has 4,783,104 compatible perfect matchings, from which the runtime samples uniformly using its hash-bound seed.

The fixed block rule is based on candidate-set order, not model outcomes. This feasibility screen was performed after the bundle's task layout was generated, so disclose it as an exploratory structural audit rather than claiming a prospectively registered sample design. Do not alter these offsets after seeing source or replay scores. A new seed/task-key bundle needs its own pre-outcome feasibility report; a failed block invalidates this four-block schedule rather than authorizing a score-informed replacement.

## Interpretation and next gate

This establishes that exact message-multiset-preserving, candidate-disjoint replay is combinatorially possible for this one 12-set fixture. It does not show that a receiver understands any protocol, that messages convey useful information, or that four blocks are independent statistical replicates. They share one task seed, one split, and potentially one learned protocol/setup artifact. Confirmatory inference still needs multiple independently generated split/task clusters, preregistered receiver conditions, an independently passed capability gate, and the full source-plus-replay cost ledger.

The source messages have not been generated, and the latest checked local resource preflight was rejected. No LLM call, local model load, service startup, or inference was made for this screen.
