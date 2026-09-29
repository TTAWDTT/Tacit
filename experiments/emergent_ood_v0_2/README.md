# Emergent OOD receiver-utility episode generator v0.2

This extends the v0.1 deterministic 18/9 composition split with a task interface in which a sender alone receives the target meaning and a separate receiver sees only a candidate set. It is a model-free task/scorer fixture, not a new benchmark claim and not evidence that a learned language is useful.

The default episode has five candidates. For every 5-element subset of the 9 held-out meanings, the generator creates one episode for each candidate as the private target: `5 × C(9,5) = 630` episodes. Therefore, conditioned on any receiver candidate set, each candidate is exactly equally likely to be the target. The exact no-message Bayes accuracy is `1/5 = 20%`. Other candidate counts from 2 to 9 are available for difficulty sweeps; each has the same target-balanced construction and chance `1/k`.

Generate role-separated files with:

```powershell
python experiments/emergent_ood_v0_2/episodes.py --seed 17 --candidate-count 5 --output-dir .cache/emergent_ood_v0_2/seed_17_k5
```

The output includes `sender.jsonl` (private target), `receiver.jsonl` (candidate meanings without a target marker), `gold.jsonl` (answer key), and a manifest with the split hash and exact no-message reference. Keep the gold ledger inaccessible to the receiver/model. Each receiver must submit a `candidate_id`; any parser and invalid-output policy must be fixed before a model call.

The task uses the v0.1 meaning split: every unary value and pairwise combination appears in the 18 training meanings, while all 9 held-out triples are novel compositions. Candidate sets are drawn only from those 9 held-out meanings, so receiver success directly tests task utility on unseen compositions. The train meanings are included in the manifest for provenance; they are not injected into role prompts by this generator.

Before any model-based comparison, separately freeze each sender/receiver prompt and representation protocol. At minimum, compare no message, full-information capability, optimized natural-language description, valid structured text, a compositional-symbol control, and a train-only holistic lookup control. Charge prompt/codebook setup and full model inference, measure exact payload bytes and receiver-native message tokens, and use a passing preregistered host-resource gate. This generator does not prescribe a winning representation, implement those protocols, or authorize model inference.

Validate the generator with:

```powershell
python -m unittest discover -s tests -p test_emergent_ood_episodes.py -v
```

The pre-inference task contract is frozen in [`preregistration.json`](preregistration.json). Held-out receiver utility and concept games have substantial prior work; this small fixture is a controlled experimental task, not a novelty claim. See the [prior audit](../../research/EMERGENT_GENERALIZATION_PRIORS_AUDIT_V0_1.md) and [LLM emergent-language audit](../../research/EMERGENT_OOD_UTILITY_AUDIT_V0_1.md).
