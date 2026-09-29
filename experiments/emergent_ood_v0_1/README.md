# Emergent OOD split v0.1

This model-free fixture defines a controlled held-out-composition split for a
three-attribute meaning space. Held-out receiver generalization is an
established evaluation idea; this small split exists only to validate a
reproducible third-order partition. It is not a novel benchmark contribution.
In particular, generating a signal for an unseen meaning does not show that
another agent can decode it correctly.

## Split contract

The universe is the Cartesian product of three axes (`shape`, `color`, and
`quantity`), each with values `{0, 1, 2}`. A seed independently permutes the
labels on each axis. A meaning is held out when the sum of its permuted labels
is divisible by three; the remaining meanings train protocol learners.

This creates 18 training and 9 held-out meanings. Every individual value and
every pairwise combination occurs in training. Therefore evaluation requires
composing three already-seen values into a previously unseen triple. Different
seeds rotate which triples are held out. The literal numeric labels are
benchmark identifiers, not a proposed communication syntax.

Generate a fixture with:

```powershell
python experiments/emergent_ood_v0_1/split.py --seed 17 --output experiments/emergent_ood_v0_1/splits/seed_17.json
```

The generator validates disjointness, full coverage, split size, unary coverage,
and pairwise coverage before writing. Its unit tests run with:

```powershell
python -m unittest discover -s tests -p test_emergent_ood_split.py -v
```

## What this fixture does not establish

It does not train or evaluate a sender/receiver, measure task utility, ensure
that a language has stable semantics, or establish that communication is
necessary. A future task runner must give a sender private access to the target
and an independently initialized receiver a candidate set, then score exact
receiver selection on held-out targets. It should use multiple split seeds and
compare learned signals with optimized text, structured text, compositional
symbolic, holistic training-set lookup, no-message, and full-information
controls. The held-out receiver endpoint and complete channel/inference costs
remain prerequisites for any protocol claim.

See the [held-out utility audit](../../research/EMERGENT_OOD_UTILITY_AUDIT_V0_1.md)
and [generalization prior audit](../../research/EMERGENT_GENERALIZATION_PRIORS_AUDIT_V0_1.md).
