# Emergent communication: held-out structure versus receiver utility v0.1

## Primary source

Tom Kouwenhoven, Max Peeperkorn, and Tessa Verhoef, “Searching for Structure: Investigating Emergent Communication with Large Language Models,” COLING 2025, [ACL Anthology paper and PDF](https://aclanthology.org/2025.coling-main.667/).

This is a direct prior for any proposal to let LLMs discover a machine-facing code. It must not be framed as an unexplored idea.

## What the study establishes

The authors use a text-only Lewis-style referential game over 27 meanings formed from 3 shapes, 3 colours, and 3 quantities. For each of 15 simulations, they start from random holistic signals, expose models to a 15-meaning training set with every attribute value represented equally, and run four communication rounds of 30 interactions each. The agents use Llama 3 70B with greedy generation and in-context vocabulary examples. Communication success is about 70% in round one and about 75% later, with fluctuations across simulations.

The trained vocabularies become more structurally similar by TopSim, and their N-gram diversity falls. The paper reports that message strings become longer across rounds, despite increasing structure. In the final test block, agents generate signals for all 27 meanings; the paper reports a correlation between TopSim and its GenScore diagnostic of \(r=0.735\). Mean unique-signal ratio is 62.1%, so structural reuse can coexist with many-to-one or underspecified labels. Six transmission chains of eight generations show improved learnability, but TopSim itself does not significantly change across generations.

These findings are useful positive and negative priors: LLMs can adapt artificial signal systems in context, some structure can emerge without an explicit compositionality objective, generation can lengthen, and apparent structure can coexist with semantic collisions.

## What the results do not establish

The held-out test block asks agents to generate signals for unseen combinations. It does not run an independent receiver through target selection on those novel combinations. The 70–75% interactive success therefore measures communication on the 15-item training meaning set; the reported held-out GenScore measures how generated-signal similarities covary with meaning similarities. Neither alone establishes exact end-to-end decoding or task utility on unseen combinations.

The paper does not report an equal-token or equal-byte frontier against optimized natural language, valid JSON, fixed symbolic composition, or a holistic lookup code. Signals lengthen during communication, and the paper reports no complete prompt/completion, inference-compute, latency, or serialized-channel accounting. Its results establish emergence under a particular scaffold and model, not a communication-cost advantage or reusable protocol across receiver models.

The paper is internally inconsistent about the candidate count. Section 3 explicitly says one target plus four distractor stimuli, which is five choices and implies 20% uniform-guess chance. Section 5.2 labels chance performance as 25%, which corresponds to four choices. No official code link is provided in the paper/ACL record, and the text alone cannot establish which candidate generator was used in the executed runs. Preserve both statements as a source discrepancy; do not silently treat 25% as verified. A replication must report the candidate-set cardinality from its actual episode generator and derive chance from that set and its target-sampling distribution.

## Updated falsifiable test

The missing test is receiver-side utility on meanings that were not present during protocol learning. On the same novel target/candidate episodes, compare:

1. an emergent signal learned from training interactions;
2. natural language and valid structured-text messages;
3. a deterministic compositional symbol code with shared per-attribute primitives;
4. a holistic lookup code restricted to meanings seen during training;
5. no-message and full-information controls.

The primary endpoint is exact receiver selection on held-out combinations. Also report sender meaning fidelity, receiver decode fidelity, ambiguity/collision rate, message bytes and receiver-native tokens, complete inference cost, latency, recovery, and transfer to an independent receiver. Hold candidate count and interaction policy fixed, use multiple split seeds, and include equal-budget and equal-quality comparisons. TopSim and N-gram statistics remain explanatory diagnostics.

The falsifiable prediction is conditional: if structural reuse represents genuinely compositional semantics, then it should preserve held-out receiver success better than a training-set holistic lookup when each attribute primitive was observed during training. If it only raises similarity metrics or generation consistency while novel receiver selection stays near chance, the structure is not yet a useful communication language.

## Project consequence

Keep emergent protocol learning as a serious route, but do not repeat its novelty claim. A future experiment should target the gap between held-out signal generation and held-out receiver utility, then challenge any positive result with optimized text, symbolic, holistic, and complete-cost baselines. Local inference remains gated by the preregistered resource thresholds; no new model run is authorized by this audit.
