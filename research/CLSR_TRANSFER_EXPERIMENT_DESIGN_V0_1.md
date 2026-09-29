# CLSR-style protocol transfer experiment design v0.1

**Status:** design proposal; not preregistered and not executed. The most recent local resource gate rejected inference. This design does not change the frozen Emergent OOD v0.3 feasibility runner or authorize model calls.

## Question and choice of task

Can a reusable, LLM-generated compositional dialect let a sender communicate a held-out meaning to an independently prompted receiver more efficiently than strong text, schema, and code baselines?

Use the **Emergent OOD receiver-utility family** for this question. Its sender sees a private meaning, its receiver sees a target-balanced candidate set, and its train split exposes all lower-order attribute combinations while holding out higher-order compositions. This makes receiver-side compositional transfer an explicit endpoint. Keep **Private Match** for exact coding and payload-frontier experiments: its messages are atomic coordinates, so it is a better bit/rate control than a test of invented compositional syntax.

CLSR is the source of the reusable-dialect lifecycle (induce → evolve/profile → freeze → route), not a drop-in benchmark result. The planned arm must be labeled **CLSR-inspired transfer adaptation**, not a CLSR reproduction, unless the paper's exact model, data, prompts, manifests, and evaluation procedure are reproduced. The paper itself identifies future transfer to multi-agent environments as open; see the [source audit](CLSR_AUDIT_V0_1.md).

## Why the current v0.2 task is a feasibility fixture only

The current meaning space is `3^3 = 27`: 18 meanings for training and nine held-out triples. It is suitable for checking the task/scorer and testing a first signal, but nine held-out meanings do not leave enough independent material to separate dialect evolution, route/profile selection, and final evaluation. The v0.3 runner has only full-information, no-message, and plain-English stages; it is not an LSF runner. Do not append an improvised CLSR arm to its frozen five-episode capability screen.

## Candidate scaling extension

For a confirmatory compositional task, use four attributes with four values each, universe `[4]^4`, giving 256 meanings. Independently permute the four values on each axis for split seed `s`, then define:

\[
H_s=\{x\in[4]^4:\sum_{i=1}^4\pi_{s,i}(x_i)\equiv0\pmod 4\},\qquad
T_s=[4]^4\setminus H_s.
\]

There are 64 held-out four-way meanings and 192 training meanings. For any fixed assignment to any three attributes, exactly one of the four completions lies in `H_s`, so the other three are in `T_s`. Thus every value, pair, and triple of attribute values appears in training, while the complete four-way combination is novel. Validation and evaluation meanings should be partitioned deterministically from `H_s` before dialect induction; neither validation nor evaluation tuples can appear in the induction examples. The model-free split implementation now lives in [`experiments/emergent_ood_v0_4/`](../experiments/emergent_ood_v0_4/README.md); focused checks cover the default counts, complete lower-order coverage, deterministic seeds, and tamper rejection. This validates the partition construction only, not episode sampling or communication performance.

Candidate sets are sampled from the evaluation meanings, not enumerated over all subsets. Each sampled set is target-balanced across its `k` members, preserving exact no-message Bayes accuracy `1/k`. Candidate-set episodes are nested observations: retain split-seed and candidate-set cluster IDs. The randomized split seed is a replicate of a partition over a fixed finite meaning universe, not automatically a new sample from a broad semantic population; meanings can recur across seeds. For claims beyond this finite ontology, replicate across independently generated attribute lexicons/schemas or explicitly limit the estimand to the declared ontology. Do not pool examples across seeds into one dialect if that would expose an evaluation tuple from one seed during training on another.

The coding references expose a real tension rather than assuming the compositional language wins. Across all 64 held-out meanings, a pre-shared holistic ID code has a zero-error floor of 6 fixed-width bits. A compositional four-attribute binary code uses 8 bits (2 bits per attribute) but can express unseen combinations from reusable primitives. The holistic code can be shorter on this finite target set; its shared dictionary/setup and transfer failure outside that set must be measured. These are payload bounds, not predictions about LLM tokens or interpretation accuracy.

## Frozen causal comparison

Hold to one sender, one receiver, one unicast message, one final receiver answer, identical private information, candidate ordering, prompts outside the protocol card, decoding settings, and fixed call schedule. No router may choose more agents, rounds, or extra verification only for one arm. Use separate ledgers and freeze each artifact before evaluation.

Candidate arms:

1. No-message and full-information controls.
2. Development-tuned concise natural language and the AutoForm-style prompted-format baseline.
3. Strict valid JSON or compact structured text, with the exact schema and parser frozen.
4. A task-aware compositional symbolic reference with explicit per-attribute primitives.
5. A holistic training-only lookup control that has no entry for a held-out full combination and uses a predeclared abstain/failure behavior.
6. A CLSR-inspired LSF card induced from training-only communication episodes, evolved/profiled only on training and validation data, then frozen before test. The same frozen card is given to sender and receiver; card contents must not include validation/evaluation targets or message-answer pairs.
7. A 6-bit holistic held-out-class index as a model-free payload oracle, clearly marked as a finite-set code requiring a shared task-specific map. It establishes a lower-cost non-compositional reference, not a practical LLM language.

If the CLSR route planner is also evaluated, report it as a **whole-system arm** with all router/profile input tokens, outputs, model calls, and schedule changes included. It cannot be used to claim a message-representation-only improvement. For the representation-only arm, freeze one route/card and one message round.

## Leakage, fidelity, and transfer gates

- Induction gets only training meanings and training messages; profile/routing selection uses only the validation partition. Evaluation messages and candidate sets stay sealed until all prompts, cards, parsers, stopping rules, and budgets are frozen.
- No exact full meaning from an evaluation episode may occur in the LSF exemplars, model-generated candidate cards, route summaries, or retry traces. Publish content hashes of split manifests and frozen protocol artifacts.
- Measure sender fidelity to the private meaning, receiver reconstruction/selection fidelity, terminal exact accuracy, syntax validity, ambiguity/collision rate, and abstention separately. A short output or high TopSim is not success.
- First evaluate same-family sender/receiver as a feasibility stratum. Then transfer the identical card, without receiver-specific edits, to a held-out receiver model family. Do not conflate a card re-prompted for the new model with zero-shot transfer.
- The full-information gate must use disjoint calibration data, be passed by each receiver stratum, and never filter the held-out evaluation set. The current frozen local resource gate remains mandatory before every model batch.

## Cost and budget frontiers

Publish two frontiers:

- **Online inference frontier:** per-recipient tokenizer input/output tokens for every call; serialized payload, framing, and envelope bytes; call count; service/wall latency; model/runtime stratum; exact task success and semantic fidelity.
- **Amortized system frontier:** the same episode costs plus all LSF generation/evolution/profile/selection calls and tokens, codebook/card size and distribution bytes, route setup, and the declared reuse horizon. Do not convert unlike tokenizer counts, bytes, calls, and latency into one scalar unless a deployment price function is predeclared.

Within one fixed tokenizer population, impose a per-message generation-token cap sweep and report task success versus complete cost. Also report a byte-cap sweep, because tokenizer-token parity does not imply equal serialized bandwidth. Count truncated/invalid messages as failures; no post hoc repair outside the fixed one-message schedule. For heterogeneous models, report each tokenizer's costs separately and use serialized bytes for the shared transport axis.

## Falsifiable prediction

If the LSF conveys reusable compositional structure rather than memorized whole meanings, then after training has exposed every unary, pair, and triple, the frozen LSF should retain exact receiver utility on held-out four-way combinations above the training-only holistic lookup control. Its advantage should be largest at tight per-message budgets and persist with a held-out receiver when the same card is supplied verbatim. If it only shortens output while exact held-out selection remains near (1/k), it is not a useful communication language. If a 6-bit holistic code or optimized structured text dominates after setup is amortized, the LSF has no demonstrated role for this task family.

## Decision gate before implementation

1. Validate the proposed (4^4) split, lower-order coverage, candidate-set balance, no-message prior, and the exact 6-/8-bit references model-free.
2. Freeze the dialect induction/evolution protocol and protocol artifact schema; explicitly label the method as CLSR-inspired and publish its deviation from the paper.
3. Estimate request counts, prompt lengths, sample-size needs, and local resource limits. The current five-episode/12-call feasibility runner is not an inferential study.
4. Only after the frozen hardware gate passes, run disjoint full-information receiver calibration. Stop if the receiver gate fails.
5. Execute a descriptive, paired feasibility pilot with all controls before designing a confirmatory sample. No pilot may be called a superiority result.

The v0.4 split generator has been implemented and its focused unit checks pass. No role-separated episode runner, dialect, or model experiment has been implemented by this design note. It remains a falsifiable plan, not experimental evidence.
