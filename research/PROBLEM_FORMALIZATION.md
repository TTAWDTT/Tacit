# Separating Communication Policy from Message Representation

**Status:** Working formalization; definitions and predictions for experiments, not a result claim.

## Motivation

The current evidence points to two different problems that are easy to confound:

1. **Information selection:** which private facts an agent chooses to disclose, which questions it asks, and when it stops exchanging information.
2. **Representation:** how selected content is encoded on the channel and reconstructed by receivers.

The HiddenBench paper's Reveal-All intervention substantially raises accuracy, while its Exchange/Decide procedure changes who surfaces information and when agents commit. Those results indicate that protocol policy must be held fixed when testing an encoding. A shorter string cannot be credited for a task improvement that came from a better disclosure policy.

## Task and protocol model

Let a task be a random variable over instances. Agent `i` observes `X_i`; agents may also observe common context `S`. The target answer is `Y = f(S, X_1, ..., X_n)`. A communication episode produces a transcript `T` and a final answer `Ŷ`.

Represent an agent communication system as three components:

- **Policy** `π`: chooses the semantic content of each message (including questions, disclosures, challenges, and silence) from the agent's observation and received history.
- **Code** `c`: maps that semantic content to a serialized representation.
- **Decoder** `d`: reconstructs an interpretation from the received representation and uses it to update the agent's state or answer.

The channel budget `B` must be stated in a concrete unit. At minimum report model-tokenizer tokens and serialized UTF-8 bytes separately. A tokenizer count is model-specific; byte count is portable but does not equal inference work. Also report generated tokens and service time because message input cost alone misses the compute used to produce and consume messages.

For fixed policy and decoder, a representation comparison estimates the effect of `c`. For fixed representation, a policy comparison estimates the effect of `π`. A full factorial design can test whether their effects interact. Changing policy, code, model prompt, and round count together does not identify a language effect.

## Diagnostic events

Where a task has an independently audited set `K` of facts sufficient to determine its answer, define:

- `Q`: the transcript semantically conveys all facts in `K` to at least one final decision maker.
- `C`: the final answer is correct.

If the task construction guarantees that `Q` is necessary for correctness (`C ⇒ Q`), then:

`P(C) = P(Q) × P(C | Q)`.

This separates a **surfacing failure** (`Q` does not occur) from an **integration failure** (the facts are present but `C` does not occur). The implication `C ⇒ Q` must be validated per task; if partial evidence or chance can produce the answer, report the full joint table `P(Q,C)` instead of using the factorization. Fact coverage needs semantic annotation with adjudication; exact string matching is insufficient.

## Information bound

For a fixed-length transcript represented by at most `B` bits, `I(Y;T) ≤ H(T) ≤ B`. Any decoder with error probability `p_e` over `|Y|` answers also satisfies Fano's inequality:

`H(Y | T) ≤ h₂(p_e) + p_e log₂(|Y| − 1)`.

Together these inequalities give a task- and distribution-dependent constraint: no representation can communicate more than its channel budget, and a compressed message only helps if it preserves information relevant to `Y`. They do **not** establish that shorter codes improve LLM task success: decoding errors, additional inference, prompt sensitivity, and compute latency can reverse a token-count gain. This is a benchmark-level bound, not a universal optimality claim about a language.

## Falsifiable predictions

1. **Policy diagnosis:** on Hidden Profile tasks with a full-information capability gate, forcing explicit private-fact disclosure should raise `Q` more than it raises `P(C | Q)`. If not, the interpretation that information surfacing is the main bottleneck is weakened.
2. **Encoding isolation:** with a frozen policy, task seeds, model, and decoder training/instructions, a shorter encoding should preserve fact coverage and conditional answer accuracy. If it reduces either, it lies on a worse fidelity-cost frontier even when its messages use fewer tokens.
3. **Budget dependence:** if an encoding advantage comes from removing linguistic redundancy rather than changing which facts agents disclose, the advantage in task success at matched cost should grow as the message budget tightens; with loose budgets, the success gap should shrink. Failure of this interaction would reject that explanation.
4. **Cross-model transfer:** a learned or compact code that depends on shared model-specific token associations should lose fidelity across model families unless a short calibration or explicit decoder is provided. Evaluate both same-model and cross-model receivers.

## Experimental consequences

The next HiddenBench study should first reproduce natural discussion, the paper's two-round Exchange / one-pass Decide condition, and Reveal-All as separate policy conditions. Reveal-All is a diagnostic upper bound, not a fair bandwidth-matched competitor. Only then should code variants be compared inside a fixed policy. Record correctness, audited fact coverage, token and byte budgets, latency, generated tokens, retries, and invalid/truncated outputs. Use multiple task instances and seeds; the current three verification tasks are too few for an efficiency frontier or scaling law.

The pinned upstream simulator accepts one static `extra_prompt` for every discussion turn and does not expose a phase-specific round index in the prompt. Therefore its stock CLI cannot faithfully implement a two-stage Exchange/Decide schedule using a single extra prompt. A controlled reproduction needs a small, versioned local adapter or an upstream-supported phase-specific interface; merely asking the model to “exchange, then decide” in the static prompt is a weaker, non-equivalent intervention.

## Sources

- Li, Naito, and Shirado. [Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](https://arxiv.org/abs/2505.11556), arXiv v4, especially §§6.3–6.4.
- Zhang et al. [SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](https://aclanthology.org/2026.acl-long.1354/), ACL 2026. Its aggregation, mesh, and global-shuffle task levels motivate measuring communication complexity beyond one benchmark difficulty point.
