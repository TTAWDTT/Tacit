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

## Joint utility-cost frontier

Let `C_ch` be the complete serialized channel cost for one episode (report bytes and each participating model's tokenizer units), and let `C_inf` be inference cost (at least total prompt+completion tokens and model calls; add wall/service time and hardware use when available). These are separate resource coordinates, not interchangeable units. For a fixed task distribution and model population, define the achievable success frontier

`F(B_ch, B_inf) = sup_{π,c,d} P(Ŷ = Y)`

subject to `E[C_ch] ≤ B_ch` and `E[C_inf] ≤ B_inf`.

This makes “more efficient” a measurable claim: a system improves the frontier if it raises success at the same pair of budgets, or reaches the same success with no more of either resource. A single weighted sum `C_ch + λ C_inf` is valid only after reporting and justifying the exchange rate `λ`; otherwise it can conceal a transfer of cost from the channel into longer prompts, decoder work, or extra calls. Shared codebook/grammar acquisition and distribution must be amortized or reported separately, and robustness should be shown as a separate constrained frontier when corruption is introduced.

**Falsifiable prediction P1.** If a representation's main advantage is payload compression, its success advantage over an equally optimized natural-language/structured baseline should be largest at tight `B_ch`, and should shrink as the channel budget becomes slack, provided `B_inf` and all policy decisions are matched. If the gap instead tracks a changed schedule, model-token count, or receiver prompting budget, the gain is not evidence of a better representation. Evaluate P1 with a small preregistered sweep of channel budgets and report the full `(success, C_ch, C_inf)` points; do not fit a scaling law unless the sweep covers enough levels and held-out task instances.

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

## A communication-complexity control task

The standard `INDEX_m` task gives the sender a private bit vector `x ∈ {0,1}^m` and the receiver a private index `i ∈ {1,…,m}`; the receiver must output `x_i`. Draw `x` uniformly and independently of `i`.

**Proposition (deterministic one-way lower bound).** Any zero-error one-way protocol from sender to receiver requires at least `m` bits in the worst case.

**Proof.** If two different vectors `x` and `x′` produced the same sender message, choose a coordinate `i` where they differ. A receiver holding that `i` would see the same message in both cases and therefore produce the same output, which must be wrong for one vector. Thus all `2^m` vectors need distinct messages, requiring at least `log₂(2^m) = m` bits. Sending `x` verbatim meets the bound.

With interaction, the receiver can send `i` in `⌈log₂ m⌉` bits and the sender can reply with `x_i` in one bit, for a total of at most `⌈log₂ m⌉ + 1` bits, before framing or protocol-negotiation overhead. This is an upper bound for one simple interactive protocol, not a claim that interaction always wins under LLM token, compute, or latency costs. With no communication and independent uniform `x`, any receiver strategy has expected accuracy `1/2`; a single-agent full-information control can read both `x` and `i` and should return the exact bit.

This task family supplies known communication requirements and a scaling axis without arithmetic. However, it initially tests a narrow information-transfer primitive, not broad reasoning or a natural-language semantic task. LLM ability to reliably read long random vectors is a separate capability bottleneck and must be gated before comparing representations.

The experimental separation is:

1. **Capability and necessity controls:** single-agent full information, no-message receiver, and a deterministic oracle channel.
2. **Policy comparison:** one-way full-vector transfer versus receiver-query / sender-answer interaction. This measures message selection and round cost, not language quality.
3. **Representation comparison:** within each fixed policy, relay identical semantic payloads as optimized natural language, structured text, and a compositional typed code. Count messages in both directions, common setup/decoder instructions, model-tokenizer tokens, bytes, generated tokens, service time, retries, and accuracy.
4. **Scaling:** vary `m` only after the smaller-length capability gate passes; compare observed cost and error against the `m`-bit one-way lower bound and the interactive `⌈log₂m⌉+1`-bit construction. A token is not a bit, so these are information bounds, not direct token predictions.

Keep the one-way and interactive results separate. If a compact encoding wins only when it changes which facts are sent, the result is a policy effect. If it wins under identical semantic payloads and decoder access, it is evidence about representation. This design is further specified in [`INDEX_PROTOCOL_DESIGN.md`](INDEX_PROTOCOL_DESIGN.md).

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
