# Working theory: task-conditioned communication rate

**Status:** v0.1, falsifiable framing. These definitions organize experiments; they do not yet prove that a particular representation is better.

## 1. Episode and protocol

An episode is a cooperative task (T) with target (Y^*), private observation (X_i) at agent (i), public/shared context (C), and receiver implementation (r) (model weights, tokenizer, prompt/runtime, and available tools). A protocol π specifies the encoder, message syntax, any shared dictionary or setup exchange, decoder/receiver policy, and interaction schedule; Section 1.1 factors this shorthand into components. The received messages (M_{1:k}) induce a final action Ŷ and task utility (U_T(Ŷ,Y^*)in[0,1]).

Define task distortion as

\[
d_T = 1 - U_T(\hat Y,Y^*).
\]

Exact-answer tasks may use 0/1 utility. Tasks with graded outputs need a task-specific proper score or rubric fixed before the run. ROUGE is a textual similarity measure and is not a substitute for task utility when exact outcomes are available.

### 1.1 Factor the communication system before attributing gains

Represent an implemented communication system as

`P = (σ, φ, ρ, δ)`

where:

- `σ` is the interaction policy: who can speak, when, to whom, and which private content is surfaced;
- `φ` is the format-selection policy: how a task/receiver/history condition is mapped to a format choice or candidate-format set;
- `ρ` is the representation itself: the grammar/encoder that maps selected content into a payload;
- `δ` is the receiver-side decoder/parser, including any explicit repair step.

The receiver model/runtime remains part of condition `r`, and its complete prompt and completion costs are measured. A format selector is not itself a language: AutoForm is prompted selection among formats, and OPTiMACS learns a task-conditioned format policy while also growing its format inventory. If either selector changes *which* information gets disclosed or *whether* a turn occurs, its effect is a policy effect unless the schedule and semantic content are held fixed.

For a preregistered operational budget vector `b`, let `J_b(σ,φ,ρ,δ)` be expected task utility when the same channel and inference caps are enforced and all failures, truncations, retries, and over-budget behavior are scored by the frozen evaluator. A codec-package contrast is

`Δ_codec(b) = J_b(σ,φ,ρ₁,δ₁) − J_b(σ,φ,ρ₀,δ₀)`

with the schedule `σ`, selector `φ` (including its selected format slot on each episode), task episodes, and semantic content to be conveyed matched. This estimates the representation-plus-decoder package; an encoder-only claim requires a shared decoder. A selector contrast instead varies `φ` over a common format inventory while freezing `σ` and the receiver. When a new representation requires its own learned selector, report both the combined system and a matched-selector codec ablation; do not label the combined delta as a language-only gain. If resources permit, use a `φ × (ρ,δ)` factorial and report their interaction rather than silently optimizing one arm more than another.

This factorization is necessary because a method can improve success by selecting different messages or turns without compressing a fixed semantic payload. Log chosen content, format choices, turns, and decoded content so each mechanism is auditable. Charge selector training/search and per-message selection calls to setup or inference at their actual boundary.

## 2. Task distortion with receiver side information

For a two-agent episode, let sender A observe private state \(X\), receiver B observe private state \(Z\), and both observe public context \(C\). The target \(W\) and task loss \(\ell(W,a)\) are fixed by the task. Without communication, the receiver chooses an action from \((Z,C)\); with communication, A sends \(M\) using only \((X,C)\), and B chooses from \((M,Z,C)\). The encoder must not silently condition on \(Z\) unless that information is explicitly part of the sender's view.

For fixed model/runtime condition \(r\), protocol \(\pi\), and operational budget \(b\), define

\[
D_{T,r,\pi}(b)=\mathbb{E}[\ell(W,A_\pi)]
\quad\text{subject to}\quad
\mathbf c_\pi\preceq b.
\]

This is the quantity experiments observe. It uses a declared task distribution, exact or preregistered task score, actual serialized channel payload, and separately recorded inference/setup/runtime budgets.

### Ideal information-theoretic reference

For an i.i.d. source and a single message, the **Wyner–Ziv task rate-distortion benchmark** is

\[
R^{\mathrm{WZ}}_{T}(D\mid C)=\inf_{p(u\mid x,c),\,g}
I(X;U\mid Z,C),
\]

where \(U-X-(Z,C)\) conditional on \(C\), the receiver outputs \(\hat W=g(U,Z,C)\), and \(\mathbb E[\ell(W,\hat W)]\le D\). Equivalently, with decoder-only side information this classical operational limit has the auxiliary-variable form \(\inf[I(X;U\mid C)-I(Z;U\mid C)]\) over the same feasible test channels and decoders. If the sender also knows \(Z\), the problem changes to conditional rate-distortion; that is a different information boundary. This adapts Shannon's distortion objective and Wyner–Ziv's decoder-side-information setting to task loss rather than literal reconstruction ([Shannon 1948](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x), [Wyner & Ziv 1976](https://doi.org/10.1109/TIT.1976.1055508)). The information-bottleneck view similarly asks a representation to discard source information irrelevant to a specified target, rather than preserve all details ([Tishby, Pereira & Bialek](https://arxiv.org/abs/physics/0004057)).

This is a **reference bound, not the token cost of an LLM**. It assumes a known source distribution, block coding over sufficiently long i.i.d. sequences, a chosen task-loss function, and an ideal encoder/decoder. Finite one-shot episodes, restricted language models, prompt overhead, compute, latency, learned codebooks, and interactive rounds need operational measurement. Information complexity provides a separate framework for interactive protocols; distributed function-computation coding is a closer analogy when the receiver needs a function of both private inputs ([Braverman, *Interactive Information Complexity*](https://epubs.siam.org/doi/10.1137/17M1139254), [Orlitsky & Roche, *Coding for Computing*](https://doi.org/10.1109/18.915643)). Neither theorem directly converts LLM tokens into an optimal semantic rate.

### Proposition: when communication has no task value

Let \(R_0=\inf_{\delta}\mathbb E[\ell(W,\delta(Z,C))]\) be the receiver's Bayes risk with no message, and \(R_{\mathrm{full}}=\inf_{\gamma}\mathbb E[\ell(W,\gamma(X,Z,C))]\) its Bayes risk if both private views were centrally available. Then

\[
R_{\mathrm{full}}\le R_0.
\]

**Proof.** Any no-message decision rule \(\delta(Z,C)\) is a feasible centralized rule \(\gamma(X,Z,C)=\delta(Z,C)\) that ignores \(X\). The centralized infimum is over a superset of rules, so it cannot have greater risk. \(\square\)

If \(R_0=R_{\mathrm{full}}\), communication cannot improve optimal task risk for that task distribution and loss, regardless of the language. If \(R_0>R_{\mathrm{full}}\), private sender information has positive decision value in aggregate, but this is only a necessary opportunity check: it does not prove any particular finite-budget message or LLM will realize the gain. In experiments, use an exact no-message/oracle comparison where possible; model no-message failures alone may reflect weak receiver capability rather than communication need.

Three distinctions matter:

1. Information rate is not UTF-8 bytes, model tokens, FLOPs, latency, or energy. Record those operational measures separately.
2. Task distortion depends on receiver state, task, and loss. Omitting irrelevant prose can have zero task distortion; omitting one rare constraint can cause a large loss.
3. With multiple rounds, later messages depend on previous messages and actions. The relevant object is an interactive protocol; independently compressing each turn can raise round count, repeated prompt cost, or repair traffic.

## 3. Operational efficiency frontier

For protocol π and task/model condition (z=(T,r,\text{channel},\text{horizon})), record a vector rather than choosing arbitrary prices in advance:

\[
\mathbf c_\pi(z) = (B_{wire},\; T_{in},\; T_{out},\; N_{calls},\; N_{turns},\; L_{wall},\; K_{setup},\; K_{decode},\; E_{decode},\; P_{failure}).
\]

The quality-cost frontier at a fixed condition is the set of protocols not dominated in expected task utility and the declared cost dimensions. Report equal-token, equal-byte, and equal-quality comparisons where meaningful. A method that reduces generated tokens while inflating repeated input context, setup, or repair remains visible as a trade-off instead of disappearing in one scalar.

## 4. A checkable setup-cost crossover

Suppose a reusable protocol requires one-time setup cost (A\ge0), then costs (c_L) per episode; a baseline costs (c_B) per episode. For (H) episodes, the reusable protocol is cheaper exactly when

\[
A + Hc_L \le Hc_B.
\]

If (c_B>c_L), the break-even horizon is

\[
H^* = \left\lceil\frac{A}{c_B-c_L}\right\rceil.
\]

If (c_B\le c_L), no finite reuse horizon repays setup under this cost measure. This elementary result predicts that task-specific codebooks or learned decoders can be worse for one-off tasks even when their steady-state messages are shorter. Experiments must charge setup and specify the reuse horizon.

## 5. Predictions to test

- **P1, task distortion:** at equal message budget, compact formats help only when omitted information is conditionally irrelevant to the receiver's optimal task action; errors should cluster around task-critical omissions, not message length itself. Where the centralized/no-message Bayes-risk gap is zero, no protocol should improve optimal task utility; a measured gain then indicates model/control mismatch, scorer noise, or a changed effective task/budget.
- **P2, receiver alignment:** for a fixed sender representation, downstream distortion depends on the receiver (r). A code's token/byte advantage should be separated from its decoder mismatch cost.
- **P3, interactive compression:** under a strict total budget, a second turn helps only if its expected value of information exceeds repeated context, prompt, and control overhead. Compare several turn allocations at a fixed total budget.
- **P4, setup horizon:** observed codebook savings should cross the baseline at the measured (H^*) within uncertainty if per-episode costs are stable. Failure to do so suggests non-stationary quality, hidden cost, or an incomplete accounting boundary.
- **P5, error recovery:** adding redundancy or clarification can lower first-pass message efficiency but improve expected task utility under corruption/receiver mismatch. Measure bytes/tokens including retries and repair turns.

## 6. Conditions required for a stronger theorem

A useful optimality or lower-bound claim needs an explicit task distribution, side-information model, receiver class, distortion function, channel (including who pays for the decoder), and interaction limit. It must also state whether encoder/decoder knowledge is shared for free or has setup cost. Without these assumptions, "most efficient language" is underspecified. The project will not claim a universal optimum from finite benchmark evidence.

## 7. A tight lower bound for a two-agent hidden-sum task

This gives the next benchmark family an exact communication reference point.

**Model.** Agent A privately holds \(x\in\{0,\ldots,M-1\}\); agent B privately holds \(y\in\{0,\ldots,M-1\}\). Both must output the exact integer \(x+y\). They have no shared input-dependent advice, no free preprocessing, and communicate over a noiseless binary point-to-point channel. Count all transmitted bits in both directions. The protocol is deterministic and must be correct on every input pair.

**Proposition.** Every such protocol has worst-case communication at least \(2\log_2 M\) bits. For \(M=2^b\), this bound is tight: each agent sends its own \(b\)-bit input once, so both can compute the sum with \(2b\) total transmitted bits.

**Proof.** A deterministic protocol transcript induces a combinatorial rectangle \(A\times B\) of input pairs. This rectangle property and the corresponding monochromatic-leaf lower bound are standard in communication complexity ([Roughgarden, *Communication Complexity*, §4](https://timroughgarden.org/w15/l/w15.pdf)). At a leaf, correctness requires every pair in \(A\times B\) to have the same sum. If \(A\) contained two distinct values \(x_1\ne x_2\), fixing any \(y\in B\) would give different sums \(x_1+y\ne x_2+y\), a contradiction; hence \(|A|=1\). Symmetrically, \(|B|=1\). Therefore each of the \(M^2\) input pairs reaches a distinct leaf. A binary protocol tree with \(M^2\) leaves has a path of length at least \(\log_2(M^2)=2\log_2M\). Sending each \(b\)-bit input once gives the matching upper bound when \(M=2^b\). \(\square\)

**Scope.** This is a worst-case deterministic binary-wire bound for exact outputs, not a lower bound in LLM tokens. It does not cover shared learned dictionaries, side information, lossy answers, or randomized protocols with error. Token cost, UTF-8 bytes, repeated prompt context, and inference cost remain separately measured. For uniformly distributed independent inputs, the transcript must distinguish all \(M^2\) input pairs, giving transcript entropy \(2\log_2 M\); under a prefix-free binary transcript, expected length is at least that entropy.

**Experimental prediction.** Use powers-of-two input domains with width \(b\in\{4,8,12,16\}\), and vary the number of bits required to describe each hidden input while fixing two agents and exact-sum distortion. A lossless wire protocol cannot go below \(2b\) bits total without shared input-dependent side information. Text formats may still differ substantially in bytes, model tokens, and decoder compute; any apparent sub-bound wire result signals an accounting-boundary or correctness error. This task tests whether a model-mediated protocol approaches a known communication floor, not whether new notation can beat it.

## 8. Necessary role gates for end-to-end communication

PrefixSum makes a distinction between conveying the sufficient statistic and executing the receiver's transformation explicit. For one episode define:

- \(A\): Agent 0 emits its exact required local output.
- \(M\): Agent 1 receives the correct sender subtotal before submitting.
- \(R\): Agent 1 emits its exact global prefix segment.
- \(J\): both agents' outputs are exact in the same episode.

For the fixed two-agent task and scorer, joint success requires all three component events:

\[
J = A \cap M \cap R.
\]

Consequently, without any independence assumption,

\[
P(J) \leq \min\{P(A),\;P(M),\;P(M)P(R\mid M)\}.
\]

The last term is simply \(P(M\cap R)\); it makes the receiver gate explicit. A hybrid with an oracle receiver measures the sender-side conjunction \(A\cap M\) under the chosen channel, while a hybrid with an oracle sender measures model receiver execution conditional on an oracle source and observed delivery. Neither hybrid alone estimates end-to-end success for two learned agents. Direct arithmetic calls go further by removing the simulator and tool scaffold, so they diagnose suboperations but do not estimate \(P(R\mid M)\) in a deployed exchange.

**Falsifiable eligibility prediction.** On the preregistered PrefixSum cells, if the estimated exact receiver rate conditional on a correct message is zero, then the observed sample provides no end-to-end headroom for a representation-only gain under that same receiver/interface; the bottleneck must first be changed or measured more precisely. If \(P(R\mid M)>0\), a communication format can still improve end-to-end utility by improving message acquisition/fidelity or receiver use, but a claim of efficiency must report each component and the joint result. This is a task-local screening condition, not a theorem that a protocol cannot affect receiver behavior: different representations can change \(P(R\mid M)\).

**Evidence status.** In v0.12, Qwen3-8B had 0 exact receiver outputs among 17/24 new ordinary-path receipts; the synthetic prefilled transcript also yielded 0/24. The v0.13 direct combined prompt yielded 7/24 exact outputs, so interface/context changes the receiver distribution and cannot be collapsed into a model-only ability parameter. These small deterministic-run samples are noisy diagnostics, not population probability estimates. v0.16 tests a larger local model before this gate is treated as a property of the task family.
