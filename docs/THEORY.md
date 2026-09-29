# Working theory: task-conditioned communication rate

**Status:** v0.4, updated 2026-09-30 with a cumulative-cost condition for stateful setup amortization, exact held-out communication bounds, and the modular-split scaling law. These definitions organize experiments; they do not prove that a particular representation is better.

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

### Error layers and feedback value

Distinguish transport corruption (the received bytes differ or a message is truncated), semantic decode error (bytes arrive intact but the receiver infers the wrong task-relevant content), and receiver/task execution error (the content is reconstructed but not applied correctly). Classical channel coding applies to an explicit transport-channel model; it does not by itself bound semantic error in an LLM decoder. A feedback/clarification message is an additional adaptive action and must be charged as a reverse-direction payload, a model call, repeated context, and latency. Evaluate its value at equal total budgets. The ICLR 2024 emergent-repair study shows that feedback can improve task generalization under synthetic channel noise while compositionality proxies decline, but it uses jointly trained RNN agents and a detectable noise token; see the [audit](../research/CHANNEL_ROBUSTNESS_AUDIT_V0_1.md).

### A conditional grounding threshold for an individual message claim

To connect message grounding to task utility, consider adding one atomic claim to an otherwise fixed message under a fixed sender policy, receiver, episode distribution, and scoring rule. Let event $B$ mean the claim is supported by the sender's observed state, true at send time, relevant to the scored task, and not already known by the receiver; let $q=P(B)$. Define $u_1=E[\Delta U\mid B]$ and $u_0=E[\Delta U\mid\neg B]$, where $\Delta U$ is the marginal task-utility change from including the claim rather than omitting it, with all other message content held fixed. The non-$B$ cases include repeated, irrelevant, unsupported, and false claims, whose effects need not be equal. Let $c\geq0$ be any additional resource opportunity cost converted to task-utility units by a preregistered deployment price; do not charge an action displacement again if its effect is already included in $\Delta U$. If there is no defensible scalar conversion for channel, latency, or compute, keep those costs as separate frontier dimensions and evaluate this condition within a fixed budget rather than inventing a weight.

The expected marginal utility is

\[
E[\Delta U_{net}] = q u_1 + (1-q)u_0 - c.
\]

**Proposition (single-claim threshold).** If $u_1>u_0$, adding the claim has positive expected net utility exactly when

\[
q > \frac{c-u_0}{u_1-u_0}.
\]

This follows by rearranging $q u_1+(1-q)u_0-c>0$; it is a decision threshold, not a theorem about natural or machine languages. In the special conservative case where a qualifying claim has mean gain $u_1=g>0$ and every non-qualifying claim has mean effect $u_0=-h\leq0$, it reduces to $q>(h+c)/(g+h)$. That simpler expression requires the non-qualifying cases to share this nonpositive mean; it must not be interpreted as saying every repeated or irrelevant fact is false or harmful. If $u_1\leq u_0$, increasing $q$ cannot improve expected utility under this two-class model. Multiple claims can interact, so summing this expression over mentions requires an additional additivity assumption and should not be done by default.

**Testable implication.** Under a frozen task, schedule, receiver, and stable conditional effects, changing only the rate $q$ predicts $\Delta E[U_{net}]=(q_2-q_1)(u_1-u_0)$. Measure event labels from task state and receiver knowledge, and retain task success as the primary outcome; do not use action-conflict reduction or belief overlap as a substitute. A comparison that only makes messaging free in simulator steps does not set the total inference, latency, or context cost to zero. The PARTNR dialogue study supplies a negative prior: it reports many unsupported entity mentions and lower task success despite fewer action conflicts, but its handle-counting method is only a proxy for semantic claim labels and its dialogue policy does not isolate message form.

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

This closed-form crossover assumes stationary per-episode costs. For stateful compression or adaptive protocols, let (C_L(H)) and (C_B(H)) be the cumulative costs through horizon (H); the exact condition is (A+C_L(H)\le C_B(H)). Do not extrapolate a finite-horizon average saving unless a stable marginal saving is justified. The Private Match dictionary follow-up found its stream savings nearly flat between 128 and 16,384 records, which falsified the linear break-even projection from the earlier 128-record sample in that specific setup; see the [horizon report](../research/PRIVATE_MATCH_COMPRESSION_HORIZON_V0_1.md).

### Vector-valued setup and selector break-even

The scalar condition is insufficient when setup and inference trade one resource for another. Let (j\in\mathcal J) index separately measured cost units, such as serialized channel bytes, tokens for a specific tokenizer, model calls, service seconds, and wall seconds. Let (A_j\ge0) be the one-time setup cost, (b_j) the baseline's stationary per-episode cost, and (\ell_j) the candidate's stationary per-episode cost. Write (\delta_j=b_j-\ell_j). The candidate is no more costly in every declared dimension after (H) episodes exactly when

\[
  A_j+H\ell_j\le Hb_j\quad\text{for all }j,
  \quad\Longleftrightarrow\quad
  H\delta_j\ge A_j\quad\text{for all }j.
\]

For positive integer reuse horizons (H), a finite componentwise crossover exists exactly when no dimension has (\delta_j<0) and every dimension with (\delta_j=0) has (A_j=0). Under that condition, the smallest horizon is

\[
  H^*_{\mathrm{vec}}=\max\left(\{1\}\cup
  \left\{\left\lceil\frac{A_j}{\delta_j}\right\rceil:\delta_j>0\right\}\right).
\]

If any dimension has (\delta_j<0), or has (\delta_j=0) while (A_j>0), there is no finite horizon at which the candidate is no more costly in every dimension. A candidate may still be useful on a Pareto frontier, but calling it “amortized” does not remove a persistent token, latency, or compute disadvantage. These statements concern cost only; the candidate must also preserve or improve task utility to dominate the baseline.

For a selector that evaluates (m) candidates before choosing one, (A_j) includes the actual selection work across all candidates (deduplicating genuinely shared setup artifacts), not only the winning candidate's work. For the Private Match natural-language selector, this makes (A_j) the combined development cost of concise- and short-template trials. The registered eight-episode horizon reports (A_j/8); it is a pilot deployment scenario, not a universal break-even estimate. Once per-episode evaluation costs are available, the vector equation tests whether selection setup can ever be repaid without inventing exchange rates between bytes, tokenizer units, calls, and time.

The exact-rational implementation is [`research/vector_setup_break_even.py`](../research/vector_setup_break_even.py). Its interface requires the three cost vectors to name the same dimensions and rejects floating-point values; tokenizers should be separate keys rather than combined counts.

**Falsifiable prediction P6 — no scalar break-even under a persistent trade-off.** If the selected representation requires positive selector setup in a tokenizer shared with the baseline and consumes at least as many inference tokens per episode in that tokenizer, then amortizing development selection can never make it componentwise dominate that baseline on the byte-and-token vector: bytes may cross their setup threshold, while the token dimension cannot. The end-to-end frontier should therefore retain a trade-off point or show domination by another arm, not report a single overall break-even. A measured token saving in every compared tokenizer removes this impossibility condition and makes a finite vector crossover testable, subject to the stationary-cost assumption.

**Proposition (cumulative reuse condition).** For a realized sequence of (H) transmissions, let (b_t) be the baseline's incremental cost and (ell_t) the reusable protocol's incremental cost at step (t), including state-dependent framing or encoding cost but excluding its fixed setup (A). Define (Delta_t=b_t-ell_t) and (S_H=sum_{t=1}^{H}Delta_t). Then the reusable protocol has lower total cost through (H) exactly when

\[
S_H \ge A.
\]

**Proof.** Baseline cost is (sum_{t=1}^{H}b_t). Reusable-protocol cost is (A+sum_{t=1}^{H}ell_t). Their difference is (S_H-A), so the reusable protocol is cheaper exactly when this difference is nonnegative. (square)

This condition also applies when each increment depends on preceding messages. If (Delta_t=0) after some (H_0), then (S_H=S_{H_0}) for all later (H); if (S_{H_0}<A), no finite extension repays setup. For stochastic task streams, report the distribution of (S_H-A) over independent sequences as well as its expectation; an expected crossover does not imply that most individual deployments cross at that horizon. A projected (H^*=A/\bar\Delta) is justified only under a stated stable-marginal-savings assumption, not merely because an initial finite sample had positive average savings.

## 5. Predictions to test

- **P1, task distortion:** at equal message budget, compact formats help only when omitted information is conditionally irrelevant to the receiver's optimal task action; errors should cluster around task-critical omissions, not message length itself. Where the centralized/no-message Bayes-risk gap is zero, no protocol should improve optimal task utility; a measured gain then indicates model/control mismatch, scorer noise, or a changed effective task/budget.
- **P2, receiver alignment:** for a fixed sender representation, downstream distortion depends on the receiver (r). A code's token/byte advantage should be separated from its decoder mismatch cost.
- **P3, interactive compression:** under a strict total budget, a second turn helps only if its expected value of information exceeds repeated context, prompt, and control overhead. Compare several turn allocations at a fixed total budget.
- **P4, setup horizon:** under stable per-episode costs, cumulative savings should cross at the measured (H^*=\lceil A/(c_B-c_L)\rceil). Under non-stationary/stateful costs, compare exact cumulative curves (A+C_L(H)) and (C_B(H)) instead; a finite average delta does not predict a crossover. Failure to cross at tested horizons is a bounded result, not proof that crossover never occurs.
- **P5, error recovery:** fixed redundancy should help mainly under the transport corruptions it is designed to detect; receiver clarification should help mainly when an intact message has task-relevant ambiguity. These mechanisms should not be assumed to improve an integrity-preserving channel. Any gain must survive matched total bidirectional bytes, inference tokens/calls, and a fixed round cap; separately score transport integrity, semantic fidelity, task execution, and final task utility. See the [channel-robustness audit](../research/CHANNEL_ROBUSTNESS_AUDIT_V0_1.md).

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

## 9. Private-query communication benchmark

This finite task isolates task uncertainty at the receiver. The sender observes (X=(X_1,\ldots,X_n)\), with independent uniform bits. The receiver privately observes a uniformly selected index (I\in\{1,\ldots,n\}\), and must output (X_I\). The sender does not observe (I\). The channel is noiseless, one-way, and restricted to one bit; no task-dependent setup, shared input-dependent advice, or extra reverse message is free.

**Proposition (one-bit random-access upper bound).** For any deterministic encoder (M=f(X)\in\{0,1\}) and any receiver decoder (\hat X_I=g_I(M)\), the average exact-answer probability obeys

\[
P[\hat X_I=X_I]\le \frac12+\frac{1}{2\sqrt n}.
\]

**Proof.** Represent the encoder output as (F=(-1)^M\) and each source bit as (S_i=(-1)^{X_i}\). Given (I=i\), the best decoder's success is (\frac12+\frac12|\mathbb E[S_iF]|\): for each of the two message values it chooses the more frequent bit. Therefore

\[
P[\hat X_I=X_I]=\frac12+\frac{1}{2n}\sum_{i=1}^{n}|\mathbb E[S_iF]|.
\]

The terms are the first-degree Fourier coefficients of the Boolean function (F\) on the uniform Boolean cube. Parseval gives (\sum_i(\mathbb E[S_iF])^2\le\mathbb E[F^2]=1\), and Cauchy–Schwarz gives (\sum_i|\mathbb E[S_iF]|\le\sqrt n\). Substitution proves the bound. A randomized encoder/decoder is a mixture of deterministic strategies and cannot exceed their maximum average success. \(\square\)

The bound is not claimed tight for finite (n\). A sender who knows the queried index can send (X_I\) in one bit and attain 1, but this changes the information boundary; if the receiver must reveal (I\), that reverse-direction message, call, and round are part of total cost. A sender-independent one-bit code is not a universal solution to all query tasks; it is judged under the declared query distribution and sender/receiver knowledge.

For the specific uniform-source, uniform-query classical (n\to1\) random access code above, a stronger exact result is already known: the deterministic majority encoder with identity decoder is optimal, with

\[
p^*(n)=\frac12+\frac{1}{2^n}\binom{n-1}{\lfloor(n-1)/2\rfloor}
\sim \frac12+\frac{1}{\sqrt{2\pi n}}.
\]

This theorem is due to established RAC literature, not a Tacit contribution ([Ambainis et al., 2009, §2.3](https://arxiv.org/abs/0810.2937)). It sharpens the Parseval upper bound above for this exact distribution and confirms that a single classical bit becomes nearly useless as the number of possible facts grows. The analytic scaling table in [`private_query_v0_3`](../experiments/private_query_v0_3/README.md) reproduces this law and the n≤4 exhaustive optima. It does not extend to skewed query distributions or to LLMs without further assumptions.

**Exact finite enumeration.** [`experiments/private_query_v0_1/`](../experiments/private_query_v0_1/README.md) enumerates every deterministic one-bit encoder for (n\le4\) and uses the Bayes-optimal coordinate decoder for each code. This checks the small-instance optima independently of an LLM and records a scaling diagnostic; it is not evidence about LLM language performance. The falsifiable asymptotic prediction is only that the one-bit task-oblivious optimum approaches chance as the number of independently queryable facts grows, consistent with the proved upper envelope. Learned/task-conditioned codes should be compared against (a) this frozen universal-code arm, (b) a query-conditioned oracle that explicitly accounts for how (I\) becomes available to the sender, and (c) a full-source communication upper bound.

### Exact budget frontier by reconstruction codebook

For fixed (n\), query weights (q_i\ge0\), and a deterministic decoder with at most (2^b\) messages, let (c_m\in\{0,1\}^n\) contain the answer the decoder returns for each possible query after receiving message (m\). The sender can map (x\) to its nearest (c_m\) under weighted Hamming distance (d_q(x,c)=\sum_i q_i\mathbf1[x_i\ne c_i]\). Conversely, any codebook of (2^b\) reconstruction vectors defines a valid encoder (choose the nearest vector) and receiver (output its queried coordinate). Hence the exact optimal success for a uniform source is

\[
J^*(n,b,q)=1-\frac{1}{2^n\sum_iq_i}\min_{\substack{\mathcal C\subseteq\{0,1\}^n\\|\mathcal C|\le2^b}}
\sum_{x\in\{0,1\}^n}\min_{c\in\mathcal C}d_q(x,c).
\]

This reduction turns an encoder/decoder search into finite binary quantizer design. It supplies an oracle frontier for validating finite-budget studies. It assumes the optimal codebook is already shared and ignores codebook search/storage/setup; these must be added to any operational system comparison. [`private_query_v0_2`](../experiments/private_query_v0_2/README.md) enumerates the exact n=2..4 frontiers under uniform and skewed query distributions and freezes full training-prior protocols for cross-prior evaluation. Changing (q\) changes the distortion objective and can change the optimal protocol even though the source distribution and bit budget remain fixed. Since multiple codebooks can tie on the training prior but differ on transfer, report the selection rule or the tied-optimum range rather than only one cherry-picked code.

## 10. Exact one-way coding floor for private record matching

This section formalizes the model-free calibration task in [`private_match_v0_1`](../experiments/private_match_v0_1/README.md). It gives a task-specific exact baseline, not a general lower bound for LLM communication.

**Model.** A record has d categorical features, each drawn from a shared vocabulary of size V; the record space is \(\mathcal{X}=[V]^d\), with \(V^d\) possible records. The receiver privately sees an ordered table \(S=(x_1,\ldots,x_n)\) of distinct records, where \(2\le n\le V^d\). The sender sees one target record \(X\in S\), but not S, its ordering, or the candidate IDs. The table is generated independently of a uniform target index \(J\in\{1,\ldots,n\}\), and \(X=x_J\). The sender sends one noiseless binary message; the receiver must return the ID of the row equal to X. The feature vocabulary and schema are shared for free in this idealized bound; their setup and serialization costs are outside the payload floor and must be charged operationally.

**Proposition (exact worst-case one-way payload).** Any deterministic zero-error encoder that does not depend on the receiver's table requires at least \(V^d\) distinct messages, hence at least

\[
\left\lceil\log_2(V^d)\right\rceil
=\left\lceil d\log_2 V\right\rceil
\]

fixed-length bits in the worst case. This is achievable by sending a fixed shared rank/code for the complete target tuple, so the bound is exact.

**Proof.** Take any two distinct records (x\ne x') in \(\mathcal{X}\). Since (2\le n\le |\mathcal{X}|\), some valid receiver table contains both. If the encoder mapped them to the same message, the receiver would observe the same table and message in two cases, but the required row IDs differ. A single decoder output cannot be correct in both cases. Thus the encoder must be injective on all \(|\mathcal{X}|=V^d\) possible records and needs at least that many messages. Conversely, a shared enumeration of \([V]^d\) assigns a distinct rank to every record; sending its rank uses \(\lceil\log_2(V^d)\rceil\) bits and lets the receiver match it against its table. \(\square\)

The no-message Bayes accuracy under the stated uniform target-index prior is exactly \(1/n\), while a centralized receiver given the target record scores 1.0. The achievable rank-code floor assumes the vocabulary ordering is already shared. A compact variable-length or learned code can improve *average* bytes under a nonuniform record distribution, or trade exactness for expected task utility, but cannot beat this worst-case zero-error payload floor without changing the side-information boundary, adding setup, or allowing error. LLM tokens are not bits; every experimental result must also report actual serialized payload bytes, receiver-tokenizer use, complete inference cost, and task success.

**Falsifiable implementation check.** On fresh tasks, compare the fixed-width rank code, optimized natural language, JSON, delimited tuples, and any learned code under exact-answer scoring. If a purported zero-error one-way protocol transmits fewer than \(\lceil d\log_2V\rceil\) payload bits on a worst-case-valid task while still covering all possible tuples, either its shared state, task distribution, error rate, or measured channel boundary differs from this model. Report that difference explicitly. This task calibrates encoding efficiency and receiver use; it does not test multi-turn dialogue or broad reasoning.

## 11. Amdahl limit for shared-state prefill reuse

Prompt Choreography can reuse key/value encodings already resident in a compatible model runtime. That reduces one component of inference work; it does not shorten the logical message or its serialized network payload. The following is the ordinary Amdahl decomposition applied to that deployment setting, not a new communication-complexity theorem. See the [source audit](../research/PROMPT_CHOREOGRAPHY_AUDIT_V0_1.md).

**Model.** Let baseline end-to-end wall time be `T`. Let `p` in `[0,1]` be the fraction spent re-encoding context that the shared cache would reuse. Suppose reuse accelerates that fraction by `s >= 1`, leaves other work and outputs unchanged, and adds cache/masking overhead `h*T`, where `h >= 0`. Then normalized optimized time is

`Tprime/T = (1-p) + p/s + h`, and `S_e2e = 1 / ((1-p) + p/s + h)`.

Reuse improves end-to-end latency if and only if `h < p(1 - 1/s)`. With zero overhead, the maximum speedup is bounded by `1/(1-p)`, even if repeated prefill becomes free. For example, if reusable prefill was only 5% of baseline runtime and that part becomes 3x faster, total speedup is at most `1/(0.95 + 0.05/3) = 1.034` (3.4%).

**Falsifiable prediction.** Within a fixed-output, fixed-call-count workflow, measure baseline repeated-prefill fraction `p`, its local speedup `s`, cache/masking overhead `h`, and end-to-end time. The formula predicts the total latency ratio from those quantities. If observed speedup exceeds the bound, another component changed (e.g. decoding, scheduling, output length, or batching), or the measured boundary is incomplete; attribute and record that change rather than crediting cache reuse alone. If measured overhead reaches `p(1 - 1/s)`, the cache cannot improve end-to-end time under this model.

**Scope.** This predicts compute/latency tradeoffs for compatible shared-runtime workflows. It says nothing about semantic fidelity, privacy, serialized KV-transfer bytes, or whether a compact message language is better. Those require separate task and transport outcomes.

## 12. Exact one-way floor for receiver-private candidate sets

This bound applies to the held-out receiver-utility task in [`emergent_ood_v0_2`](../experiments/emergent_ood_v0_2/README.md). It distinguishes the number of choices the receiver sees from the number of possible targets the sender must encode.

**Model.** There are (n) possible target meanings. The sender observes only the target (x\in[n]); the receiver privately observes a candidate set (C\subseteq[n]) of fixed size (k\), where (2\le k\le n\), and is promised (x\in C\). Every (k)-subset is a valid receiver view. A deterministic one-way protocol sends a message (m=f(x)), after which the receiver must identify (x) exactly from ((m,C)). The sender does not observe (C). Assume any shared codebook is already available to both agents; codebook distribution/setup is excluded from this payload-only floor and is charged in operational experiments.

**Proposition.** Any zero-error protocol requires at least (n) distinct messages and therefore at least \(\lceil\log_2 n\rceil\) worst-case fixed-width payload bits. A fixed shared index code attains the bound. Thus, for the v0.2 task with (n=9\), the ideal fixed-width floor is 4 bits for every candidate count (2\le k\le9\), even though the receiver chooses among only (k\) candidates.

**Proof.** For any two distinct targets (x,y\in[n]\), there exists a size-(k\) candidate set containing both because (k\ge2\) and (k\le n\). If (f(x)=f(y)\), then on that same receiver set the decoder receives identical inputs in the (x) and (y) cases, but exact correctness requires different outputs. This is impossible; hence (f) must be injective over all (n\) targets. Therefore it needs at least (n\) messages. Conversely, assigning each target a shared fixed-width binary index and sending its index uses \(\lceil\log_2 n\rceil\) bits; the receiver selects the matching candidate. \(\square\)

**Falsifiable implication.** Varying candidate count (k\) changes the no-message Bayes accuracy to exactly (1/k\) in the balanced v0.2 construction, but it does not lower the worst-case zero-error one-way **fixed-width** payload floor while the sender still lacks (C\) and every candidate subset remains possible. A 3-bit **fixed-width** code when (n=9\) and (k=5\) would need extra shared input, allow errors, or change the fixed-width boundary; otherwise it violates this model. The bound charges neither codebook setup nor LLM decoding compute and is not an LLM-token bound. Compare observed protocols against the 4-bit fixed-width symbolic reference only after adding their actual serializer, setup, semantic fidelity, receiver decoding, and inference costs.

### Exact floor for a fixed candidate-set suite

The complete-support assumption above can be too strong for a small, fixed evaluation suite. Given its candidate sets, construct a graph (G) whose vertices are the target meanings appearing in the suite and whose edges connect any pair that co-occurs in at least one candidate set.

**Proposition.** For deterministic one-way communication with zero error on this fixed suite, the minimum number of message symbols is exactly the chromatic number (χ(G)); the minimum fixed-width binary payload is therefore (\lceil\log_2χ(G)\rceil) bits.

**Proof.** A zero-error encoder cannot assign the same message to adjacent meanings: on a candidate set containing both, the receiver would see the same message and same candidate set for either target. Hence every valid encoder induces a proper coloring, requiring at least (χ(G)) symbols. Conversely, send the proper color of the target. Each candidate set is a clique in (G), so its meanings have distinct colors; the receiver selects the unique candidate matching the received color. Thus (χ(G)) symbols suffice. The complete graph case gives (χ(G)=n), recovering the full-support bound. (\square)

**Scope and prediction.** This is a deterministic, zero-error, one-way payload result for a fixed, known candidate suite; it excludes the cost of sharing the coloring, and it is not an LLM-token or lossy-accuracy bound. A small random suite may have a much smaller (χ(G)) than its target support size, making an oracle code deceptively cheap. Report graph coverage and this finite-suite floor alongside the full-support (n)-symbol reference. Increasing candidate-set coverage should move (χ(G)) upward toward (n); if it does not, inspect the candidate generator or support assumptions. The v0.4 fixture must not call the full-support bit count a strict lower bound on its realized sample unless every target pair co-occurs.

### Lossy accuracy reference from random binning

The zero-error chromatic-number floor does not describe how quickly task accuracy rises as a finite message alphabet grows when collisions are allowed. Random binning is established in source coding with decoder side information (for example, [Merhav 2015](https://arxiv.org/abs/1507.01255)); universal hash families provide another way to approximate collision-controlled random assignments ([Carter and Wegman 1979](https://doi.org/10.1016/0022-0000(79)90044-8)). The finite one-shot decision formula below is derived directly for our candidate-set task; it is not claimed as a theorem from those papers. Consider a fixed candidate set $C$ with $k$ distinct meanings and a uniform target $Y\in C$. A sender maps each meaning to one of $M$ message symbols, without seeing $C$. For a fixed encoder $f$, a Bayes-optimal receiver that sees both $C$ and $f(Y)$ can choose one representative from every code class present in $C$, so its exact-selection accuracy on that set is

\[
a(f,C)=\frac{|f(C)|}{k}.
\]

Now draw the shared encoder as an ideal random function $F$ that independently assigns every meaning a uniform symbol in $[M]$. For any one of the $M$ symbols, the probability that it is absent from $C$ is $(1-1/M)^k$. Linearity of expectation gives

\[
\mathbb{E}_F[a(F,C)] = \frac{M\left(1-(1-1/M)^k\right)}{k}.
\]

This expectation depends only on $k$, not on which candidate meanings appear or how the fixed suite samples its candidate sets. For $M=1$ it recovers the no-message prior $1/k$. For a power-of-two alphabet $M=2^b$, it is an exact expected-accuracy reference at a $b$-bit fixed-width payload cap. Example: for $k=4$, $b=0,1,2,3$ gives $1/4$, $15/32$, $175/256$, and $1695/2048$ respectively. This is a lossy **coding baseline**, not a language, not a model result, and not a claim that a random mapping is useful to an LLM.

**Proof of the receiver rule.** Let the code classes intersecting $C$ have sizes $n_1,\ldots,n_r$. Conditional on receiving class $j$, the target is uniform over those $n_j$ meanings because the prior is uniform and the encoder is deterministic. Any decoder therefore succeeds with probability at most $1/n_j$ on that class, attained by choosing any one candidate in it. Averaging over the $k$ possible targets, each occupied class contributes exactly one correct target, so accuracy is $r/k=|f(C)|/k$. For the random-function calculation, write $|F(C)|=\sum_{m=1}^{M}\mathbf{1}\{m\text{ appears in }F(C)\}$ and take expectations.

**Operational boundary.** The derivation assumes the same target-independent random codebook is already available to both endpoints. Its seed/codebook distribution, serialization, model instructions or codec runtime, latency, and compute are excluded. A keyed hash is only a pseudorandom approximation to an ideal random function; a family with only pairwise-independent outputs does not in general justify the exact occupancy formula for $k>2$. Treat this as an oracle payload reference until a concrete shared-codebook implementation is tested and fully charged. The executable exact-rational calculator [`candidate_set_random_code.py`](../research/candidate_set_random_code.py) and exhaustive small-alphabet regression test make the curve reproducible.

**Falsifiable prediction P11.** On uniformly targeted candidate sets of size $k$, averaging a truly random codebook over independent draws must match the formula at each $M$; a systematic discrepancy indicates target-prior imbalance, target-dependent codebook selection, non-independent labels, or an incorrect decoder model. A semantic protocol's claim of value should be tested against this random-binning reference as well as the no-message and zero-error code, while keeping payload-only and complete LLM/system costs separate.

### Fixed-width, prefix-free, and framed payloads are different bounds

The 4-bit statement above is specifically a worst-case **fixed-width payload** bound. It must not be presented as the minimum for every variable-length wire protocol. If the communication medium is a concatenated bitstream and each message must be self-delimiting, a binary prefix code is appropriate. Under the uniform target marginal induced by enumerating every target in every k-subset, a Huffman code for nine meanings has seven codewords of length 3 and two of length 4: expected payload is `29/9 ≈ 3.222` bits and maximum payload is 4 bits. The length multiset satisfies Kraft equality, and Huffman's algorithm minimizes expected prefix-code length for this source distribution ([Huffman 1952](https://doi.org/10.1109/JRPROC.1952.273898)). Regenerate these finite references with `python experiments/emergent_ood_v0_2/code_bounds.py --symbols 9`; regression checks are in `tests/test_emergent_ood_code_bounds.py`.

If the transport provides an observable packet boundary, a decoder can instead distinguish arbitrary variable-length payload strings, including prefix-related strings. Ignoring the cost of conveying that boundary, the nine shortest **non-empty** binary strings comprise two 1-bit, four 2-bit, and three 3-bit strings, totaling 19 payload bits over nine equiprobable targets (mean `19/9 ≈ 2.111`, maximum 3). This is only a payload-only combinatorial reference: a real protocol must count its packet delimiter, length field, envelope, or other framing signal. If the boundary is not free, those bits can erase this apparent saving; actual `tlu.costs.v3` serialized bytes remain authoritative. An empty payload is excluded because a zero-byte event could otherwise encode a target through the mere presence of a message.

**Prediction.** When message boundaries are supplied out of band, a variable-length code can reduce mean payload below the 4-bit fixed-width reference while retaining exact decoding. When boundaries must be serialized, the total-byte frontier improves only if length-dependent payload savings exceed the measured framing premium. Test both accounting scopes explicitly; neither bit count predicts LLM token use, semantic fidelity, codebook setup, or receiver compute.

## 13. Exact average-accuracy frontier for private record matching

The worst-case zero-error floor above does not describe optimal expected accuracy when a protocol may confuse some records. The uniform candidate-table distribution in Private Match v0.2 admits an exact finite frontier.

**Model.** Let the record space contain (N=V^d) records. The receiver observes a uniformly sampled size-(k) subset (S) without replacement. The target (X) is selected uniformly from (S); the sender observes only (X), not (S), and sends one noiseless symbol from a shared alphabet of at most (B) symbols. The receiver sees that symbol and the full candidate table, and is scored on the exact target row. The shared codebook is free in this information-theoretic reference; setup and serialized framing are charged separately in operational comparisons.

**Proposition (exact finite-budget Bayes frontier).** Put (B'=min(B,N)). Write (N=B'a+r), with (0\le r<B'). The optimal encoder partitions records into (B'-r) classes of size (a) and (r) classes of size (a+1). Its optimal expected exact-match probability is

\[
P^*(N,k,B)=\frac{1}{k}\left[B'-\frac{(B'-r)\binom{N-a}{k}+r\binom{N-a-1}{k}}{\binom{N}{k}}\right],
\]

where (\binom{m}{k}=0) for (m<k). Thus (P^*(N,k,1)=1/k) and (P^*(N,k,N)=1). A fixed-width (b)-bit payload uses (B'=min(2^b,N)) symbols in this bound.

**Proof.** For any encoder, its message classes partition the records into sizes (s_1,\ldots,s_{B'}). Given a message and the receiver's table, candidates in the same class have identical likelihood, so a Bayes decoder's success for a target in a class of size (s) is the reciprocal of the number of same-class candidates in the table. For a fixed target, the number of colliding other candidates is hypergeometric with population (N-1), marked size (s-1), and draw count (k-1). Using (\binom{s-1}{j}/(j+1)=\binom{s}{j+1}/s) and Vandermonde's identity, the expected reciprocal class size is

\[
\mathbb E\left[\frac{1}{1+J}\right]=\frac{\binom{N}{k}-\binom{N-s}{k}}{s\binom{N-1}{k-1}}.
\]

Weighting by the (s) equally likely targets in that class gives contribution (\frac1k[1-\binom{N-s}{k}/\binom Nk]). Summing classes yields

\[
P=\frac{1}{k}\left[B'-\frac{\sum_i\binom{N-s_i}{k}}{\binom Nk}\right].
\]

The function (h(s)=\binom{N-s}{k}) is discretely convex: its second forward difference is (\binom{N-s-2}{k-2}\ge0), taking out-of-domain binomial coefficients as zero. Therefore (\sum_i h(s_i)) is minimized, for fixed sum (N), by balanced integer class sizes. Substitution yields the stated optimum, attained by the balanced encoder and Bayes decoder. Randomized encoders and decoders are mixtures of deterministic strategies and cannot exceed this optimum. \(\square\)

This task-specific result predicts the ideal success ceiling as distinguishable-message count grows, before model errors, codebook setup, and channel framing. It is neither a universal language bound nor an LLM result. The exact rational calculator is [`experiments/private_match_v0_2/code_bounds.py`](../experiments/private_match_v0_2/code_bounds.py); tests compare the formula to exhaustive encoder enumeration on small spaces. Future codec trials should report this oracle next to no-message, full-information, and operational formats, while keeping bits, UTF-8 bytes, and model tokens as separate cost axes.

### Scaling with choice-set size and message capacity

For fixed (k) and (B), as (N\to\infty) the balanced class sizes grow and the hypergeometric collision probability converges to independent collisions with probability (1/B). The exact frontier therefore converges to

\[
P^*_\infty(k,B)=\frac{B}{k}\left[1-\left(1-\frac1B\right)^k\right].
\]

This yields two scaling predictions. With (k) fixed and (B\to\infty),

\[
P^*_\infty(k,B)=1-\frac{k-1}{2B}+O(B^{-2}),
\]

so the ideal error falls inversely with the number of message symbols. If both scale with fixed ratio (\lambda=k/B), then as (k,B\to\infty),

\[
P^*_\infty(k,B)\to\frac{1-e^{-\lambda}}{\lambda}.
\]

Thus keeping a constant number of payload bits while the number of candidate choices grows gives success proportional to (B/k); keeping a fixed bits-per-choice ratio instead gives a nonzero limiting accuracy below one. These are scaling laws for the ideal private-record task and balanced codebooks, not for model inference or natural-language length. The calculator reports both the exact finite-(N) value and this large-record-space limit. A falsifiable model experiment must vary (k) and measured serialized-byte budgets while holding the record prior and receiver capability fixed; deviation from the oracle can arise from codebook, model, or framing constraints, all of which must be reported rather than folded into a language claim.

## 14. Exact bit-budget frontier for complementary coordinate matching

This result formalizes the three-agent [Private Match v0.3](../experiments/private_match_v0_3/README.md) task. It is a finite simultaneous-communication bound for that task family, not a language-efficiency or LLM-performance theorem.

**Model.** Let `q = 2^w`, with integer `w >= 1`. Independent uniform coordinates `X,Y ∈ [q]` define the hidden target row `(X,Y)`. The receiver observes the complete `q × q` Cartesian table, with candidate order and IDs randomized independently of the target and source values. Sender X observes only `X`, Sender Y only `Y`; they each send one simultaneous, noiseless, fixed-width binary message of `b_x` and `b_y` bits. The receiver knows the sender slot from a fixed schedule and must output the exact target row ID. There is no target-dependent shared state, feedback, or task-dependent codebook setup in the bound. The total payload cap `B` is a nonnegative integer and requires `b_x + b_y ≤ B`.

This is an ideal probability model: table order, IDs, and target are independent-uniform draws. The executable generator uses a secret-key HMAC pseudorandom function with separate domains and unbiased rejection sampling; for a fixed key and seed, an episode is deterministic. Thus the proposition is exact for the declared ideal prior, while generated shards are computational pseudorandom realizations under the HMAC PRF assumption rather than fresh literal random draws on each run.

**Proposition (exact finite-budget Bayes frontier).** The maximum exact-match probability is

\[
A^*(q,B)=\max_{b_x+b_y\le B}\frac{\min(q,2^{b_x})\min(q,2^{b_y})}{q^2}
=\frac{2^{\min(B,2w)}}{q^2}.
\]

**Proof.** For fixed encoders `f:[q] → {0,1}^{b_x}` and `g:[q] → {0,1}^{b_y}`, let `K_x = |im(f)|` and `K_y = |im(g)|`. Every message pair `(u,v)` with nonempty source classes corresponds to one Cartesian block `f⁻¹(u) × g⁻¹(v)`. Conditional on that message pair and the receiver's complete table, the target is uniform over that block. A decoder can select at most one row from the block, so its unconditional success contribution is at most `1/q²`, with equality when it outputs a row from the block. All `K_x K_y` source-class pairs are possible, and a decoder can select one row from each block, giving optimal success `K_x K_y / q²`. A `b_x`-bit encoder has at most `min(q, 2^{b_x})` classes, attainable by partitioning `[q]` into that many nonempty classes; likewise for Y. Maximizing the product under the integer budget gives the first expression. Since `q = 2^w`, the numerator is `2^{min(w,b_x)+min(w,b_y)}`. The largest feasible exponent is `min(B,2w)`, attained by allocating at most `w` bits to each source, which proves the second expression. Randomized encoders or decoders are mixtures of deterministic strategies and cannot exceed the deterministic maximum. ∎

The endpoint `B = 0` gives the no-message Bayes accuracy `1/q²`. At `B = w`, one source can be encoded exactly and the frontier is `1/q`; at `B = 2w = log₂(q²)`, both coordinates are exact and success is 1. For `q = 4`, the values for `B = 0,1,2,3,4` are `1/16, 1/8, 1/4, 1/2, 1`. The executable exact-rational calculator is [`bit_frontier.py`](../experiments/private_match_v0_3/bit_frontier.py); its q=4 results are checked against exhaustive encoder-pair enumeration in [`test_private_match_v03.py`](../tests/test_private_match_v03.py).

### Expected prefix-free payload cost at the uniform prior

Allow each sender to use a binary prefix code instead of a fixed-width code, and count payload bits while requiring each message to be self-delimiting. Exact reconstruction still requires q distinct codewords per sender: if two coordinate values share a codeword, fixing the other coordinate yields an indistinguishable transcript for two different target rows. Since each coordinate is uniform, Shannon's source-coding bound gives expected prefix length at least (H(X)=log₂q=w) for each source. The fixed-width w-bit code attains this bound, so the minimum expected prefix-free payload is 2w bits total. For q=2^w, a variable-length prefix code cannot improve expected payload bits over the fixed-width reference; variable codeword lengths may only tie when all used lengths remain w.

This is a payload-only result. A transport that supplies message boundaries out of band changes the coding model; if it serializes a length, delimiter, or envelope, those framing bytes remain part of the measured channel cost. Neither case predicts LLM-token count, instruction cost, decoder error, or inference compute. The entropy bound follows Shannon's source-coding theorem ([Shannon 1948](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x)).

**Falsifiable implication.** On the registered uniform q=4 task, a variable-length prefix-code baseline cannot have expected source payload below 4 bits across the two senders. If a measured protocol appears to do so, inspect whether it uses packet-boundary side information, a nonuniform task prior, shared task-dependent state, or an uncounted framing signal. An operational gain from variable-length serialization would have to come from measured framing/instruction/token effects, not a lower prefix-code entropy cost.

### Lossy task-specific prefix coding under an expected payload budget

The zero-error entropy argument does not settle the frontier when the receiver may return a wrong row. For deterministic one-shot senders, partition each sender's `q` equally likely coordinate values into `K` nonempty classes and assign one binary prefix codeword to each class. The receiver maps each class pair to one representative target row. Under the uniform Cartesian prior, there are `K_x K_y` nonempty transcript cells, each contributing one correct guess among `q²` equally likely target rows, so exact-row success is `K_x K_y/q²`. Randomized mixtures across codebooks or episodes can convexify this pure-strategy set and are not included in the enumerator.

For a fixed class-size multiset `(n_1,…,n_K)`, the minimum expected payload is the Huffman length `Σ_i n_i ℓ_i/q`. Enumerating the integer partitions of `q` therefore gives the exact minimum expected prefix payload for each `K`; pairing sender choices and removing dominated points yields the deterministic one-shot expected-payload/success frontier. [`prefix_frontier.py`](../experiments/private_match_v0_3/prefix_frontier.py) implements this finite oracle for `2 ≤ q ≤ 32` using exact rational arithmetic. It is a task-specific functional-compression baseline: it assumes both endpoints already share the codebook, the prior, the schedule, and the representative rule. It is not a learned or general-purpose language, and excludes randomized mixtures, framing, instructions, model tokens, and inference cost.

For `q=4`, the three-class partition has sizes `(2,1,1)` and an optimal code has lengths `(1,2,2)`, giving expected payload `3/2` bits per sender, worst-case 2 bits per sender, and `3/4` exact coordinate recovery. With three classes at both senders, expected payload totals 3 bits, worst-case payload totals 4 bits, and task success is `9/16`. This exceeds the fixed-width reference's `1/2` success at a **hard maximum** of 3 bits, but those budgets differ: the prefix point charges expected payload and admits a 4-bit episode. The comparison only exposes an expected-rate/loss trade-off; it does not show a same-budget win or a practical wire-cost reduction. At zero error, each sender needs four classes and the prefix entropy lower bound still gives 4 expected bits total.

**Falsifiable prediction P8 — bounded-error prefix frontier.** Under the independent uniform `q=4` prior and a shared, free, self-delimiting codebook, the deterministic one-shot expected-payload frontier contains `(3 bits, 9/16 success)` and the point's worst-case payload is 4 bits. An exhaustive small-alphabet enumeration must agree with the partition/Huffman oracle. Randomized mixtures are outside the prediction and may add convex combinations. Under evaluation with all setup, framing, and model costs charged, this point predicts no representation advantage by itself; a practical advantage requires either an improved complete cost/success frontier or transfer beyond this task. Changing the prior, allowing boundary side information, or changing the receiver loss invalidates the registered numbers and requires a new derivation.

### Shared-randomness convexification under an expected budget

Suppose a public random seed, independent of the task, selects one complete deterministic protocol before each episode and is known to both senders and the receiver. If protocol `i` has expected payload `C_i` and task success `P_i`, a mixture with probabilities `λ_i` has expected pair `(Σ_i λ_i C_i, Σ_i λ_i P_i)`. Conversely, every finite convex combination is implemented by this shared lottery. Therefore the optimum under an *average* payload constraint is the upper concave hull of the deterministic cost/success points, provided the common coin and protocol setup are free. This statement concerns expected task-level cost; it does not satisfy a per-episode hard cap and does not make setup or coin distribution free in a real system.

For `q=4`, the deterministic prefix points all lie below the chord from no message `(0,1/16)` to full coordinate disclosure `(4,1)`. The hull therefore randomizes between these endpoints: send both complete 2-bit coordinates with probability `B/4`, and otherwise send nothing. At expected payload budget `B=3`, this reaches `1/16 + (3/4)(1 - 1/16) = 49/64` success, while 25% of episodes still use zero bits and 75% use 4 bits. It dominates the deterministic 3-bit-expected prefix point `(3,9/16)`, but only because the comparison permits shared randomness and a 4-bit tail. A fixed-width protocol with the same common lottery attains the same bound, so this is a budget-policy baseline, not a variable-length-language gain.

[`prefix_frontier.py`](../experiments/private_match_v0_3/prefix_frontier.py) reports both the deterministic Pareto points and their free-shared-randomness upper concave hull using exact rational arithmetic. Its interpolation helper predicts success at any expected payload budget on that hull. Operational comparisons must separately charge seed/codebook setup, model instructions and calls, framing, latency, and worst-case per-episode cost. In particular, if deployment or evaluation caps every episode, use the hard-cap frontier and do not claim the expected-budget convexification as attainable.

**Falsifiable prediction P9 — q=4 shared-randomness upper bound.** Under the same uniform task and a free task-independent shared coin, the exact prefix-oracle hull has only the no-message and full-disclosure vertices. At expected payload 3 bits its success is `49/64`, achieved by full disclosure in 3/4 of episodes and no message in 1/4; no deterministic intermediate point reaches that value. An independent exact enumeration of all class partitions and all convex combinations must reproduce this value. This bound is ineligible as an operational result unless the complete coin/selection and protocol costs are included and the evaluation genuinely constrains expected rather than per-episode payload.

### Exact shared-randomness frontier for powers-of-two alphabets

The q=4 chord generalizes to every registered alphabet `q=2^w`. Consider any deterministic sender encoder that partitions its uniform `q`-value coordinate into `K` nonempty message classes. Let `H` be the entropy in bits of the class message. The most concentrated class-size distribution for fixed `K` is `(r,1,…,1)/q`, where `r=q-K+1`; entropy is Schur-concave, so

\[
H \ge \log_2 q - \frac{r}{q}\log_2 r
  \ge \log_2 q\,\frac{K-1}{q-1}.
\]

For the second inequality, set `g(x)=x ln(x)/(x-1)` with `g(1)=1`. Its derivative is `(x-1-ln x)/(x-1)^2 ≥ 0`, so `g(r)≤g(q)`, which is equivalent to the displayed bound.

For the two senders, write `u=(K_x-1)/(q-1)` and `v=(K_y-1)/(q-1)`. Under the uniform Cartesian prior and exact-row loss, the best decoder succeeds with probability `K_x K_y/q²`: each nonempty class-pair cell contains one selected correct row among q² equally likely targets. Algebra gives

\[
\frac{K_xK_y-1}{q^2-1}
 =\frac{u+v+(q-1)uv}{q+1}
 \le \frac{u+v}{2}
 \le \frac{H_x+H_y}{2\log_2 q}
 \le \frac{C}{2\log_2 q},
\]

where `2uv≤u+v` for `u,v∈[0,1]`, `H_x+H_y` is the sum of source-message entropies, and `C` is the sum expected self-delimiting prefix payload. The final inequality is Shannon's source-coding bound. Therefore every deterministic protocol, and every mixture of them, satisfies

\[
P_{\rm exact} \le \frac{1}{q^2}+
  \left(1-\frac1{q^2}\right)\frac{C}{2\log_2 q}
\quad (0\le C\le 2\log_2 q).
\]

For `q=2^w`, this bound is tight at every expected budget `C`: a shared task-independent coin selects full fixed-width disclosure (cost `2 log₂ q`, success 1) with probability `C/(2 log₂ q)`, and silence otherwise (cost 0, success `1/q²`). Thus the exact average-payload frontier with free common randomness is the endpoint chord; no intermediate deterministic or randomized prefix protocol can exceed it. Under a per-episode hard cap, this lottery is unavailable and the integer fixed-width frontier remains the relevant reference. If codebook/coin setup, model calls, tokens, or tail costs are charged, this payload-only theorem is only an upper-bound control. The result is preregistered as P10 and numerically checked by the finite oracle for `q=2,4,8,16,32`; those finite checks supplement, but do not replace, the proof.

**Falsifiable prediction P10 — power-of-two expected-budget optimum.** For every power-of-two `q`, exact-row success under any deterministic prefix encoding with a free shared random mixture and mean payload budget `C` cannot exceed the affine bound above; the silence/full-disclosure lottery attains equality. Finite enumeration for `q≤32` must report only the two endpoint hull vertices. A counterexample must identify a violated assumption or refute the entropy/class-count inequality. The statement makes no claim about equal complete LLM cost, independent private coins, or per-episode caps.

**Scaling prediction.** For fixed `B` and `q ≥ 2^{B/2}`, `A*(q,B) = 2^B/q²`, so ideal accuracy falls as `q⁻²` while receiver-table size grows as `q²`. If payload scales as a fraction `ρ ∈ [0,1]` of the zero-error budget, `B ≈ 2ρ log₂(q)`, then ideal success scales as `q^{2ρ−2}`, up to integer-budget rounding. These are predictions for the noiseless payload-only task; table-context tokens, framing, codebook exposition, model errors, and inference compute are excluded. They are not an LLM scaling law. An empirical study should vary q and total payload caps, measure complete model and channel costs separately, and compare outcomes with this oracle rather than treating the ideal curve as an expected model result.

## 15. Exact frontier for independent non-uniform coordinate priors

Section 14 assumes uniform X and Y. When independent coordinate values have unequal frequencies, the optimal fixed-width partition should preserve high-probability values and merge lower-probability values. This result generalizes the triadic frontier without claiming that empirical LLMs will realize the optimal partition.

**Model.** Let X and Y be independent with finite supports of sizes `q_x` and `q_y`, and exact rational marginal probabilities `p_X(x)` and `p_Y(y)`. The receiver observes the complete Cartesian candidate table and a uniform random candidate ID assignment independent of the hidden target. Each source sends one simultaneous noiseless fixed-width message; the schedule identifies each sender slot. The receiver outputs the most probable target row consistent with the two messages. The shared codebook is free in this payload-only bound; its storage, explanation, and inference costs belong in an operational comparison.

Write the marginal probabilities in decreasing order as `p_X↓(1) ≥ … ≥ p_X↓(q_x)` and `p_Y↓(1) ≥ … ≥ p_Y↓(q_y)`, and define `S_X(K)=Σ_(i=1)^K p_X↓(i)` and `S_Y(K)=Σ_(j=1)^K p_Y↓(j)`. For a total integer payload budget `B`, the exact Bayes frontier is

\[
A^*_{p_X,p_Y}(B)=\max_{b_x+b_y\le B}
S_X(\min(q_x,2^{b_x}))\,S_Y(\min(q_y,2^{b_y})).
\]

**Proof.** Fix any sender partitions into `K_x` and `K_y` nonempty message classes. For one class pair `A × C`, independence makes the most probable row have mass `max_(x∈A) p_X(x) × max_(y∈C) p_Y(y)`. Summing over all class pairs factorizes, so the Bayes success is

\[
\left(\sum_A \max_{x\in A}p_X(x)\right)
\left(\sum_C \max_{y\in C}p_Y(y)\right).
\]

For a source with K classes, each class contributes the probability of one representative, so the sum cannot exceed the total mass of its K most probable values. This bound is attained by making the top K−1 values singleton classes and placing every remaining value in the final class, whose maximum is the Kth value. A `b`-bit message permits at most `min(q,2^b)` nonempty classes. Applying the one-source optimum independently to X and Y and maximizing over bit allocations proves the result. ∎

For uniform marginals, `S_X(K)=K/q_x` and `S_Y(K)=K/q_y`, recovering Section 14 when `q_x=q_y=q=2^w`. With a fixed payload budget, the optimal allocation depends on the two marginal tails: another bit is valuable where it increases retained probability mass most. Thus a task-specific code can beat a uniform partition without establishing a reusable language advantage; frequency discovery and codebook setup must be charged, and transfer to a changed prior must be measured.

**Executable check and prediction.** `optimal_nonuniform_success_probability(...)` in [`bit_frontier.py`](../experiments/private_match_v0_3/bit_frontier.py) computes this frontier exactly from rational priors. Exhaustively enumerate all encoder pairs on small supports and compare their Bayes success with the formula. Under a shift toward a flatter prior, an encoder optimized for a skewed development prior should lose some of its in-prior gain; a fixed uniform code should transfer more stably. This is falsifiable only with the same task semantics and matched setup/inference costs, and the theorem excludes model errors, framing, and non-independent X/Y priors.

### Frozen codebooks under prior shift

The frontier is attainable by an explicit fixed-width codebook. For a source prior sorted as `p(1) ≥ … ≥ p(q)` and `K = min(q, 2^b)` available symbols, assign the first `K−1` values distinct symbols and put all remaining values in symbol `K−1`. The decoder maps each symbol to its most probable training-prior value; ties use the lowest source index. This realizes retained mass `S(K)`. It is one optimal codebook, not the only one, and it is specialized to the training prior.

For a frozen codebook with encoder `c` and decoder representative `r`, let `p'` be an evaluation prior on the same support. Its exact transfer accuracy is

\[
T(c,r;p')=\sum_{x=1}^{q}p'(x)\,\mathbf 1[r(c(x))=x].
\]

For two independent coordinates and frozen codebooks, exact target-row success is the product `T_X T_Y`. This measures prior transfer without letting either encoder or decoder adapt. If evaluation-prior adaptation is allowed, that is a separate condition: the sender partition stays fixed but the receiver may change each class representative. Any experiment must state which condition it uses and charge prior discovery, codebook communication/storage, framing, and inference outside this payload-only probability.

`optimal_nonuniform_codebook(...)` in [`bit_frontier.py`](../experiments/private_match_v0_3/bit_frontier.py) returns the encoder symbols, frozen representatives, exact fixed-width serialization, and exact transfer score. Tests check its in-prior optimum and a preregisterable shift example: for training prior `(1/2, 1/4, 1/8, 1/8)` with one bit, accuracy is `3/4`; under evaluation prior `(1/8, 1/8, 1/2, 1/4)`, the frozen codebook scores `1/4`, while a newly optimized codebook scores `3/4`. This is an exact model-free illustration, not an empirical language result. The current task generator remains uniform, so no non-uniform shard or model comparison is implied.

### Exact robustness radius under total variation

For a frozen codebook, let `E` be the set of source values decoded exactly, and let the training prior assign `t = p(E)`. If the support contains at least one value outside `E`, then over every evaluation prior `p'` with total variation distance `TV(p,p') ≤ δ`,

\[
\inf_{p':\,\mathrm{TV}(p,p')\le\delta} p'(E)=\max(0,t-\delta).
\]

The event probability changes by at most total variation, giving the lower bound. It is tight: move up to `δ` mass from `E` to its nonempty complement; once the original `t` mass is exhausted, success reaches zero. If `E` is the full support, success remains one for every prior, so that perfect-code case is handled separately. For independent X and Y whose evaluation marginals may shift independently within radii `δ_X,δ_Y`, the frozen two-sender codebook has worst-case exact-row success equal to the product of the two one-coordinate bounds. This product statement does not cover dependent evaluation coordinates.

The method `FixedWidthCodebook.worst_case_success_probability(...)` computes this exact one-coordinate robustness radius with rational arithmetic. Tests enumerate all denominator-eight priors within a `1/8` total-variation ball around the four-value example, verify the tight bound, and check the perfect-code exception. This supplies a model-free, falsifiable calibration for prior-shift experiments; finite data, prior estimation, and codebook setup uncertainty require additional treatment.

## 16. Few-shot onboarding limit for a holistic protocol

An independently trained symbolic protocol can be semantically valid within its community and still be opaque to a newcomer: permuting the symbols and permuting their meanings together preserves task success. Cope & McBurney make this symmetry explicit when motivating cooperative language acquisition: conventions learned by separate self-play communities need not interoperate ([primary preprint](https://arxiv.org/abs/2402.16247)). The finite result below quantifies one deliberately structure-free onboarding baseline; it is not a model of natural-language priors or compositional learning.

**Model.** There are `M ≥ 1` meanings and `M` symbols. A community uses a fixed but newcomer-unknown bijection chosen uniformly from the `M!` possible meaning-to-symbol maps. Before a test, the newcomer receives `n` distinct, correct meaning-symbol examples (`0 ≤ n < M`). At test, the sender observes one uniformly sampled meaning and sends its symbol. The newcomer knows the examples, observes the symbol, and uses the Bayes-optimal decoder. Examples and decoding are noiseless; their communication, inference, and setup costs are not part of the accuracy formula.

**Proposition (holistic onboarding accuracy).** The newcomer's expected exact-decoding accuracy is

\[
P_{\mathrm{holistic}}(M,n)=\frac{n+1}{M}.
\]

**Proof.** Each of the `n` demonstrated meanings is decoded exactly. For any other test meaning, its symbol is also unseen because the protocol is bijective. Conditional on the observed examples and that symbol, each of the remaining `M-n` meanings is equally likely to own it; the Bayes-optimal success probability is `1/(M-n)`. Averaging over the uniform test meaning gives `n/M + ((M-n)/M)(1/(M-n)) = (n+1)/M`. In particular, `M-1` examples suffice because bijectivity identifies the final pair by elimination. ∎

**OOD candidate corollary.** Suppose instead that the receiver sees a candidate set `C` of size `k`, every meaning in `C` is absent from the calibration examples, and the target is uniform in `C`. Under the same random-bijection prior, the transmitted symbol is independent of the target conditional on the examples: each unseen candidate is equally likely to own that symbol. Thus the exact Bayes accuracy is `1/k`, the no-message accuracy. This is the relevant holistic lookup control for a strictly disjoint held-out support. The `(n+1)/M` formula must not be applied to a split where test targets are guaranteed to be outside the calibration set.

The result applies to a one-to-one holistic lookup convention. It does not lower-bound all few-shot protocol learning: compositional rules, pretrained semantic alignment, task structure, candidate-set side information, or interactive queries can reduce the number of examples needed. It also counts examples, not their byte/token or adaptation-compute cost. Use it as a control against which exemplar-driven onboarding can be compared, and separately price the entire calibration exchange and adaptation work.

The exact-rational calculator [`protocol_onboarding.py`](../research/protocol_onboarding.py) emits the full curve and the disjoint-support reference. Its tests exhaust all bijections on small meaning sets.

**Falsifiable prediction P12.** When meanings are paired with symbols by a uniformly random holistic bijection and only `n` distinct pairs are shown, a Bayes decoder's mean held-out exact accuracy over uniformly sampled meanings is exactly `(n+1)/M`. A discrepancy falsifies an assumption or the scorer. In a separate matched-cost experiment, a learned compositional protocol that exceeds this curve on unseen combinations would show productive structure beyond memorizing holistic pairs; failure to exceed it would provide no evidence of productive onboarding. Count calibration examples, prompt/context tokens, adaptation inference or training, and test-use cost separately.

**Falsifiable prediction P13.** With a uniformly random unknown one-to-one holistic code, if calibration covers only meanings outside the receiver's candidate set, the message does not change the receiver's posterior over that set. For any fixed `k`-candidate table whose rows are all unseen, Bayes success remains exactly `1/k`, regardless of how many other meaning-symbol pairs have been observed. This exact null is checked by [`disjoint_holdout_candidate_accuracy(...)`](../research/protocol_onboarding.py). An empirical gain on such a setup requires information outside the random holistic code assumption (for example, semantic/pretrained prior or compositional structure), not just seeing more unrelated lookup pairs.

## 17. Exact support and communication scaling for the modular OOD split

The v0.4 task family is defined for `d ≥ 2` categorical attributes with `V ≥ 2` values per attribute. Independently permute the rank order on every axis and hold out a complete tuple when the sum of its `d` ranks is zero modulo `V`. This creates an exact calibration fixture for productive higher-order composition; the construction alone does not establish that a model or language learned such composition.

**Proposition (split support counts).** The meaning universe contains `V^d` tuples, of which exactly `V^(d-1)` are held out and `V^(d-1)(V-1)` remain for training. Fix any assignment on `r` attributes, where `0 ≤ r < d`. Across the remaining axes, exactly `V^(d-r-1)` completions are held out and `(V-1)V^(d-r-1)` completions are in training. Thus every proper partial assignment has training support, while each complete held-out tuple is absent from training.

**Proof.** Once ranks for all but one free axis have been chosen, exactly one of the `V` values on the final axis makes the total rank sum zero modulo `V`. There are `V^(d-1)` choices for the other coordinates, proving the held-out count. With `r` coordinates fixed, there are `V^(d-r-1)` assignments to the other free coordinates before the final one; one of `V` final values is held out and the other `V-1` are training completions. Summing the `V^r` partial assignments across each choice of `r` axes gives `binom(d,r)V^r` distinct partial assignments. ∎

For a balanced receiver candidate set of size `k` whose order is target-independent, the no-message Bayes reference is exactly `1/k`. Separately, if the sender does not see the candidate table and the receiver must identify any target tuple with zero error, the sender's one-way message must distinguish all `V^d` tuples. The exact worst-case payload floor is therefore `ceil(log₂(V^d)) = ceil(d log₂ V)` bits, achieved by a shared rank code. The schema/value vocabulary, decoder setup, and model inference costs are outside that ideal payload floor and must be reported operationally; see Section 10 for the full communication model.

The exact report generator [`research/emergent_ood_scaling.py`](../research/emergent_ood_scaling.py) computes these quantities without simulation. Its exhaustive tests compare every partial assignment in small generated splits against the formula for multiple `d` and `V` values.

**Falsifiable scaling prediction P14.** At fixed `V`, the held-out meaning support grows as `V^(d-1)` while the ideal zero-error payload lower bound grows as `ceil(d log₂ V)`. At fixed `d`, increasing `V` changes these quantities to `V^(d-1)` and `ceil(d log₂ V)`, respectively. A generated split with different exact support counts, or a zero-error code below the stated payload floor under the same side-information boundary, falsifies the construction, implementation, or assumptions. These are task-size and oracle communication scaling laws, not an empirical law for LLM token use or inference cost.

## 18. Separate task-information value from coordination value

A message can improve a multi-agent outcome in at least two ways: it can change a receiver's posterior about a task-relevant hidden state, or it can help agents coordinate their actions through a shared signal or focal convention. A task-success gain alone does not identify which mechanism operated. An information codec should be judged on whether the correct private fact reaches the receiver; an action-coordination protocol may work by establishing common intent without transferring that fact.

**Decision model.** Let `W` be the task-relevant hidden state, `Y` the receiver's local information, and `M` the received message. For receiver loss `ℓ(a,W)`, its Bayes risk before and after communication is

\[
R(Y)=\mathbb{E}\left[\min_a\mathbb{E}[\ell(a,W)\mid Y]\right],\qquad
R(Y,M)=\mathbb{E}\left[\min_a\mathbb{E}[\ell(a,W)\mid Y,M]\right].
\]

The decision value of information is `V_info = R(Y) - R(Y,M)`, which is nonnegative for an ideal receiver allowed to ignore the message.

**Proposition (no new posterior, no Bayes decision value).** If `M` and `W` are conditionally independent given `Y`, then `R(Y,M)=R(Y)` and `V_info=0`.

**Proof.** Conditional independence gives `P(W | Y,M)=P(W | Y)` almost surely. Therefore every action has the same conditional expected loss with or without observing `M`; minimizing and averaging preserves equality. ∎

This proposition applies to an individual decision whose loss depends on its action and `W`. It does not say communication cannot help a team. When team utility `U(a_1,...,a_n,W)` contains coordination externalities, a shared message can alter correlation among agents' actions, reveal intended actions, or select a focal equilibrium even when it carries no new information about `W`. Such a gain is coordination value and needs a team-utility estimand plus an intent-signal control. It must not be reported as successful transfer of private task knowledge.

**Falsifiable prediction P15 (message-association control).** In a receiver task whose reward depends on recovering a sender-private task state, use the same sender outputs, delivered-message multiset, receiver inputs, schedule, and cost accounting, but randomly permute the association between sender states and delivered messages across compatible episodes. If improvement depends on decoding task-relevant content, this semantic derangement should remove that improvement; if it persists, investigate shared-message activation, a public/focal signal, leakage, or an unmatched prompt effect. The permutation must avoid placing a source meaning in the destination candidate set, or the remaining overlap must be measured and included in the null. On tasks with joint-action externalities, add a separate one-symbol intent/cheap-talk arm and score coordination and welfare. The two controls answer different questions and must not be collapsed into one baseline.
