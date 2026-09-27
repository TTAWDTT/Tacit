# Working theory: task-conditioned communication rate

**Status:** v0.1, falsifiable framing. These definitions organize experiments; they do not yet prove that a particular representation is better.

## 1. Episode and protocol

An episode is a cooperative task (T) with target (Y^*), private observation (X_i) at agent (i), public/shared context (C), and receiver implementation (r) (model weights, tokenizer, prompt/runtime, and available tools). A protocol π specifies the encoder, message syntax, any shared dictionary or setup exchange, decoder/receiver policy, and interaction schedule. The received messages (M_{1:k}) induce a final action Ŷ and task utility (U_T(Ŷ,Y^*)in[0,1]).

Define task distortion as

\[
d_T = 1 - U_T(\hat Y,Y^*).
\]

Exact-answer tasks may use 0/1 utility. Tasks with graded outputs need a task-specific proper score or rubric fixed before the run. ROUGE is a textual similarity measure and is not a substitute for task utility when exact outcomes are available.

## 2. Ideal semantic rate-distortion

For a fixed task distribution, receiver (r), and shared side information (C), an ideal one-message semantic rate-distortion function is

\[
R^*_{T,r}(D\mid C) = \inf_{q(m\mid x,c):\;\mathbb{E}[d_T]\le D} I(X;M\mid C,r).
\]

Here (X) is the sender's private observation and (M) is the message representation available to the receiver. This is a **theoretical reference**, not the token cost of an LLM. It captures the central question: how much information about private state must cross the boundary to keep downstream task distortion below (D), given what the receiver already knows? Shannon's fidelity criterion motivates minimizing rate under task-relevant distortion; interactive communication complexity motivates extending this to sequences of messages rather than compressing each turn independently ([Shannon 1948](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x), [Shannon 1959](https://mast.queensu.ca/~math474/shannon59.pdf), [Braverman et al. 2016](https://epubs.siam.org/doi/10.1137/100811969)).

Three distinctions matter:

1. (I(X;M\mid C,r)) is not UTF-8 bytes, model tokens, FLOPs, latency, or energy. We report these operational measures separately.
2. Semantic distortion is receiver- and task-dependent. A message can omit irrelevant prose with zero distortion, but omit a rare constraint and cause a large task loss.
3. In multi-turn collaboration the next message can depend on previous messages and actions. The relevant object is then an interactive protocol; per-message compression may raise rounds, repeated prompt cost, or repair traffic.

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

- **P1, task distortion:** at equal message budget, compact formats help only when omitted information is conditionally irrelevant to the receiver's optimal task action; errors should cluster around task-critical omissions, not message length itself.
- **P2, receiver alignment:** for a fixed sender representation, downstream distortion depends on the receiver (r). A code's token/byte advantage should be separated from its decoder mismatch cost.
- **P3, interactive compression:** under a strict total budget, a second turn helps only if its expected value of information exceeds repeated context, prompt, and control overhead. Compare several turn allocations at a fixed total budget.
- **P4, setup horizon:** observed codebook savings should cross the baseline at the measured (H^*) within uncertainty if per-episode costs are stable. Failure to do so suggests non-stationary quality, hidden cost, or an incomplete accounting boundary.
- **P5, error recovery:** adding redundancy or clarification can lower first-pass message efficiency but improve expected task utility under corruption/receiver mismatch. Measure bytes/tokens including retries and repair turns.

## 6. Conditions required for a stronger theorem

A useful optimality or lower-bound claim needs an explicit task distribution, side-information model, receiver class, distortion function, channel (including who pays for the decoder), and interaction limit. It must also state whether encoder/decoder knowledge is shared for free or has setup cost. Without these assumptions, "most efficient language" is underspecified. The project will not claim a universal optimum from finite benchmark evidence.
