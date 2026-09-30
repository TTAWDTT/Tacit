# Exact lossy frontier for simultaneous private-input sum (v0.1)

**Status:** exhaustive finite communication oracle; no model result and no proposed LLM language.

## Why add an interior frontier?

The existing sum analysis provides the exact no-message Bayes point and the zero-error payload endpoint. Comparing model messages only with those endpoints leaves the intermediate bandwidth regime uncalibrated. We enumerate every deterministic fixed-width simultaneous encoder for small instances, so an LLM protocol can be compared with the best possible symbolic code at the same ideal payload budget.

This is a finite distributional problem that differs from the asymptotic results on one-round Sum-Distinguish, which ask a coordinator to distinguish two promised sum cases or test one target sum. Those are useful related models, not this exact full-valued-sum prior. See [Apon, Katz & Malozemoff (2013)](https://arxiv.org/abs/1301.4269) and the general simultaneous-message model in [Babai, Gál & Kimmel (2003)](https://epubs.siam.org/doi/10.1137/S0097539700375944).

## Model

There are `m` simultaneous senders, each independently holding `X_i` uniform on `{0,1,2,3}`. Sender `i` applies a fixed deterministic map `f_i:{0,1,2,3}→{0,1}^{b_i}` once, without seeing other inputs. The referee observes the tuple of messages and outputs a single exact integer sum. The budget is `B=Σ_i b_i`; each message has fixed width, sender identity and message boundaries are known, and the shared codebook/setup is fixed. We charge payload bits only. For each `m≤4`, the enumerator covers all feasible total budgets from 0 through `2m`.

Variable-length coding, error-correcting transport, interactive protocols, model computation, prompt/setup costs, and per-token costs are outside this finite oracle. A real serialized envelope may cost far more than the ideal payload floor.

## Exact enumeration

Every encoder is equivalent, up to renaming message labels, to a set partition of the four private values into nonempty preimage cells. A partition with `k` cells needs `ceil(log₂ k)` fixed-width bits. There are 15 set partitions in total. Exchanging two sender roles preserves the i.i.d. source distribution and sum objective, so the search can enumerate multisets of sender partitions rather than ordered assignments.

For a received message tuple `u=(u_1,…,u_m)`, let `A_{i,u_i}` be sender `i`'s preimage cell. The number of input vectors consistent with this transcript and having sum `s` is

`c_s(u) = [z^s] ∏_i (Σ_{x∈A_{i,u_i}} z^x)`.

The best exact-answer decoder returns a modal `s`, scoring `max_s c_s(u)` vectors for that transcript. Thus the protocol's exact success probability is

`4^(−m) Σ_u max_s c_s(u)`.

Summing these modal counts gives an integer exact numerator. The decoder table in the emitted JSON uses the smallest sum on ties.

**Randomization does not improve this finite optimum under the same fixed-width constraints.** Fixing all sender and decoder random tapes turns any randomized protocol into one of the enumerated deterministic protocols. Its expected success is a mixture of their success probabilities, which cannot exceed the maximum deterministic value. This does not apply to protocols whose length bound is only an expectation or to models given extra shared input-dependent state.

## Exact frontier

The values below are the optimal number of correct vectors out of `4^m`; budgets are *at most* the displayed total payload. A plateau means another bit does not improve optimal exact-sum success for that finite instance.

| Senders | Total bits `B` | Best correct vectors | Success |
|---:|---:|---:|---:|
| 2 | 0–1 | 4/16 | 0.2500 |
| 2 | 2–3 | 8/16 | 0.5000 |
| 2 | 4 | 16/16 | 1.0000 |
| 3 | 0 | 12/64 | 0.1875 |
| 3 | 1 | 14/64 | 0.21875 |
| 3 | 2 | 16/64 | 0.2500 |
| 3 | 3 | 24/64 | 0.3750 |
| 3 | 4–5 | 32/64 | 0.5000 |
| 3 | 6 | 64/64 | 1.0000 |
| 4 | 0 | 44/256 | 0.171875 |
| 4 | 1 | 48/256 | 0.1875 |
| 4 | 2 | 56/256 | 0.21875 |
| 4 | 3 | 64/256 | 0.2500 |
| 4 | 4–5 | 96/256 | 0.3750 |
| 4 | 6–7 | 128/256 | 0.5000 |
| 4 | 8 | 256/256 | 1.0000 |

The optimal partitions clarify how the small-budget allocation changes with `m`: for `m=3, B=2`, all bits can be spent fully describing one sender (`[4,1,1]` cells across sender encoders); at `B=3`, one bit from each sender is better (`[2,2,2]`). For `m=4`, the optimal `B=4` point uses one bit per sender. These are task-specific exact codes, not LLM language proposals.

## Implications and falsifiable prediction

This oracle gives the strongest deterministic fixed-width symbolic baseline for `m≤4`, rather than merely a convenient hand-authored code. Any learned or natural-language protocol constrained to the same sender-specific payload alphabet and fixed total bit budget cannot beat its Bayes success under the stated task prior. A measured score above the table means some condition differs: byte/bits accounting, shared setup, side information, task distribution, output metric, or scorer.

**P24.** Reproduction of the enumerator must yield the exact integer frontiers above; every emitted codebook must attain its numerator when exhaustively evaluated on all `4^m` inputs. If a purported deterministic code within the same class exceeds a row, the enumeration or proof is wrong. Future experiments should compare LLM arms with this oracle only after mapping actual serialized messages to a genuinely fixed payload-bit budget; text envelope bytes and inference costs remain separate axes.

The executable [`multiparty_sum_lossy_frontier.py`](multiparty_sum_lossy_frontier.py) emits every budget point, one attaining sender partition profile, and a full MAP receiver table. The committed [machine-readable results](data/MULTIPARTY_SUM_LOSSY_FRONTIER_V0_1.json) bind rows to the calculator SHA-256. Focused tests independently execute emitted codebooks over all input tuples for the two-sender fixture, check exact no-message/zero-error endpoints for `m=1..4`, and reproduce the JSON artifact. Its exhaustive scope is intentionally small; it is not a scaling law and says nothing about performance at larger `m` without new computation or proof.

The follow-on [transport ledger](MULTIPARTY_SUM_TRANSPORT_FRONTIER_V0_1.md) measures these same codebooks through both SDK serializers and separately reports fixed empty-envelope slots versus skipping singleton senders. It shows why ideal payload bits cannot stand in for total application bytes; no model behavior or setup amortization is measured.
