# Pointer chasing: parity-assisted low-round oracle control

**Status:** preregistered analytic extension to [v0.1](INTERACTIVE_POINTER_CHASING_ORACLE_FRONTIER_V0_1.md); model-free, no model loaded or inference run.

## Why extend the frontier?

The first analytic comparison used only the direct pointer relay and a one-batch exchange of both full maps. Re-reading the cited ITCS 2025 paper surfaced a stronger sequential low-round idea: share the parity of each function at every coordinate, then omit the final pointer transmission because the answer asks only for its parity. This gives a more relevant control near the paper's `(k−1)`-round theorem and costs less than exchanging full maps in many regimes.

The paper says Alice and Bob send the parity of their function values for all coordinates and then skip the final round. It does not specify a wire serialization or detailed message schedule in that paragraph. The protocol below is therefore recorded as our explicit implementation of that idea, not quoted as the paper's exact transcript.

## Explicit protocol

Let `w = ceil(log2(n))` and require `k >= 3`.

1. Alice's first sequential message sends her parity table `(f_A(x) mod 2)_{x∈[n]}` and pointer `p_1=f_A(1)`.
2. Bob's second message sends his parity table `(f_B(x) mod 2)_{x∈[n]}` and pointer `p_2=f_B(p_1)`.
3. The agents alternate pointer messages through `p_(k−1)`. Each sender's pointer is known to both after its message.
4. Both look up `p_(k−1)` in the already shared parity table for the owner of step `k`; this gives `p_k mod 2` without transmitting `p_k`.

The first two messages contain `n` parity bits plus one `w`-bit pointer each; each remaining message contains one `w`-bit pointer. Thus the exact total is:

`C_parity = 2n + (k−1)w` bits across `k−1` sequential speaker turns.

**Exactness proposition.** For every pair of valid maps and every `k≥3`, this protocol makes both agents output `p_k mod 2` after `k−1` turns. At turn `r`, the scheduled sender knows `p_(r−1)` (publicly at `r=1`, then by induction from the previous transmitted pointer), computes and sends `p_r`, and the receiver learns it; so both know `p_(k−1)` after turn `k−1`. The first two turns also deliver both parity tables to both agents. The owner of step `k` is therefore known to both, and each can look up `p_(k−1)` in that owner's table to obtain `f_owner(p_(k−1)) mod 2 = p_k mod 2`. No distributional assumption is needed for this zero-error correctness claim.

The cost counts unframed ASCII `0`/`1` payload bytes, so payload bytes equal bit characters. It excludes message framing, instructions/schema, tokenizer units, latency, inference, and setup. The implementation decodes/executes the transcript and asserts that both agents produce the generator's gold bit. The preregistration is [`oracle_frontier_v0_2_preregistration.json`](../experiments/pointer_chasing_v0_1/oracle_frontier_v0_2_preregistration.json), the implementation is [`protocol_baselines.py`](../experiments/pointer_chasing_v0_1/protocol_baselines.py), and the analytic grid is [`POINTER_CHASING_ORACLE_FRONTIER_V0_2.json`](data/POINTER_CHASING_ORACLE_FRONTIER_V0_2.json).

## Three exact oracle points

| Control | Success | Turns/batches | Aggregate payload bits |
|---|---:|---:|---:|
| Direct pointer relay | 1 | `k` sequential turns | `k w` |
| Parity-assisted skip-final-pointer (`k≥3`) | 1 | `k−1` sequential turns | `2n+(k−1)w` |
| Simultaneous full-map exchange | 1 | 1 simultaneous batch | `2nw` |

These points use distinct schedules; report sequential turns and simultaneous batches as separate axes. At the same `n,k`, the parity-assisted control uses one fewer sequential turn than the direct relay, at an extra `2n−w` bits (`C_parity − C_relay = 2n−w`). It is cheaper in payload than the full-map exchange exactly when:

`2n + (k−1)w < 2nw`, equivalently `k < 1 + 2n(1−1/w)`.

This is an oracle policy trade-off, not a representation result. A model that cannot reliably parse a parity table or pointer may fall far below these exact points.

## Connection to the theorem and its limits

Mao, Yang, and Zhang's Theorem 2 proves that any deterministic Alice-first `(k−1)`-round protocol for uniform function pairs with success at least `2/3` uses `Ω(n/k+k)` communication bits; Corollary 3 gives the randomized bounded-error form. Their paper also observes that sending both coordinate-wise function-parity tables allows the final round to be skipped and notes an `O(n)` deterministic `(k−1)`-round protocol, which is not tight for their cited Nisan–Wigderson upper bound when `k=o(log n)`. Our explicit transcript makes the parity idea's cost and bilateral output behavior auditable for `k≥3`; it does not improve or reprove the theorem.

The full-map condition is a separate simultaneous-batch protocol, not an Alice-first `(k−1)`-round protocol. The theorem is about abstract bit communication and a common protocol output; none of these bounds are model-token predictions. Source: Mao, Yang, and Zhang, [*Gadgetless Lifting Beats Round Elimination: Improved Lower Bounds for Pointer Chasing*](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/html/LIPIcs.ITCS.2025.75/LIPIcs.ITCS.2025.75.html), Definition 1, Theorem 2, and the final paragraph of §1.1.

## Falsifiable checks and next experiment

- The exact parity-assisted payload for every valid episode must equal `2n+(k−1)w`, and both local outputs must equal the scorer output. The regression suite exercises sizes 2, 4, 8, and 16 across depths 3, 4, 5, and 7.
- For any chosen parameter grid, the formula predicts a `2n−w`-bit parity-protocol premium over direct relay and the stated crossover against full-map exchange. Serializer length is the concrete falsifier; a mismatch requires revising this ledger.
- A future LLM study must first pass full-information and direct-oracle capability gates. It should then compare relay and parity-assisted schedules under the same frozen model pair and a matched channel cap, with model/tokenizer cost, turn latency, and failure recovery measured separately. Until then, the frontier is analytic only.

## Reproduction

```powershell
python experiments/pointer_chasing_v0_1/protocol_baselines.py `
  --sizes 2 4 8 16 32 --depths 1 2 3 4 8 16 32 `
  --output research/data/POINTER_CHASING_ORACLE_FRONTIER_V0_2.json
python -m unittest discover -s experiments/pointer_chasing_v0_1 -p "test_*.py" -v
```
