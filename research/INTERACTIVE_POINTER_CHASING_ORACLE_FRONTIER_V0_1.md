# Pointer chasing: an exact oracle bandwidth/round control

**Status:** preregistered analytic control; no models loaded and no inference run.

## Question and protocols

This control makes the bandwidth/interaction trade-off concrete under the project's bilateral-output rule: both agents must independently produce the same exact parity bit.

1. **Pointer relay:** alternate one fixed-width pointer per sequential message for `k` messages. With `w = ceil(log2(n))`, it uses `k w` payload bits and `k` message batches.
2. **Full-map exchange:** in one simultaneous batch, each agent sends its complete private function to the other as `n` fixed-width binary offsets. Both then know the full instance and compute the answer locally. It uses `2 n w` aggregate payload bits in two directed transmissions.

The concrete serialization is unframed ASCII `0`/`1` payload. Thus its payload-byte count equals the number of bit characters, but this is not a packed binary transport. The ledger intentionally excludes network framing, prompt/schema text, receiver-tokenizer units, model inference, latency, and one-time setup. The protocol implementation is in [`protocol_baselines.py`](../experiments/pointer_chasing_v0_1/protocol_baselines.py); the frozen control definition is [`oracle_frontier_preregistration.json`](../experiments/pointer_chasing_v0_1/oracle_frontier_preregistration.json), and the generated analytic grid is [`POINTER_CHASING_ORACLE_FRONTIER_V0_1.json`](data/POINTER_CHASING_ORACLE_FRONTIER_V0_1.json).

## Exact implications

For fixed `n`, the relay uses fewer aggregate payload bits than the full-map exchange when `k < 2n`; they tie at `k = 2n`. The full-map protocol always uses one synchronous batch, while the relay uses `k`. Therefore these two zero-error oracle points expose a simple trade-off: the relay is lower-bandwidth at shallow depths, and full-map exchange can reduce interaction depth by paying bandwidth. For `k >= 2`, the one-batch map exchange fits within a cap of `k−1` synchronous batches.

This analytic frontier has no sampling error: both algorithms solve every valid input by construction. It is not an empirical frontier for LLMs. In particular, the full-map control gives away all private information and bypasses model comprehension; ASCII byte counts are not native model tokens or network wire bytes.

## Relationship to communication-complexity theory

Mao, Yang, and Zhang define pointer chasing with alternating maps and parity output, and show a direct `k`-round relay at `O(k log n)` communication. Their Theorem 2 lower-bounds deterministic `(k−1)`-round protocols that speak Alice-first and achieve at least `2/3` accuracy on uniformly random function pairs by `Omega(n/k + k)` bits; Corollary 3 gives the randomized counterpart. Their round definition counts changes in speaker along a protocol path, and standard communication complexity specifies a protocol output at a leaf.

Our bilateral requirement asks both participants to return the answer, and the full-map control uses one **simultaneous two-message batch**. Those conventions differ from the cited theorem's sequential-speaker model. The comparison here is a transparent upper control, not a construction of the theorem's low-round protocol, nor evidence that an LLM benefits from interaction. See the [source paper, Definition 1, Theorem 2, and protocol definitions](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/html/LIPIcs.ITCS.2025.75/LIPIcs.ITCS.2025.75.html).

## Falsifiable next steps

- The analytic formula predicts a crossover at `k = 2n` in payload bits, independent of the fixed-width symbol size. A future serializer audit can falsify the byte accounting if the emitted payload length differs from the registered count.
- An LLM experiment would need a separate preregistration and a passed resource gate. At matched episode sets and channel-byte caps, measure bilateral exact success, inference tokens by each model tokenizer, latency, and error recovery. The oracle curves only establish what perfect protocol execution can achieve.
- Any comparison that adopts simultaneous batches must keep batch count, directed transmission count, and aggregate bytes as separate axes. It must not label the control as a sequential `(k−1)`-round protocol.

## Reproduction

```powershell
python experiments/pointer_chasing_v0_1/protocol_baselines.py `
  --sizes 2 4 8 16 32 --depths 1 2 4 8 16 32 `
  --output research/data/POINTER_CHASING_ORACLE_FRONTIER_V0_1.json
python -m unittest discover -s experiments/pointer_chasing_v0_1 -p "test_*.py" -v
```
