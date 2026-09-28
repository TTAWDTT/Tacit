# Exact five-bit private-query frontier v0.4

This is a model-free extension of the exact finite random-access-code control. It asks how much a frozen task-oblivious code can benefit from a larger channel when the receiver's query distribution is uniform or puts weight 7 on one coordinate and weight 1 on each of four others.

## Method

The sender observes a uniform five-bit vector. The receiver privately chooses a coordinate and must recover that bit. A decoder for a fixed message alphabet induces a codebook of reconstructed five-bit vectors. For each source vector, the optimal sender chooses the nearest codeword under query-weighted Hamming distance. The script exhaustively enumerates every decoder codebook at budgets 0, 1, and 2 bits: 32, 496, and 35,960 candidates per cell. It computes exact fractions, all tied optimal codebooks, and transfer of each frozen optimal codebook family to the other query prior. There are no model calls.

The result is an oracle frontier: shared codebook discovery/storage/setup is treated as free. Query disclosure to the sender is excluded; a query-aware arm would change the information boundary and must account for reverse-channel cost. The task is a classical random access code, established prior work, not a Tacit invention.

## Reproduce

```powershell
python experiments/private_query_v0_4/extend_frontier.py --output experiments/private_query_v0_4/results.json
```

The implementation uses Python's standard library. It precomputes weighted distances before enumerating codebooks. The smaller n=2..4 exact frontier remains in [v0.2](../private_query_v0_2/README.md); the uniform one-bit scaling theorem remains in [v0.3](../private_query_v0_3/README.md).

## Interpretation

Compare each frozen-transfer range with the target-prior oracle, not just the training-prior optimum. If transfer loss grows as the query prior becomes more concentrated or channel budgets change, that supports prior sensitivity for this task family; it does not establish that a learned LLM protocol will find, communicate, or generalize the code. Any eventual LLM realization must be evaluated against these exact task-and-budget oracles and must charge codebook/setup and query-disclosure costs.

The generated exact results are: uniform-query success is 1/2, 11/16, and 31/40 at 0, 1, and 2 bits; the 7:1:1:1:1 prior reaches 1/2, 9/11, and 39/44. At one bit, uniform-optimal frozen protocols transfer to the skewed prior at 11/16, below its 9/11 oracle; skew-optimal frozen protocols transfer to uniform queries at 3/5, below its 11/16 oracle. At two bits, the transfer means are 31/40 (uniform to skew; tied optima range 67/88 to 73/88) and 3/4 (skew to uniform), versus target-prior oracles 39/44 and 31/40. This extends a known finite RAC control by one source bit and two channel budgets; it is not evidence of LLM language superiority.
