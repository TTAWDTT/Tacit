# Private-query budget and distribution frontier v0.2

## Question

How does the best possible deterministic code change when (a) the number of private facts grows, (b) the sender receives more bits, and (c) the receiver's private query distribution is skewed?

The task remains the classical random access code: the sender observes a uniform (n)-bit vector (X); the receiver privately draws index (I) with declared distribution (q_i) and must return (X_I). The sender never sees (I). A (b)-bit one-way channel has at most (2^b) messages. Classical RACs are established theory, not a project novelty; this artifact is an exact task/control calibration for later LLM experiments.

## Exact reduction and enumeration

For any decoder, each possible message corresponds to an (n)-bit reconstruction vector whose (i)-th bit is the receiver's answer for query (i). Given a source vector, the sender's optimal message is the nearest such vector under weighted Hamming distance (d_q(x,c)=\sum_i q_i 1[x_i\ne c_i]\). Therefore, with a fixed (b)-bit channel, optimizing every deterministic encoder and decoder is equivalent to choosing a codebook of (2^b) reconstruction vectors minimizing expected weighted Hamming distortion.

The script exhaustively checks all codebooks for (n\in\{2,3,4\}), every integer budget (b=0,\ldots,n), and two query distributions: uniform and a 7:1:…:1 preference for coordinate 1. The maximum is \(\binom{16}{8}=12{,}870\) candidate codebooks in any cell. It also freezes the complete encoder/decoder protocol optimized under one prior and evaluates it under the other, reporting min/mean/max transfer success across all tied optimal codebooks. The encoder uses its training-prior nearest-codeword rule, with ties broken by lowest codeword; it is **not** re-optimized for the test prior. Source and query expectations are computed exactly as integer counts; reported fractions are exact.

## Important accounting boundary

This is an **oracle coding frontier**, not a free deployable language. It assumes the encoder/decoder share the optimal codebook without paying to discover, communicate, store, or select it. The task index remains private to the receiver. The query-conditioned sender oracle would need the index to cross the boundary; that reverse communication, call, and latency are excluded from these one-way curves and must be charged in a system comparison. No model, tokenizer, parser, or prompt is involved.

## Reproduce

From the repository root:

```powershell
python experiments/private_query_v0_2/enumerate_frontiers.py --output experiments/private_query_v0_2/results.json
```

The standard-library-only script deterministically regenerates [`results.json`](results.json). For uniform queries and a one-bit message, its n=2,3,4 results cross-check the independent truth-table enumeration in [v0.1](../private_query_v0_1/README.md): 3/4, 3/4, and 11/16 success.

## Interpretation

The frontier isolates a basic pressure that any LLM protocol study must preserve: task distribution and budget jointly determine which content a code should retain. The cross-prior rows directly quantify transfer loss (or tolerance) of the **frozen full protocol** on a different query mixture. This does not show that a learned code can be found by LLMs, understood by an independent receiver, transferred to new task mixtures, or beat natural language. Future tests must add codebook/setup costs and model execution while retaining no-message, query-conditioned, and full-information controls.

For the four-bit, one-message-bit example, the uniform-prior oracle reaches 11/16; all its optimal frozen protocols transfer to the 7:1:1:1 query prior at 23/40–29/40 (mean 11/16), versus a 17/20 oracle trained for that target prior. In the reverse direction, the skew-prior optimum reaches 17/20 but its frozen protocol scores 5/8 on uniform queries, versus 11/16 for the uniform-prior oracle. This is a task-local, one-bit mismatch result. The uniform-optimal set includes protocols with different transfer, so codebook choice/tie-breaking also matters; do not reduce it to “one code always wins.”
