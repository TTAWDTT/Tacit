# Noisy-channel private-query frontier v0.5

## Research question

How does an exactly optimized task-oblivious random-access code degrade when its *binary transport channel* flips bits, and how does that interact with receiver query priors and message budget? Existing project frontiers assume an intact message. Classical coding theory already treats noisy channels broadly; this finite calculation is a local exact control, not a new channel-coding theorem.

The preregistration freezes a uniform four-bit source; uniform and 7:1:1:1 private-query priors; message lengths 0, 1, and 2 bits; and binary symmetric channel flip probabilities 0, 1/8, 1/4, and 1/2. The reference task is the established classical RAC family ([Ambainis et al.](https://arxiv.org/abs/0810.2937)); classical channel coding originates with Shannon's noisy-channel analysis ([Shannon 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)). A directly related paper studies noisy *quantum* RACs, a different resource model ([Marques & da Silva 2022](https://arxiv.org/abs/2204.09485)).

## Exact finite objective

For a decoder map (c:{0,1}^b\to{0,1}^4), channel transition matrix (W_\epsilon(r\mid m)), source (x), and query weights (q_i), the best deterministic sender chooses

\[
m^*(x)=\arg\min_m\sum_r W_\epsilon(r\mid m)\sum_iq_i\mathbf{1}[x_i\ne c(r)_i].
\]

The reported accuracy is one minus this minimum expected weighted distortion, averaged exactly over all 16 source vectors and normalized by \(16\sum_iq_i\). The implementation enumerates all decoder maps, including duplicate reconstruction outputs, so noisy-channel redundancy and task-specific graceful degradation are not artificially excluded. Ties select the lowest sender message. Every rational transition probability and final score remains an integer fraction.

For frozen-prior transfer, the full decoder map and each source's training-prior-optimal transmitted message are frozen, then evaluated under the other query prior. All tied optimal decoder maps are retained for min/mean/max transfer reporting. No model, tokenizer, or external dataset is involved.

## Frozen predictions and controls

1. At noise 0, exact accuracies must reproduce v0.2's noiseless n=4 frontier.
2. At noise 1/2, the received word is independent of the transmitted word; under the uniform source, every budget/prior must yield 1/2 accuracy.
3. At each fixed noise rate, the optimum must be nondecreasing with message budget, since a larger channel can ignore extra bits.

These are consistency checks, not novelty claims. The exploratory question is whether the *size and tie sensitivity* of cross-prior transfer loss changes with physical noise and redundancy. This does not tell us how an LLM decodes an intact string.

## Exact results

| Bit-flip probability | Uniform prior, 1 bit | Uniform prior, 2 bits | Skew prior, 1 bit | Skew prior, 2 bits |
|---:|---:|---:|---:|---:|
| 0 | 11/16 | 13/16 | 17/20 | 37/40 |
| 1/8 | 41/64 | 47/64 | 61/80 | 131/160 |
| 1/4 | 19/32 | 21/32 | 27/40 | 57/80 |
| 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |

At one bit, the mean score of protocols optimized under one query prior and frozen under the other is:

| Bit-flip probability | Uniform-trained → skew queries | Skew-trained → uniform queries |
|---:|---:|---:|
| 0 | 11/16 | 5/8 |
| 1/8 | 41/64 | 19/32 |
| 1/4 | 19/32 | 9/16 |
| 1/2 | 1/2 | 1/2 |

For two bits, both directions' mean frozen-transfer scores are 13/16, 47/64, 21/32, and 1/2 at the four noise levels. The extra bit's absolute accuracy gain shrinks with noise: for the uniform prior it falls from 3/16 without noise to 3/32 at 1/8 and 1/16 at 1/4; for the skew prior it falls from 3/40 to 9/160 and 3/80. The exact [results JSON](results.json) also reports ranges across all tied optimal indexed decoder maps. These finite patterns are descriptive; no scaling claim is made from n=4.

## Reproduce

```powershell
python experiments/private_query_v0_5/noisy_frontier.py --output experiments/private_query_v0_5/results.json
```

The study is model-free. Its scope is bit-flip robustness only; the separate [channel-robustness audit](../../research/CHANNEL_ROBUSTNESS_AUDIT_V0_1.md) explains why semantic confusion, transport corruption, and task execution must be tested separately in an LLM system.
