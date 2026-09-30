# Simultaneous private-input sum: agent-count scaling reference (v0.1)

**Status:** model-free exact reference and falsifiable LLM scaling proposal. This is not an LLM result and does not propose a language.

## Research question

The project's Private Match work has exact communication frontiers for one or two private senders, but it does not isolate how the number of privately informed agents changes the ideal communication requirement. We use a standard number-in-hand simultaneous-message setup: each sender sees one private input and sends one message to a referee. This setting is established in multiparty communication complexity; see [Babai et al. (2003)](https://epubs.siam.org/doi/10.1137/S0097539700375944) for the simultaneous-message model and [Phillips, Verbin & Zhang (2016)](https://epubs.siam.org/doi/10.1137/15M1007525) for multiparty lower-bound methods. The small sum task here is a scoped calibration family, not a new communication-complexity model.

## Frozen mathematical question

There are `m` senders. Sender `i` privately holds `x_i ∈ {0,…,M−1}`. All senders transmit one simultaneous message to a referee, which must output `Σ_i x_i` exactly. A fixed deterministic protocol/codebook is shared; no input-dependent state, interaction, side channel, or free task-specific codebook is allowed. We charge all payload bits and use a noiseless channel. The tight closed form uses `M=2^w`.

Each sender's encoder must be injective. If two values at sender `i` mapped to the same message, hold every other input fixed: the referee would see the same transcript for two different correct sums. Therefore sender `i` needs at least `M` distinct codewords, or `w` fixed-width bits. Summing over senders gives a worst-case lower bound `m w`; sending each input verbatim attains it. For independent uniform inputs, prefix-free coding also costs at least `m w` expected bits by entropy, again attained by fixed-width messages.

The no-message receiver's optimal exact-answer probability is

`max_s [x^s](1 + x + … + x^(M−1))^m / M^m`.

This is the modal probability of the sum distribution. For fixed `M`, it is nonincreasing and tends to zero at order `m^−1/2` by the lattice local central limit theorem. These formulas are encoded in [`multiparty_sum_scaling.py`](multiparty_sum_scaling.py); all numerators and denominators remain exact integers.

## Exact fixture calculation

For `M=4` (`w=2`), the calculated no-message success and ideal exact payload are:

| Senders `m` | No-message exact-sum success | Tight zero-error payload | One-call star LLM calls* |
|---:|---:|---:|---:|
| 1 | 1/4 | 2 bits | 2 |
| 2 | 1/4 | 4 bits | 3 |
| 3 | 12/64 = 0.1875 | 6 bits | 4 |
| 4 | 44/256 = 0.171875 | 8 bits | 5 |
| 8 | 8,092/65,536 ≈ 0.12347 | 16 bits | 9 |
| 16 | 379,061,020/4,294,967,296 ≈ 0.08826 | 32 bits | 17 |

\*The call count is an architectural accounting placeholder: one sender invocation per sender plus one referee invocation. Calls can be batched or endpoints can share a model process; payload and LLM-call accounting must remain separate.

## Predictions and limitations

**P23:** at fixed power-of-two `M`, each added private sender raises the ideal zero-error payload floor by exactly `log₂ M` bits. A purported exact, input-independent simultaneous codec below `m log₂ M` received payload bits must reveal which assumption changed (side information, shared input-dependent state, interaction, errors, or an accounting boundary). No-message exact success follows the exact convolution formula above.

For a complementary deployment study, hold total private information fixed and repartition it over increasing `m`. The ideal semantic payload then stays constant, while a star-shaped LLM architecture may add prompt, call, synchronization, and receiver-context cost per additional agent. That is an empirical systems prediction, not part of the bit lower bound. Compare the same fixed representation, task semantics, model settings, and total inference/channel budget across agent counts; do not let the protocol change with `m` without charging adaptation/setup.

The theorem is for exact deterministic simultaneous communication. It does not bound lossy protocols, randomized protocols with allowed error, interactive exchanges, arbitrary functions, model compute, semantic fidelity, or tokens. An LLM sum task may also be dominated by arithmetic ability; an oracle-sender/receiver ablation or another task family is needed before interpreting a failure as a communication bottleneck.

## Reproduction

Run `python research/multiparty_sum_scaling.py --domain-size 4 --agents 1 2 3 4 8 16`. The standard-library calculator emits exact no-message numerators/denominators, lower and matching upper payload bounds, and optional one-call-per-role message/call counts. Focused tests verify linear tight payload growth, exact modal probabilities, and input validation. No model, tokenizer, dataset, endpoint, or GPU was used.
