# Interactive communication control: pointer chasing

**Status:** theory-grounded task implemented as a model-free generator/oracle in v0.1; not preregistered or evaluated with an LLM.

## Why this task may fill a real gap

`INDEX_m` isolates a one-way information bottleneck. Silo-Bench includes multi-agent sequential dependencies (including List Ranking), but its published task levels do not isolate a controlled two-party round-depth intervention. MT-PingEval measures multi-turn behavior, but its name-game gains are substantially reproduced by a random-guess-per-turn baseline, and its fixed per-turn caps complicate a clean round comparison. A task from communication complexity can provide a complementary control where required interaction depth is part of the mathematical problem definition.

This is a benchmark candidate, not a claim that pointer chasing is a realistic proxy for scientific collaboration. Its value is causal and diagnostic: determine whether agents can exploit a message protocol whose benefit and communication cost vary predictably with the number of interaction rounds.

## Task definition

For even `n >= 2` and depth `k >= 1`, the target distribution gives two agents independent private functions `f_A, f_B : [n] -> [n]`, uniform over all such functions. Let `p_0 = 1`; for each step `r = 1,...,k`, set `p_r = f_A(p_{r-1})` when `r` is odd and `p_r = f_B(p_{r-1})` when `r` is even. The required answer is the parity bit `p_k mod 2`. Both agents must independently submit the same correct bit after the communication episode; the primary score is joint exactness. Restricting `n` to even values balances the two parity classes in the output alphabet, but does not by itself guarantee a balanced answer prior at every depth. The public generator produces deterministic SHA-256 pseudorandom instantiations of this target distribution; the exact small-size prior audit enumerates the full distribution rather than the seeded shard.

Each agent sees only its own function; the shared prompt contains `n`, `k`, the recurrence, and the answer rule. Inputs, seeds, and answer keys are generated locally and deterministically from a recorded seed. The runner must strip `master_seed`, split, and episode ID metadata from both prompts; because generation is deterministic, exposing the seed metadata would let either agent reconstruct both maps. This definition matches the pointer-chasing function used by Mao, Yang, and Zhang (ITCS 2025), including its uniform input distribution and parity output.

## Established theoretical result and scope

Mao et al. prove that every `(k-1)`-round deterministic protocol in which Alice speaks first and which succeeds with probability at least `2/3` under the uniform function distribution has communication cost `Omega(n/k + k)` bits. Their Corollary 3 gives the corresponding lower bound for randomized `(k-1)`-round protocols with error at most `1/3`. A direct `k`-round protocol alternates the current pointer between owners, costing `O(k log n)` bits. These results make pointer chasing a principled source for hypotheses about an interaction-round/communication frontier.

The theorem concerns abstract bit protocols and asymptotic parameters. It is **not** a lower bound on LLM tokens, does not predict any particular model's accuracy, and does not establish a practical benefit for a new language. Small local-model settings may be below the regime where the asymptotic separation is visible. Empirical claims must report held-out success and measured serialized bytes; theoretical curves remain reference bounds with assumptions stated.

## Controls and staged eligibility

Before any format comparison, a future runner must separately establish:

1. **Scorer and generator control:** independently recompute every pointer chain and answer; validate private-input separation and deterministic regeneration.
2. **Full-information capability:** one model instance given both functions reaches a preregistered threshold at that `n,k`.
3. **Transport/oracle control:** deterministic agents using the same message API solve the `k`-round pointer protocol exactly and their payload-byte counts match the serializer.
4. **No-message controls:** report participant accuracy and a task-prior/random baseline over held-out seeds. Do not assume exactly 50% for every participant-conditioned view without deriving it for the chosen finite distribution. For `n=2,4`, the artifact exhaustively computes exact individual Bayes accuracies and thus the valid joint upper bound `min(Bayes_A, Bayes_B)`; its feasible joint strategies are not claimed to attain the globally optimal zero-communication protocol. See the [finite prior audit](INTERACTIVE_POINTER_CHASING_NO_MESSAGE_V0_1.md).
5. **Round-policy intervention:** compare `k`-round pointer relay with at most `k-1` exchanges while matching total channel-byte cap. Both agents submit a final answer after the exchange schedule; collect answers as sealed outputs so one final answer cannot become an uncounted message to the other. Score individual and joint exactness. Apply the same termination/evaluator rules and record unused turns, invalid messages, and retries.

Only if the local model passes controls 1-4 should a round-policy pilot be considered. Only after it passes should message representations be compared within each fixed schedule: optimized concise natural language, strict structured data, compact typed symbols, and any learned codebook. Charge schemas/codebooks and report inference tokens, channel bytes, latency, and setup separately. Cross-model transfer requires a held-out receiver and must not be inferred from same-model self-play.

## Falsifiable predictions

- Under a frozen model pair and fixed total byte cap, the `k`-round condition will outperform the `k-1`-round condition on some held-out `(n,k)` strata only when the task is within the model's full-information capability range. If performance is equal, the theoretical opportunity may be masked by model/task limits or the extra round may have no practical value at those scales.
- At fixed round policy, increasing `n` should increase the cost of a direct lookup relay approximately with `k log2(n)` bits, while a successful low-round protocol must pay a larger aggregate cost in the theorem's regime. This prediction is about algorithmic communication, not native tokenizer counts.
- Any compact-code advantage should survive a held-out seed and receiver test at matched bytes. A win confined to memorized map layouts, one model pairing, or an oracle-decoded message is not a general communication-language result.

## Decision

Keep this as the leading **round-complexity control**, alongside the existing one-way `INDEX_m` control and reused Silo-Bench tasks. Its model-free generator, bilateral scorer, fixed-width binary pointer codec, and oracle relay are implemented in [`experiments/pointer_chasing_v0_1/`](../experiments/pointer_chasing_v0_1/README.md). Do not launch a model capability screen until the repository's frozen resource preflight and experiment-specific capability criteria permit it. Offline generator/oracle checks are not an inference eligibility bypass.

## Sources

- Xinyu Mao, Guangxu Yang, and Jiapeng Zhang, [*Gadgetless Lifting Beats Round Elimination: Improved Lower Bounds for Pointer Chasing*](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/html/LIPIcs.ITCS.2025.75/LIPIcs.ITCS.2025.75.html), ITCS 2025, Theorem 2 and Definition 1. CC-BY 4.0.
- Yuzhe Zhang et al., [*Silo-Bench: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems*](https://aclanthology.org/2026.acl-long.1354/), ACL 2026, Appendix E.
- Jacob Eisenstein et al., [*MT-PingEval: Evaluating Multi-Turn Collaboration with Private Information Games*](https://arxiv.org/html/2602.24188), v2, Sections 3-4 and Appendix B.
