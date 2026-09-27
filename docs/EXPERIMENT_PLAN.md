# Initial experiment plan

**Status:** design plus exploratory diagnostics from Silo-Bench v0.1 and interaction calibration v0.2. No confirmatory evidence has been collected. Freeze each concrete protocol before its first model run.

## First benchmark requirement

Start by reusing existing suites rather than creating another benchmark. Evaluate Silo-Bench's private data shards / algorithmic tasks for scaling and MT-PingEval's private-information games for multi-turn behavior, subject to artifact license and reproducibility checks. Establish communication necessity in each selected split with (a) full communication, (b) no communication, and (c) a centralized oracle/upper bound. Add a small deterministic synthetic task only if these suites do not permit clean message-format/budget interventions.

Potential tasks to retain only if the existing suites leave a concrete gap:

- Distributed constraint satisfaction: agents hold disjoint constraints; only joint aggregation identifies a valid assignment.
- Multi-hop fact synthesis: agents receive non-overlapping evidence and must communicate provenance/conflicts to answer an unseen query.
- Collaborative debugging: agents hold disjoint traces/spec fragments; success requires exchanging the critical interaction.

Avoid tasks that can be solved from public context or common model knowledge alone.

AutoForm's existing HotpotQA split-context setting already demonstrates distributed supporting facts and provides a direct replication target. Its paper reports Rouge-L/F1 and a `# Tokens` measure; this project should inspect and reproduce exact definitions rather than assume a missing cost measure. First extend the task/metric coverage; do not claim communication necessity is previously untested.

## Baseline families

1. Strong natural-language handoff with concise, task-specific prompting; include a model-optimized concise NL version.
2. JSON/schema with explicit required fields and validation.
3. Code or compact symbolic representation where the task is naturally executable/formal.
4. LLM-selected/ad hoc format from Chen et al. (2024), replicated as closely as feasible.
5. Shared dictionary / task-specific codebook with setup cost charged.
6. Learned discrete protocol if a reproducible training setup is justified.
7. Latent/KV transfer (LatentMAS, Interlat, CondenseFlow) where open models and compatible architectures permit fair runs.
8. No-communication, full-information, and random-message controls.

## Core controlled comparisons

- Same underlying model weights and task episodes while varying only representation; then vary receiver model and model family.
- Equal *total* communication budget sweep (input/output model tokens and serialized bytes, including instruction/codebook setup amortized over episode) and equal-quality operating points. Report native model-token counts as well as bytes because tokenizers differ.
- Fix and report turns, wall time, tool calls, model calls, temperature, prompt, and maximum rounds. On MT-PingEval, compare one-shot against multiple turns at the same overall budget.
- Hold task context, agent count, information partition, number of turns, decoding settings, and answer evaluator constant.
- Separate content messages from instructions, schema definitions, codebook negotiation, shared prompt context, and hidden state transfer.
- Measure decode success, semantic reconstruction, task completion, correctness, latency, generated and consumed tokens, bytes on the wire, setup cost, and retries.
- Sweep task complexity, context length, agent count, rounds, and information partition to test scaling, not just one fixed benchmark.

## Initial predictions

- If gains are only output brevity, equal-byte/equal-token quality curves will erase them.
- If gains depend on shared prior/task-specific conventions, unseen receivers/tasks will reduce them; charging setup will create a horizon-dependent crossover.
- If the task bottleneck is an exact symbolic constraint, typed structured forms may dominate prose; if the evidence is uncertain/contradictory, confidence/provenance and recovery may matter more than raw density.
- Under tighter budgets, compression should first improve success, then hurt as distortion and decoding errors dominate.

## Analysis and reporting

Use repeated seeds/episodes, paired comparisons on the same examples, bootstrap uncertainty intervals, and disclose all exclusions. Report the full Pareto frontier and bootstrap uncertainty for key frontiers; avoid selecting a protocol from a single aggregate score. Track failures by ambiguity, omission, hallucinated inference, schema/decode error, receiver mismatch, and error propagation.

## Local feasibility

The machine currently exposes an RTX 4060 with 8 GiB VRAM and about 32 GiB system RAM. Start with small open instruction models and deterministic symbolic environments. Do not assume full fine-tuning or large multi-agent models fit. API-backed models may be an optional later replication, not required for core reproducibility.
# Research update: exact-sum benchmark candidate

The first Silo-Bench run showed that task names and formats do not guarantee communication necessity. A next benchmark candidate is two-agent exact sum: each agent receives a private positive integer and both must submit the total. Neither private value alone determines the answer. Use varying input widths and multiple deterministic seeds, with a no-communication control on every task. This isolates the communication channel from long-chain reasoning, but is intentionally a low-complexity calibration task, not the final benchmark.

For two private integers from a domain of size \(M\), exact sum requires at least \(2\log_2M\) bits of worst-case deterministic binary communication when both agents must output the sum; for power-of-two domains, sending each input once attains the bound. This gives a known wire-bit floor. Model-token, payload-byte, repeated-context, decoder, and setup costs remain separate and are not directly bounded by that proposition. The task is a probe of protocol overhead and receiver fidelity, not a claim that sum aggregation is representative of all agent work.

Promotion gate: generate exact task JSON from a pinned seed; verify per-agent observation does not determine the sum; verify every model condition against the exact answer; measure how often communication is actually used and whether the no-communication control fails; then add more complex task families before any language claim.
