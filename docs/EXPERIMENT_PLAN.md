# Initial experiment plan

**Status:** design sketch only. No result has been collected. Freeze concrete versions, prompts, splits, and analysis before any confirmatory run.

## First benchmark requirement

Construct tasks with private information distributed across agents, where a single agent cannot observe all required facts. Establish communication necessity by evaluating (a) full communication, (b) no communication, and (c) a centralized oracle/upper bound. Begin with deterministic synthetic tasks to control ground truth, then test transfer to a realistic task if the measurement is sound.

Possible seed task families to investigate—not yet selected:

- Distributed constraint satisfaction: agents hold disjoint constraints; only joint aggregation identifies a valid assignment.
- Multi-hop fact synthesis: agents receive non-overlapping evidence and must communicate provenance/conflicts to answer an unseen query.
- Collaborative debugging: agents hold disjoint traces/spec fragments; success requires exchanging the critical interaction.

Avoid tasks that can be solved from public context or common model knowledge alone.

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
- Equal maximum message budget sweep (for example several byte and model-token caps) and equal-quality operating points.
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
