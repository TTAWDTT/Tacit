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

## v0.4 outcome and next gate

In four held-out episodes, no-communication semantic success was 0/8 agent outputs; communicating conditions ranged from 4/8 to 7/8. This supports continuing the task as a communication-necessity probe, not as a format ranking. Strict benchmark success was nearly zero because the model often submitted an equation rather than the required integer even after a common instruction. A trace audit found only 1/8 binary-arm messages had binary-digit syntax and 0/8 decoded to the sender's value; 0/8 JSON-arm messages parsed as JSON. The labeled arms mostly sent decimal strings and all JSON-arm messages used single quotes. These instruction-adherence failures mean their measured payloads/outcomes cannot be interpreted as binary/JSON efficacy. The next protocol must preserve raw messages/submissions and report syntax validity, sender-value fidelity, upstream strict scores, and a frozen, format-neutral semantic score. Freeze and validate the evaluators before new model calls. Then use more paired seeds and at least one additional local model; only proceed to matched communication budgets if task completion is reproducible in conditions that actually follow their representation instructions.

## v0.5 preregistered diagnostic

The v0.5 protocol uses the remaining two held-out replicates per width (eight episodes), retains the v0.4 controls, adds an AutoForm-style model-selected format, and adds explicit examples to JSON and binary instructions. The AutoForm wording is adapted from the released configuration in [AutoForm's official repository](https://github.com/thunlp/AutoForm). This arm has no fixed syntax target. All arms retain the same interaction and terminal-answer instruction, and the local semantic grader does not rewrite raw outputs. The policy pins prompts, cases, model, weight hashes, and engine before any new model call.

**Prediction:** if v0.4's JSON/binary failures mainly came from unclear instructions, examples should improve syntax validity and sender-value fidelity. That may not improve semantic task success or total token use. If those formats remain unused or misdecoded, this local model/setup cannot fairly estimate their representational efficiency. Eight tasks on one 1.7B model remain exploratory; budget sweeps wait for reliable execution and a second receiver model.
