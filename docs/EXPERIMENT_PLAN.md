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

**Observed v0.5 result:** semantic exactness was 15/16 scaffold, 11/16 concise-NL, 10/16 compact-KV, 4/16 JSON, 13/16 binary-labeled, 12/16 AutoForm, and 1/16 no-communication. The fixed-format fidelity audit found compact-KV 17/17, concise-NL 21/23, binary 4/17, and valid JSON 0/24; AutoForm selected plain-English messages on all eight episodes. Upstream strict integer success was much lower because the model often submitted correct equations as strings. The actual messages support reliable compact-KV transmission, but the end-task scores do not show a gain over the plain scaffold; binary and JSON outcomes still cannot be interpreted as successful formats. One local guess scored in no-communication. See [`research/DUOSUM_PILOT_V0_5.md`](../research/DUOSUM_PILOT_V0_5.md).

**Next gate:** v0.6 tests a second local model on the same pinned cases and prompts. Its result is below. A matched-budget frontier and richer tasks come only after reliable, interpretable comparisons.

## v0.6 cross-model replication

The frozen v0.6 protocol pairs the same eight held-out tasks and seven prompt conditions with Qwen3-4B Q4_K_M on llama.cpp. It preserves the v0.5 task files, scaffold, condition wording, max rounds, output cap, scoring rules, and local-only execution. The prediction is conditional: if the v0.5 format-adherence failures mostly reflect model capability, exact JSON/binary fidelity should improve. Record strict and semantic task success separately and inspect message traces before interpreting outcomes. Since model size, quantization, runtime, and chat template change together, this is a replication across model setups, not a clean scale experiment. Do not claim a budget frontier, causal scaling effect, or broad superiority from eight episodes.

**Observed:** strict per-agent success was 7/16 scaffold, 14/16 concise-NL, 15/16 compact-KV, 16/16 JSON-labeled, 3/16 binary, 11/16 AutoForm, and 0/16 no-communication. Compact-KV was syntax-valid and sender-faithful in all 16 messages. Concise-NL sent bare decimal payloads in all 16 messages, faithfully conveying sender values but violating its sentence instruction. The JSON arm emitted invalid single-quoted Python-style mappings in all 16 messages yet achieved 16/16 task success, showing receiver tolerance rather than valid-JSON adherence. Binary syntax was valid in 15/16 messages but sender-value fidelity was only 8/16. Relative to v0.5, this second setup improved strict success for concise-NL, compact-KV, and JSON-labeled prompts, but binary declined. These are descriptive differences confounded by model size, quantization, backend, and chat template. See [the v0.6 report](../research/DUOSUM_PILOT_V0_6.md).

**Next gate:** evaluate a typed parser/decoder as an explicit system condition against prompt-only formatting, then replicate across a richer task family and additional sender/receiver pairs. Account for parser cost, malformed-message recovery, and wire size. Do not call the invalid JSON arm JSON adherence or compare backend token totals directly across v0.5/v0.6.

## v0.7 PrefixSum transfer

The v0.6 result is compatible with useful compact scalar messages, but DuoSum only requires one agent to add one hidden integer. The frozen v0.7 suite adapts Silo-Bench II-11 Prefix Sum to two agents, with 6-, 15-, and 30-value segments and four seeded cases per length. It uses the exact v0.6 Qwen3-4B Q4_K_M/llama.cpp setup and seven arms: free-form shared scaffold, prescribed concise English subtotal, compact `s=<sum>`, JSON subtotal, binary subtotal, full-shard JSON list, and no communication.

For a segment of length (L) with values in ({1,ldots,50}), Agent 1's target depends on Agent 0 only through (S=sum_i x_i). There are (49L+1) possible subtotal values, so an exact fixed-length encoding needs at least (lceillog_2(49L+1)ceil) bits in the worst case (9, 10, and 11 bits for the tested lengths). This lower bound predicts that subtotal messages can be much smaller than the full shard, but it does not predict whether model agents can compute, encode, decode, or use them reliably.

The falsifiable prediction is conditional: if compact-KV/binary messages accurately carry the subtotal, they should beat full-shard messages on payload bytes; the full-shard arm may still yield higher task success if arithmetic aggregation or decoding is a bottleneck. No equal-token budget, efficiency-frontier claim, or scaling law is part of v0.7. The task generator validates exact local and distributed prefix outputs plus Agent 1's information insufficiency. See [`experiments/pilot_v0_7/policies.json`](../experiments/pilot_v0_7/policies.json).

**v0.7 audit:** the run was stopped after 59 cells when upstream parsing was found to coerce numeric-only messages to integers and JSON message strings to Python objects; an oversized converted integer also crashed logging. Agents also reversed sender/receiver roles. Those cells are not valid protocol-comparison data; see [the audit](../research/PREFIXSUM_PILOT_V0_7_AUDIT.md).

**v0.8 correction:** use fresh seeds, role-specific private prompts, a lossless `send_message.content` adapter, and separate direction/order metrics. The seven message conditions and the sufficient-statistic prediction remain; each new case is held out from the revised frozen protocol. Freeze the adapter and all conditions before inference. Still no equal-token budget or scaling claim.

**v0.8 outcome:** all 84 cells completed with the pinned local Qwen3-4B setup. The lossless adapter delivered messages and corrected role prompts produced the intended direction, but strict task success was weak and no episode was fully correct. Subtotal syntax did not translate into correct sums (0/12 faithful messages in each of four subtotal encodings), while full-shard messages were syntactically valid and faithful (12/12) yet still did not produce a correct joint episode. The raw audit identifies two bottlenecks: sender aggregation for sufficient-statistic messages, and receiver offset application / final array execution even when all source values arrive. The public [v0.8 report](../research/PREFIXSUM_PILOT_V0_8.md) separates send attempts, delivery, receipt, syntax, value fidelity, and exact task success.

**Oracle control:** deterministic sender and receiver achieved exact joint success in 12/12 cases using the same pinned engine, raw-message adapter, and scorer, with one faithful subtotal delivery per case. This validates the task/scoring path without measuring model ability. The next diagnostic is a paired hybrid: deterministic sender + Qwen receiver to measure offset decoding/application, and Qwen sender + deterministic receiver to measure aggregation/message formation. Use the same seed set for paired diagnosis and report it separately from the frozen format comparison.

**v0.9 hybrid diagnostic outcome:** with oracle Agent 0, Qwen Agent 1 received the correct subtotal in 11/12 episodes but was exact in 0/11 received cases; 4/12 outputs were exactly local-only prefixes, 7/12 were other incorrect arrays, and 1/12 was absent. With Qwen Agent 0, message syntax and delivery were 12/12 but subtotal fidelity was 0/12 and Agent 0 exactness was 3/12. An oracle Agent 1 exactly applied the wire subtotal in 12/12, showing the sender's wrong value—not the receiver control—prevented joint success. This identifies separate sender arithmetic and receiver execution problems for this model/task, not a language-format effect. See [the report](../research/PREFIXSUM_HYBRID_DIAGNOSTIC_V0_9.md).

**v0.10 preregistration:** compare local Qwen3-8B Q4_K_M with the paired Qwen3-4B v0.9 baseline in the same two hybrid roles and same task seeds. This checks whether scale moves either sender or receiver capability enough to create a viable protocol-comparison operating region. Same seeds mean the scale pilot is a within-suite diagnostic, not new held-out evidence. The model revision, GGUF hash, hypotheses, and metrics are frozen at [`preregistration.json`](../experiments/pilot_v0_10/preregistration.json) before 8B inference.
