# Initial research thesis

**Status:** v0.2, updated 2026-09-29 after the OPTiMACS full-paper audit. This is a falsifiable starting point, not a conclusion.

## Thesis

There is no reason to expect one universally optimal language for all LLM-to-LLM communication. Two related questions must be kept separate: which interaction policy chooses who sends what and when, and which reusable representation carries the chosen content. OPTiMACS already learns task-conditioned message-format selection from multi-agent outcomes; that alone is not an open novelty claim for Tacit. The remaining test is whether a stable, compositional protocol can improve the task-success versus *total* communication-and-computation frontier and transfer across held-out tasks or receivers, especially under deployment conditions where a learned selector's search/training or per-message classification cost matters.

The first research goal is therefore not to invent notation or duplicate AutoForm/OPTiMACS. It is to compare strong fixed-format, prompted-format, learned-format, and latent alternatives on communication-dependent, budget-controlled tasks; charge their complete cost; identify their transfer limits; and determine whether any reusable compositional protocol repairs a measured gap.

## Why this is a plausible research gap

- Non-natural-language formats are already an established baseline: Chen et al. (Findings EMNLP 2024) report model-selected formats, transfer to different LLMs, and up to 72.7% lower token use in their multi-agent communication experiments. Any new proposal must reproduce or outperform the relevant baseline under matched conditions.
- Learned adaptive format selection is also established: Gupta et al. (Findings ACL 2026) introduce OPTiMACS, which learns task-conditioned representations from multi-agent reward trajectories and reports quality gains with mixed token changes. The paper does not target OOD transfer and its headline token tables do not expose a complete classifier/search/setup ledger. A reusable protocol must be tested against this policy baseline where feasible, not presented as the first adaptive formatter. See the [OPTiMACS audit](../research/OPTIMACS_AUDIT_V0_1.md).
- AutoForm already evaluates a HotpotQA split-context task where evidence is divided between agents. Its results vary by model: the reported GPT-4 pair gains quality while using fewer tokens, whereas the GPT-3.5 pair loses quality. Thus both conditional success and failure are documented; the precise unresolved issue is whether these trade-offs persist on explicit budget curves, interactive tasks, and current open/heterogeneous receivers after setup and decoding costs are charged.
- Latent collaboration papers report efficiency/quality gains, but latent representations introduce compatibility, transmission-size, decoding, runtime, and auditability questions. Their claims should be evaluated in the native execution setting and against text baselines with equivalent task opportunity.
- TFlow (Bao et al., 2026) extends this challenge to receiver-specific weight-space communication: it reports large processed-token and latency reductions against text-agent baselines on a shared Qwen3-4B setup, while showing a larger quality gap on HumanEval+ and higher latency than a standalone receiver on four of five tasks. Token counts do not price its tensor channel, parameter-generator setup, or receiver-specific runtime. Include it conditionally where compatible weights and compute are available; otherwise narrow any superiority claim to the explicit heterogeneous/API-only or portable-wire setting. See the [TFlow audit](../research/TFLOW_AUDIT_V0_1.md).
- Existing communication-necessary suites now include Silo-Bench (algorithmic information silos at varying agent scales) and MT-PingEval (private-information games at fixed total budget and varying interaction turns). Prefer these before creating a redundant task set.
- Production agent protocols such as A2A chiefly standardize interoperability, task lifecycle, and artifacts. These are important system layers but do not settle which semantic encoding is most efficient.
- LLM-designed protocols are an existing research direction: LMAC uses an LLM offline to design and refine executable communication code for trained MARL agents. The gap studied here is direct protocol use between LLM endpoints, with end-to-end semantic fidelity, inference cost, transfer, and strong matched-format/policy baselines; see the full-text audit in [`research/RELATED_WORK.md`](../research/RELATED_WORK.md).
- Emergent-language results caution that compositionality and generalization are distinct outcomes. We must measure both directly rather than infer one from the other.

## Falsifiable hypotheses

**H1 — Conditional protocol value.** A reusable representation can improve the total quality-cost frontier over optimized NL, AutoForm, and a task-conditioned learned-format selector on at least one preregistered communication-dependent task family under equal-budget evaluation. Any omitted baseline must be tied to an explicit access, training, or deployment constraint; token-only savings do not satisfy H1.

**H2 — Transfer penalty.** Task-specific shorthand or learned codes may look efficient in-distribution but lose much of their advantage under unseen task combinations, new receivers, or corrupted messages. A reusable compositional representation should degrade less than a per-task format policy when the reuse horizon and held-out task distribution are controlled.

**H3 — Budget interaction.** If a representation's benefit comes from higher effective information density, its relative success advantage should increase as the communication budget tightens, until decoding errors dominate. This predicts a non-monotonic trade-off, not unconditional gains.

**H4 — Setup-cost crossover.** Learning or negotiating a codebook can pay off only after enough repeated exchanges. Its break-even horizon should be predictable from setup cost and per-message savings; for short tasks, familiar formats may win.

**H5 — Receiver dependence.** A compact code optimized for one model or model family can lose to natural/structured text for heterogeneous receivers when decoder mismatch cost is included.

## Measurement commitments

Report task success and calibration/fidelity alongside output and input tokens, serialized bytes, number of turns, wall-clock latency, setup/training cost, local inference time or energy where measurable, parse/decode failures, recovery cost, and cross-model transfer. Plot a Pareto frontier rather than collapsing these dimensions into an arbitrary scalar. Compare both equal-budget and equal-quality operating points.

For attribution, record interaction schedule/content selection, format-selection policy, representation/encoder, and receiver decoder as separate experimental factors. A reusable-language claim needs a matched-policy comparison or a scripted semantic-payload ablation; a jointly optimized selector-plus-format result must be reported as a system result, with both training/search and runtime selection costs included.

## Scope exclusions

This thesis does not claim language is inherently superior to latent transfer, tool calls, shared memory, or a transport protocol. These are competing communication mechanisms or different system layers and belong in comparisons where their assumptions can be made explicit. The project will not claim a universal optimum from a finite model/task sample.

## Decision gates

1. **Relevance gate:** reuse Silo-Bench and/or MT-PingEval first; each tested task must require information held by different agents and show a measurable drop when communication is removed.
2. **Baseline gate:** no new format is evaluated against English alone; include optimized concise NL, JSON/schema, code or symbolic forms where appropriate, AutoForm, OPTiMACS-style learned selection, and current learned/latent methods when feasible.
3. **Cost gate:** count format instructions, task classification, candidate discovery, training/search, codebook setup, decoding, and retries. Token savings that shift equal cost elsewhere do not establish efficiency.
4. **Replication gate:** a result must repeat across seeds and at least two receiver conditions before being described as robust.
5. **Existence gate:** if matched existing methods dominate, publish that finding and redirect toward measurement/runtime infrastructure rather than inventing a language.
