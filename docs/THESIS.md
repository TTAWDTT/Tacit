# Initial research thesis

**Status:** v0.1, 2026-09-27. This is a falsifiable starting point, not a conclusion.

## Thesis

There is no reason to expect one universally optimal language for all LLM-to-LLM communication. The useful object of study is a task-, receiver-, and budget-conditioned communication policy over representations and interaction acts. A candidate representation earns value only if it improves the task-success versus *total* communication-and-computation cost frontier, while preserving semantic fidelity and transfer under the intended deployment conditions.

The first research goal is therefore not to invent notation or duplicate AutoForm. It is to replicate and extend strong alternative-format results on communication-dependent, budget-controlled, multi-round tasks, identify where current options fail, and determine whether any reusable compositional protocol repairs that gap.

## Why this is a plausible research gap

- Non-natural-language formats are already an established baseline: Chen et al. (Findings EMNLP 2024) report model-selected formats, transfer to different LLMs, and up to 72.7% lower token use in their multi-agent communication experiments. Any new proposal must reproduce or outperform the relevant baseline under matched conditions.
- AutoForm already evaluates a HotpotQA split-context task where evidence is divided between agents. Its results vary by model: the reported GPT-4 pair gains quality while using fewer tokens, whereas the GPT-3.5 pair loses quality. Thus both conditional success and failure are documented; the precise unresolved issue is whether these trade-offs persist on explicit budget curves, interactive tasks, and current open/heterogeneous receivers after setup and decoding costs are charged.
- Latent collaboration papers report efficiency/quality gains, but latent representations introduce compatibility, transmission-size, decoding, runtime, and auditability questions. Their claims should be evaluated in the native execution setting and against text baselines with equivalent task opportunity.
- Existing communication-necessary suites now include Silo-Bench (algorithmic information silos at varying agent scales) and MT-PingEval (private-information games at fixed total budget and varying interaction turns). Prefer these before creating a redundant task set.
- Production agent protocols such as A2A chiefly standardize interoperability, task lifecycle, and artifacts. These are important system layers but do not settle which semantic encoding is most efficient.
- Emergent-language results caution that compositionality and generalization are distinct outcomes. We must measure both directly rather than infer one from the other.

## Falsifiable hypotheses

**H1 — Conditional representation gains.** A representation chosen for the task and receiver can improve the quality-cost frontier over fixed NL and AutoForm baselines on at least one preregistered family of communication-dependent tasks under an explicit equal-budget sweep.

**H2 — Transfer penalty.** Task-specific shorthand or learned codes may look efficient in-distribution but lose much of their advantage under unseen task combinations, new receivers, or corrupted messages. A reusable compositional representation should degrade less.

**H3 — Budget interaction.** If a representation's benefit comes from higher effective information density, its relative success advantage should increase as the communication budget tightens, until decoding errors dominate. This predicts a non-monotonic trade-off, not unconditional gains.

**H4 — Setup-cost crossover.** Learning or negotiating a codebook can pay off only after enough repeated exchanges. Its break-even horizon should be predictable from setup cost and per-message savings; for short tasks, familiar formats may win.

**H5 — Receiver dependence.** A compact code optimized for one model or model family can lose to natural/structured text for heterogeneous receivers when decoder mismatch cost is included.

## Measurement commitments

Report task success and calibration/fidelity alongside output and input tokens, serialized bytes, number of turns, wall-clock latency, setup/training cost, local inference time or energy where measurable, parse/decode failures, recovery cost, and cross-model transfer. Plot a Pareto frontier rather than collapsing these dimensions into an arbitrary scalar. Compare both equal-budget and equal-quality operating points.

## Scope exclusions

This thesis does not claim language is inherently superior to latent transfer, tool calls, shared memory, or a transport protocol. These are competing communication mechanisms or different system layers and belong in comparisons where their assumptions can be made explicit. The project will not claim a universal optimum from a finite model/task sample.

## Decision gates

1. **Relevance gate:** reuse Silo-Bench and/or MT-PingEval first; each tested task must require information held by different agents and show a measurable drop when communication is removed.
2. **Baseline gate:** no new format is evaluated against English alone; include optimized concise NL, JSON/schema, code or symbolic forms where appropriate, and current learned/latent methods when feasible.
3. **Cost gate:** count format instructions, codebook setup, decoding and retries. Token savings that shift equal cost elsewhere do not establish efficiency.
4. **Replication gate:** a result must repeat across seeds and at least two receiver conditions before being described as robust.
5. **Existence gate:** if matched existing methods dominate, publish that finding and redirect toward measurement/runtime infrastructure rather than inventing a language.
