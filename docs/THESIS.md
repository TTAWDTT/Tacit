# Initial research thesis

**Status:** v0.1, 2026-09-27. This is a falsifiable starting point, not a conclusion.

## Thesis

There is no reason to expect one universally optimal language for all LLM-to-LLM communication. The useful object of study is a task- and receiver-conditioned communication policy over representations and interaction acts. A candidate representation earns value only if it improves the task-success versus *total* communication-and-computation cost frontier, while preserving semantic fidelity and transfer under the intended deployment conditions.

The first research goal is therefore not to invent notation. It is to identify a reproducible setting where the strongest existing options fail, quantify the failure, and determine whether a reusable compositional protocol can repair it.

## Why this is a plausible research gap

- Non-natural-language formats are already an established baseline: Chen et al. (Findings EMNLP 2024) report model-selected formats, transfer to different LLMs, and up to 72.7% lower token use in their multi-agent communication experiments. Any new proposal must reproduce or outperform the relevant baseline under matched conditions.
- Latent collaboration papers report efficiency/quality gains, but latent representations introduce compatibility, transmission-size, decoding, runtime, and auditability questions. Their claims should be evaluated in the native execution setting and against text baselines with equivalent task opportunity.
- Production agent protocols such as A2A chiefly standardize interoperability, task lifecycle, and artifacts. These are important system layers but do not settle which semantic encoding is most efficient.
- Emergent-language results caution that compositionality and generalization are distinct outcomes. We must measure both directly rather than infer one from the other.

## Falsifiable hypotheses

**H1 — Conditional representation gains.** A representation chosen for the task and receiver can improve the quality-cost frontier over a fixed natural-language prompt on at least one preregistered family of genuinely communication-dependent tasks.

**H2 — Transfer penalty.** Task-specific shorthand or learned codes may look efficient in-distribution but lose much of their advantage under unseen task combinations, new receivers, or corrupted messages. A reusable compositional representation should degrade less.

**H3 — Budget interaction.** If a representation's benefit comes from higher effective information density, its relative success advantage should increase as the communication budget tightens, until decoding errors dominate. This predicts a non-monotonic trade-off, not unconditional gains.

**H4 — Setup-cost crossover.** Learning or negotiating a codebook can pay off only after enough repeated exchanges. Its break-even horizon should be predictable from setup cost and per-message savings; for short tasks, familiar formats may win.

**H5 — Receiver dependence.** A compact code optimized for one model or model family can lose to natural/structured text for heterogeneous receivers when decoder mismatch cost is included.

## Measurement commitments

Report task success and calibration/fidelity alongside output and input tokens, serialized bytes, number of turns, wall-clock latency, setup/training cost, local inference time or energy where measurable, parse/decode failures, recovery cost, and cross-model transfer. Plot a Pareto frontier rather than collapsing these dimensions into an arbitrary scalar. Compare both equal-budget and equal-quality operating points.

## Scope exclusions

This thesis does not claim language is inherently superior to latent transfer, tool calls, shared memory, or a transport protocol. These are competing communication mechanisms or different system layers and belong in comparisons where their assumptions can be made explicit. The project will not claim a universal optimum from a finite model/task sample.

## Decision gates

1. **Relevance gate:** a task must require private information held by different agents and show a measurable drop when communication is removed.
2. **Baseline gate:** no new format is evaluated against English alone; include optimized concise NL, JSON/schema, code or symbolic forms where appropriate, and current learned/latent methods when feasible.
3. **Cost gate:** count format instructions, codebook setup, decoding and retries. Token savings that shift equal cost elsewhere do not establish efficiency.
4. **Replication gate:** a result must repeat across seeds and at least two receiver conditions before being described as robust.
5. **Existence gate:** if matched existing methods dominate, publish that finding and redirect toward measurement/runtime infrastructure rather than inventing a language.
