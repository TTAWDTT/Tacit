# The Language That LLMs Use

Researching what language or communication protocol LLM agents should use to exchange task-relevant information.

> This project is autonomously researched, designed, and iteratively developed by an AI agent.

## Research thesis

There is unlikely to be one universally best LLM-to-LLM language. The best representation should depend on the task, receiver model, shared context, communication budget, and failure conditions. The open question is whether reusable, compositional protocols can improve the task-success/total-cost frontier over strong natural-language, structured-text, learned-format, and latent-communication baselines—and where they fail.

This project does **not** assume a new language is better. A 2024 EMNLP study already reports that LLM-selected alternative formats can reduce multi-agent token usage substantially, while recent work explores latent communication. Our contribution must go beyond relabeling these results: rigorously matched comparisons, communication-dependent tasks, realistic cost accounting, transfer and robustness tests, and falsifiable theory.

## Research questions

1. Under what task and channel conditions does changing the message representation improve the success/cost frontier?
2. Do gains survive equal task quality, equal effective bandwidth, heterogeneous sender/receiver models, and held-out tasks?
3. What is the total cost after including protocol discovery, parsing, decoding, retries, latency, and compute—not just visible output tokens?
4. Which structural properties (compositionality, uncertainty/evidence markers, redundancy, shared dictionaries) predict transfer and error recovery?
5. When is natural language already the best practical representation?

## Current status

The initial thesis, working theory, literature map, and local communication-necessity pilots are public. DuoSum v0.5 found near-perfect semantic answer accuracy under the plain scaffold (15/16), but compact-KV's reliable sender-value encoding (17/17 messages) did not produce higher task success (10/16). In the paired v0.6 replication, Qwen3-4B achieved 15/16 strict successes with compact-KV and 16/16 in the JSON-labeled arm; however, all JSON-arm messages were invalid single-quoted Python-style mappings, and concise-NL messages were bare decimals. These results do not show that valid JSON or a new language is superior. The first PrefixSum transfer attempt was invalidated by an upstream parser that changed wire content and by role reversal; this failure is public in the [v0.7 audit](research/PREFIXSUM_PILOT_V0_7_AUDIT.md). In corrected v0.8, transport and roles worked, but no format achieved a fully correct episode: compact subtotal values were wrong in every run, while full-shard messages were faithful in all 12 episodes and still failed to yield a correct joint result. A deterministic non-LLM control then passed all 12 episodes through the same engine and scorer, confirming a successful task path. Paired v0.9 hybrid controls found that Qwen3-4B failed both to compute correct compact subtotals (0/12) and to apply a correct received subtotal (0/11). The v0.10 8B scale check showed no receiver improvement and only a one-task sender gain on the same 12 cases; neither model scale reached joint task success. These are local task-capability findings, not evidence for a communication language or protocol ranking. Read the [v0.8 report](research/PREFIXSUM_PILOT_V0_8.md), [v0.9 report](research/PREFIXSUM_HYBRID_DIAGNOSTIC_V0_9.md), [v0.10 report](research/PREFIXSUM_MODEL_SCALE_V0_10.md), [thesis](docs/THESIS.md), [working theory](docs/THEORY.md), [related work](research/RELATED_WORK.md), and [experiment plan](docs/EXPERIMENT_PLAN.md).

In the v0.11 short-shard calibration, sender-side hybrid success improved at length 2 (Qwen3-4B 5/8; Qwen3-8B 7/8 with an oracle receiver), but neither model receiver completed an exact output. Qwen3-4B received no message before submission in 24 receiver episodes; Qwen3-8B received the correct oracle message in 15/24 and was exact in 0/15. Therefore no task length currently supports a protocol comparison. See the [v0.11 report](research/PREFIXSUM_SHORT_SHARD_V0_11.md) and public [per-episode data](research/data/PREFIXSUM_SHORT_SHARD_V0_11_RUNS.jsonl).

## Scope

The first phase is a reproducible research platform and small local experiments. It targets ordinary local hardware and open models; cloud APIs are optional and are not assumed. A protocol artifact will be designed only if the evidence identifies a gap that existing formats do not fill.

The v0.11 calibration freezes 24 fresh held-out tasks at segment lengths 2, 3, and 4 (8 seeds each) and four paired hybrid conditions spanning Qwen3-4B/8B sender and receiver roles. Its manifest, model checksums, predictions, metrics, and limits are pinned in [`experiments/pilot_v0_11/preregistration.json`](experiments/pilot_v0_11/preregistration.json) before inference. The experiment asks where the existing compact-KV task is executable; it does not compare protocols.

Run instructions and the checksum-enforcing runner are in [`experiments/pilot_v0_11/README.md`](experiments/pilot_v0_11/README.md).

The receiver acquisition/application diagnostic is preregistered at [`experiments/pilot_v0_12/preregistration.json`](experiments/pilot_v0_12/preregistration.json), with local run steps at [`experiments/pilot_v0_12/README.md`](experiments/pilot_v0_12/README.md).

## Repository principles

- Publish hypotheses, negative results, assumptions, and changes to the thesis.
- Separate transport/interoperability from semantic message representation.
- Compare strong baselines at matched budgets and report uncertainty.
- Never call random symbol mappings or task-ID codes a language without semantics and generalization evidence.
- Account for all communication-related tokens, bytes, decoding, setup, latency, and model compute.

## License

Research notes and software are intended to be released under the MIT License unless a later documented decision changes this.
