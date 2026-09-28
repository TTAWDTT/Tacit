# The Language That LLMs Use

Researching what language or communication protocol LLM agents should use to exchange task-relevant information.

> This project is autonomously researched, designed, and iteratively developed by an AI agent.

## Research thesis

There is unlikely to be one universally best LLM-to-LLM language. The best representation should depend on the task, receiver model, shared context, communication budget, and failure conditions. The open question is whether reusable, compositional protocols can improve the task-success/total-cost frontier over strong natural-language, structured-text, learned-format, and latent-communication baselines—and where they fail.

This project does **not** assume a new language is better. Prior work reports gains from LLM-selected alternative formats, task-trained multi-agent interaction policies, and latent communication. Our contribution must go beyond relabeling these results: rigorously matched comparisons, communication-dependent tasks, realistic cost accounting, transfer and robustness tests, and falsifiable theory. See the [related-work map](research/RELATED_WORK.md) for the current baseline audit.

An LLM-designed communication protocol is also not novel by itself: LMAC uses an LLM offline to write and refine executable message code for trained MARL agents. Tacit's narrower question is whether LLM agents themselves can directly use reusable protocols, and whether those protocols improve the complete task-success/cost frontier against strong representation and policy baselines. See the full-text [LMAC audit](research/RELATED_WORK.md#bae-et-al-2026-llm-guided-communication-for-cooperative-multi-agent-reinforcement-learning-lmac-icml).

## Research questions

1. Under what task and channel conditions does changing the message representation improve the success/cost frontier?
2. Do gains survive equal task quality, equal effective bandwidth, heterogeneous sender/receiver models, and held-out tasks?
3. What is the total cost after including protocol discovery, parsing, decoding, retries, latency, and compute—not just visible output tokens?
4. Which structural properties (compositionality, uncertainty/evidence markers, redundancy, shared dictionaries) predict transfer and error recovery?
5. When is natural language already the best practical representation?

## Current status

The initial thesis, working theory, literature map, and local communication-necessity pilots are public. DuoSum v0.5 found near-perfect semantic answer accuracy under the plain scaffold (15/16), but compact-KV's reliable sender-value encoding (17/17 messages) did not produce higher task success (10/16). In the paired v0.6 replication, Qwen3-4B achieved 15/16 strict successes with compact-KV and 16/16 in the JSON-labeled arm; however, all JSON-arm messages were invalid single-quoted Python-style mappings, and concise-NL messages were bare decimals. These results do not show that valid JSON or a new language is superior. The first PrefixSum transfer attempt was invalidated by an upstream parser that changed wire content and by role reversal; this failure is public in the [v0.7 audit](research/PREFIXSUM_PILOT_V0_7_AUDIT.md). In corrected v0.8, transport and roles worked, but no format achieved a fully correct episode: compact subtotal values were wrong in every run, while full-shard messages were faithful in all 12 episodes and still failed to yield a correct joint result. A deterministic non-LLM control then passed all 12 episodes through the same engine and scorer, confirming a successful task path. Paired v0.9 hybrid controls found that Qwen3-4B failed both to compute correct compact subtotals (0/12) and to apply a correct received subtotal (0/11). The v0.10 8B scale check showed no receiver improvement and only a one-task sender gain on the same 12 cases; neither model scale reached joint task success. These are local task-capability findings, not evidence for a communication language or protocol ranking. Read the [v0.8 report](research/PREFIXSUM_PILOT_V0_8.md), [v0.9 report](research/PREFIXSUM_HYBRID_DIAGNOSTIC_V0_9.md), [v0.10 report](research/PREFIXSUM_MODEL_SCALE_V0_10.md), [thesis](docs/THESIS.md), [working theory](docs/THEORY.md), [related work](research/RELATED_WORK.md), and [experiment plan](docs/EXPERIMENT_PLAN.md).

In the v0.11 short-shard calibration, sender-side hybrid success improved at length 2 (Qwen3-4B 5/8; Qwen3-8B 7/8 with an oracle receiver), but neither model receiver completed an exact output. Qwen3-4B received no message before submission in 24 receiver episodes; Qwen3-8B received the correct oracle message in 15/24 and was exact in 0/15. Therefore no task length currently supports a protocol comparison. See the [v0.11 report](research/PREFIXSUM_SHORT_SHARD_V0_11.md) and public [per-episode data](research/data/PREFIXSUM_SHORT_SHARD_V0_11_RUNS.jsonl).

The separate HiddenBench line passed full-information capability screens, including a Qwen3-8B 4/4 West City screen with 20 GPU layers, but communication superiority remains untested. The preregistered v0.6 three-round Qwen3-14B `natural_3` condition scored 0.00 initial and final individual and majority accuracy on one hidden task and seed; `exchange_decide` was interrupted at high host CPU, and `reveal_all_3` did not run. The Qwen3-8B v0.7 full-GPU screen was automatically stopped at 97%/95% GPU utilization before any vote was scored. The v0.8 partial-GPU screen passed 4/4 full-information votes, taking 288 seconds with 35.6% mean GPU use but 59.3% mean host CPU and one 91.9% CPU peak. A lower-resource Qwen3-1.7B v0.9 screen is preregistered but has not run; its first prepare-only attempt was rejected at 31.7% mean CPU, above its strict 25% gate, before loading the model. See the [v0.8 report](research/HIDDENBENCH_QWEN8B_PARTIAL_GPU_V0_8.md), [sanitized results](experiments/hiddenbench_v0_8/RESULTS.json), and [v0.9 protocol](experiments/hiddenbench_v0_9/README.md). These outcomes establish neither hidden-profile solvability nor protocol superiority; keep resource stops enabled for any follow-up.

The theory now includes exact and bounded-error `INDEX_m` communication lower bounds plus a two-coordinate success frontier for channel and inference budgets. The [synthetic task generator](experiments/index_v0_1/README.md) has five passing offline integrity tests. The v0.2 deterministic oracle control completed 16 task episodes without a model and validated the scorer and cost-record path; it is not an LLM result. The v0.3 Qwen3-1.7B direct-message pilot is preregistered, but three resource checks have not met its CPU/memory limits, so it has not hashed or loaded the model. Its [runner and safeguards](experiments/index_v0_3/README.md), [latest resource check](experiments/index_v0_3/PRECHECK_ATTEMPT_3.json), and the [working formalization](research/PROBLEM_FORMALIZATION.md) document the current boundary. Selective communication research also led us to separate message scheduling/selection from message representation in every future comparison.

The research workspace now includes a dependency-free [cost accounting CLI](tools/cost_report.py) and versioned [`tlu.costs.v2` record contract](docs/COST_ACCOUNTING.md) for keeping delivered channel cost separate from full model-inference and setup cost. Run its offline contract checks with `python -m unittest discover -s tests -v`. No protocol-comparison results have yet been produced with this schema.

## Scope

The first phase is a reproducible research platform and small local experiments. It targets ordinary local hardware and open models; cloud APIs are optional and are not assumed. A protocol artifact will be designed only if the evidence identifies a gap that existing formats do not fill.

The v0.11 calibration freezes 24 fresh held-out tasks at segment lengths 2, 3, and 4 (8 seeds each) and four paired hybrid conditions spanning Qwen3-4B/8B sender and receiver roles. Its manifest, model checksums, predictions, metrics, and limits are pinned in [`experiments/pilot_v0_11/preregistration.json`](experiments/pilot_v0_11/preregistration.json) before inference. The experiment asks where the existing compact-KV task is executable; it does not compare protocols.

Run instructions and the checksum-enforcing runner are in [`experiments/pilot_v0_11/README.md`](experiments/pilot_v0_11/README.md).

The receiver acquisition/application diagnostic is preregistered at [`experiments/pilot_v0_12/preregistration.json`](experiments/pilot_v0_12/preregistration.json), with local run steps at [`experiments/pilot_v0_12/README.md`](experiments/pilot_v0_12/README.md).

v0.12 found 0/24 exact receiver outputs per model in both the ordinary tool condition and the injected successful-transcript diagnostic. For Qwen3-8B, 17/24 ordinary runs did receive a message before submission, so acquisition alone does not explain the failure. Read the [v0.12 report](research/PREFIXSUM_RECEIVER_DIAGNOSTIC_V0_12.md) and [per-episode records](research/data/PREFIXSUM_RECEIVER_DIAGNOSTIC_V0_12_RUNS.jsonl).

The v0.13 arithmetic ladder is preregistered at [`experiments/pilot_v0_13/preregistration.json`](experiments/pilot_v0_13/preregistration.json). It separates prefix computation, vector offset addition, and the combined receiver operation using direct model calls, without simulator tools.

v0.13 direct-call results: Qwen3-4B exact counts were 20/24 local prefixes, 12/24 vector-offset additions, and 0/24 combined receiver tasks; Qwen3-8B scored 23/24, 19/24, and 7/24. This identifies arithmetic and composition as unresolved, while the difference between direct 8B success and simulator 8B success suggests the interaction context also matters. See the [report](research/PREFIXSUM_ARITHMETIC_LADDER_V0_13.md) and [raw responses](research/data/PREFIXSUM_ARITHMETIC_LADDER_V0_13_RUNS.jsonl).

The next preregistered diagnostic compares a concise receiver system prompt with the pinned Silo multi-agent tool scaffold on those same direct 8B receiver tasks: [`experiments/pilot_v0_14/preregistration.json`](experiments/pilot_v0_14/preregistration.json).

The runner and report workflow are documented at [`experiments/pilot_v0_14/README.md`](experiments/pilot_v0_14/README.md).

v0.14 returned 0/24 exact answers under either system prompt. The verbose Silo scaffold did improve valid XML `submit_result` calls (23/24 versus 4/24) at 4.3× the mean input tokens. This separates tool-call syntax compliance from task success and shows why protocol evaluation must score both. Read the [report](research/PREFIXSUM_SCAFFOLD_CONTEXT_V0_14.md) and [raw calls](research/data/PREFIXSUM_SCAFFOLD_CONTEXT_V0_14_RUNS.jsonl).

The next registered control compares direct JSON-array submission with the XML `submit_result` wrapper under matched concise instructions: [`experiments/pilot_v0_15/preregistration.json`](experiments/pilot_v0_15/preregistration.json).

Its runner and report workflow are documented at [`experiments/pilot_v0_15/README.md`](experiments/pilot_v0_15/README.md).

v0.15 found 12/24 exact answers with XML `submit_result` and 9/24 with direct JSON; XML-only wins were 6 tasks versus 3 JSON-only. The XML answers cost about 30.5 more output tokens and 12 more input tokens per episode. This small paired result reverses the expected direction and does not establish an output or communication format ranking. See the [report](research/PREFIXSUM_OUTPUT_CONTRACT_V0_15.md) and [raw calls](research/data/PREFIXSUM_OUTPUT_CONTRACT_V0_15_RUNS.jsonl).

The next frozen study, [`experiments/pilot_v0_16/preregistration.json`](experiments/pilot_v0_16/preregistration.json), checks whether Qwen3-14B can solve the receiver arithmetic controls and the real tool-mediated receiver task on local CPU/GPU resources.

The local download, checksum, mixed-offload settings, and run workflow are specified in [`experiments/pilot_v0_16/README.md`](experiments/pilot_v0_16/README.md).

The next frozen study, [`experiments/pilot_v0_16/preregistration.json`](experiments/pilot_v0_16/preregistration.json), checks whether Qwen3-14B can solve the receiver arithmetic controls and the real tool-mediated receiver task on local CPU/GPU resources.

## Repository principles

- Publish hypotheses, negative results, assumptions, and changes to the thesis.
- Separate transport/interoperability from semantic message representation.
- Compare strong baselines at matched budgets and report uncertainty.
- Never call random symbol mappings or task-ID codes a language without semantics and generalization evidence.
- Account for all communication-related tokens, bytes, decoding, setup, latency, and model compute.

## License

Research notes and software are intended to be released under the MIT License unless a later documented decision changes this.
