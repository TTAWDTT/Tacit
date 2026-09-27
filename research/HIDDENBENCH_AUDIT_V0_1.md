# HiddenBench audit v0.1

**Audit date:** 2026-09-27  
**Paper:** Li, Naito, and Shirado, *Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs* ([arXiv](https://arxiv.org/abs/2505.11556), version 4 dated 2026-05-13)  
**Code/data:** [Yassellee/HiddenBench_ICML](https://github.com/Yassellee/HiddenBench_ICML), MIT license, pinned local source revision `3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1`.

## Why this benchmark is relevant

HiddenBench uses a hidden-profile design: group members see shared facts plus disjoint private facts; the shared facts favor a wrong answer, while combining the private facts identifies the correct one. The paper checks the same tasks under full information and hidden information, helping distinguish communication failure from task reasoning difficulty. It reports average answer accuracy of 30.1% for multi-agent systems under hidden information and 80.7% for single agents with complete information across 65 tasks. It also reports that a lightweight two-stage Exchange / Decide protocol improved results on 18 tasks, five runs each, across three frontier models. These are strong task and protocol baselines that should be reproduced before proposing another representation.

The public `benchmark_short.json` contains the paper's three manually designed verification tasks. Each has four shared facts, four distinct hidden facts, and three possible answers; all use four agents. The full local benchmark contains 65 tasks. The task generation workflow enforces individual-full-information and hidden-information difficulty thresholds, although the existing task suite remains a decision benchmark rather than a long-horizon environment.

## Implementation findings

- `simulator.run_scenario` begins with one JSON-constrained vote per agent, runs a sequential round-robin discussion (15 rounds by default), then asks each agent for a final JSON-constrained vote. Default episode length is 4 initial votes + 60 messages + 4 final votes = 68 model completions.
- Each agent sees the same shared facts plus one unique hidden fact in `hidden` profile. Under `full`, each sees every hidden fact. Available answer labels and scoring are explicit.
- All agents use one shared `ModelClient`. The official OpenAI-compatible adapter sends chat completions and JSON mode; it has no token/byte budget, cost tracing, or explicit latency accounting. The project must add transport and inference accounting without changing the task/scorer.
- The open-source code supports a local OpenAI-compatible endpoint. Local smoke preflight should confirm the pinned llama.cpp server honors JSON-mode replies for Qwen3-8B.
- The official `extra_prompt` hook can test prompts, but it is not sufficient by itself to reproduce the paper's phase-changing Exchange/Decide protocol. Do not label a static extra prompt as that stronger baseline.
- The result JSON stores private fact assignments, all prompts, rationales, and full transcripts. Keep raw traces local under `.cache`; publish sanitized aggregate metrics and redact private reasoning from public artifacts.

## Suitability and limits for Tacit

Use HiddenBench as a validated distributed-information task family, not as proof that a proposed language will generalize. Its full-profile/hide-profile controls and gold labels give a useful causal foundation, and its public Exchange/Decide baseline directly challenges any claimed protocol contribution. However, its three short verification tasks are too small for strong generalization claims, and the 65-item set is still finite. The task asks agents to deliberate and choose one option; it does not test long-horizon plans, compositional machine protocols, dynamic bandwidth schedules, or heterogeneous-model transfer by default.

The first local stage should run the three verification tasks on the pinned Qwen3-8B model in both profiles, at the paper's 15 rounds and one fixed seed. Treat this as a capability/adapter feasibility check, not protocol evidence. If full-profile performance is poor, diagnose individual reasoning before comparing communication formats. If it is good while hidden-profile performance is weak, compare original natural discussion with the paper's phase-structured baseline and a token-budgeted protocol on the 65-task suite. Record exact completion calls, prompt/output tokens including cache behavior, serialized message tokens/bytes, latency, task accuracy, and cross-agent answer transfer.

## Main evidence-based prediction

If local single-agent full-profile accuracy is materially above hidden-profile group accuracy, increasing discussion length alone should not be assumed to close the gap. The paper reports that 15 rounds outperformed 20 rounds in one GPT-4.1 ablation and that its Exchange/Decide structure improved hidden-profile accuracy across three frontier models. A falsifiable local prediction is therefore: on this Qwen3-8B setup, a targeted information-exchange schedule will outperform unstructured discussion at equal total decoded-message-token budget only if the single-agent full-profile gate is first met. No communication format is presumed to win.

## Source notes

- The paper reports full-vs-hidden and protocol values in the primary article and appendices; they should not be treated as local-model expectations.
- The GitHub `README` states that 65 tasks are distributed under MIT and documents the OpenAI-compatible client interface.
- This audit did not modify the third-party checkout; it lives under ignored `.cache/research/HiddenBench_ICML/`.
