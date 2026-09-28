# v0.5 Run Pause and Research Update

**Date:** 2026-09-28  
**Status:** Paused after local machine responsiveness degraded during inference.

## What completed

- All three preregistered Full Profile tasks completed (68 successful responses each). Official HiddenBench scoring gave 1.00 initial and final individual accuracy and 1.00 initial and final majority accuracy for each task.
- Hidden Profile West City completed. Initial and final individual accuracy and majority accuracy were all 0.00.
- Hidden Profile North Hill had 37 generated responses when the local run was stopped. Its task-level result was not written by the upstream CLI, so it is not scoreable. East Town Hidden Profile did not start.
- The local model server and runner have been stopped. No inference process remains active. The in-progress task's partial responses are not included in a result JSON and must not be reconstructed or scored from server timing logs.

The run completed 309 v0.5 inference requests out of 408 planned. Raw task outputs remain in the ignored local `.cache/pilot_hiddenbench_v0_5/` directory; no private benchmark facts or prompts are published here.

## Research implication

The Full Profile result confirms that this local model/configuration can solve these three tasks when it has complete information. The zero West City Hidden Profile score is consistent with a coordination failure, but one completed task is not enough to estimate a treatment effect or compare message formats. No new language or protocol claim follows from v0.5.

The primary-source paper is stronger than the natural-discussion baseline alone. Its 2026 revision reports a two-round **Exchange** followed by one-pass **Decide** protocol (share 1–2 decision-relevant facts, challenge the current front-runner, then state strongest evidence and remaining uncertainty), achieving 0.800 on 18 tasks for GPT-4.1 versus 0.037 baseline. **Reveal-All** is a mechanistic upper-bound diagnostic that appends each agent's full information to its first message; it reaches 0.926, isolating information surfacing as the primary bottleneck. These results mean the next protocol experiment should reproduce the paper's baselines before evaluating a new encoding. The public paper and pinned benchmark details are cited in the project research log.

## Resource and design decision

The Qwen3-14B Q4_K_M model ran with partial CPU offload and 24 GPU layers. The local server accumulated 177,472 CPU-seconds over its lifetime and used approximately 4.7 GB resident memory at inspection. This is cumulative CPU time, not a point-in-time utilization measurement, but the inference run coincided with noticeable system lag. The machine has 16 logical processors. The pinned llama.cpp binary supports separate `--threads` and `--threads-batch` limits; a future run can preregister a four-thread cap for both, then measure actual CPU use and wall time. That setting has not been benchmarked here and is not claimed to guarantee a particular system load. The run was stopped and should remain paused until a lower-impact configuration is selected.

Before more local inference, choose one of these reproducible approaches: cap llama.cpp inference threads and accept longer wall time; use a smaller model that passes the Full Profile gate; or move to a hardware session with more GPU memory. Any changed model/configuration must be recorded as a new preregistered run. Do not mix its measurements with v0.5.

## Next experiments

1. Reproduce the paper's Exchange/Decide and Reveal-All conditions on the same three tasks, using a configuration that does not make the host unresponsive.
2. Measure transcript tokens and bytes, generated tokens, per-call latency, retries, truncations, accuracy, and message-level fact coverage. Separate information surfacing from integration.
3. Only after establishing the stronger baselines, compare full-fact disclosure, structured fact/uncertainty records, and natural-language exchange under matched communication budgets.
4. Expand to more benchmark tasks and seeds before making claims about generalization or efficiency frontiers.
