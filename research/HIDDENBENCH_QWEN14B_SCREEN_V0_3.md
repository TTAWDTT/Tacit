# HiddenBench Qwen3-14B full-information capability screen v0.3

**Run date:** 2026-09-28  
**Model:** Qwen3-14B Q4_K_M; 24 GPU layers with CPU offload, 8,192 context.  
**Task source:** pinned HiddenBench three-task verification split ([paper](https://arxiv.org/abs/2505.11556), [MIT code/data](https://github.com/Yassellee/HiddenBench_ICML)).

## Result

Qwen3-14B correctly answered **4 of 12** full-information initial votes (mean individual accuracy **0.333**), below the preregistered 0.8 gate. It answered all four votes correctly on `evacuation_east_town`, and none on either of the other two tasks. The 12 requests generated 597 tokens in 72.687 seconds of summed server time. There were no retries or truncated completions.

The paired Qwen3-8B v0.2 screen scored 2/12 (0.167) on the same task/seed schedule. The 14B run's increase to 4/12 comes entirely from one task; this three-task descriptive difference is not a causal scaling result. The full per-task counts and runtime summary are [RESULTS.json](../experiments/hiddenbench_v0_3/RESULTS.json). Private votes, rationales, and assignments remain in ignored `.cache/pilot_hiddenbench_v0_3/`.

## Decision

The larger cached local checkpoint does not meet the capability gate either. No communication protocol comparison is justified on these tasks with these checkpoints. The result points to a task/model operating-region problem: the full-information control is already too difficult for the locally tested models. Next work should investigate a validated simpler hidden-information task or a better-performing local prompt/model configuration, while preserving a full-information control before any protocol claims.
