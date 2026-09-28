# HiddenBench Qwen3-8B partial-GPU capability screen v0.8

**Status:** complete; passed the preregistered one-task capability gate.

The v0.8 screen used the pinned Qwen3-8B Q4_K_M checkpoint, HiddenBench West City, full-information profile, seed `20260927`, and the official Qwen3 thinking-mode sampler. Twenty of 36 model layers were offloaded to the GPU. The official HiddenBench scorer reports **4/4 initial votes correct**, so this checkpoint meets the preregistered 4/4 eligibility threshold for a later communication study on this task/configuration.

This is one task and one seed. It does not establish hidden-profile solvability, communication benefit, protocol superiority, generalization, or a compute advantage over the 14B setup. Earlier Qwen3-8B runs used different inference settings and broader task sets; their results are not a matched comparison.

## Inference and resource accounting

The four sequential requests took 288.3 seconds wall time and 287.7 seconds summed llama.cpp service time. The server recorded 1,404 prompt-evaluation tokens and 2,804 generated tokens, with no truncations. Eighteen resource samples averaged 59.3% host CPU (maximum 91.9%), 35.6% GPU utilization (maximum 43%), and 3.43 server CPU cores (maximum 3.67). GPU memory averaged 4.12/8.19 GiB; the server was stopped after completion and memory returned to about 0.69 GiB.

The automatic stop did not trigger. The isolated 91.9% host CPU sample was followed by a lower sample; sustained-load thresholds were not met. The result confirms the 20-layer placement materially reduced GPU utilization relative to v0.7's 99-layer run, but host CPU still reached a high transient, so a multi-condition study needs the same automatic stop and careful resource sampling.

Machine-readable sanitized aggregates are in [`experiments/hiddenbench_v0_8/RESULTS.json`](../experiments/hiddenbench_v0_8/RESULTS.json). Private raw records remain under ignored `.cache/pilot_hiddenbench_v0_8/`.

## Decision

The pass permits a separately preregistered hidden-profile communication pilot on this task and model setting; it does not justify restarting the 60-call v0.6 plan unchanged. Any next pilot should limit its condition count, preserve the same task/model/inference configuration across arms, count actual requests and tokens, include natural-language and optimized-format baselines, and stop under the frozen resource thresholds. Do not interpret the v0.6 Qwen3-14B result as a direct baseline for Qwen3-8B.
