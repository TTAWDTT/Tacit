# HiddenBench phase-policy pilot v0.6: interrupted run

**Status:** incomplete; no protocol comparison is possible.

Machine-readable sanitized aggregates are available in [`HIDDENBENCH_PHASE_PILOT_V0_6_INTERRUPTED.json`](HIDDENBENCH_PHASE_PILOT_V0_6_INTERRUPTED.json).

The preregistered single-task run began on 2026-09-28 with Qwen3-14B Q4_K_M, one hidden-profile HiddenBench task, four agents, three rounds, and a fixed seed. The first condition (`natural_3`) completed. The run was then stopped during `exchange_decide` after host CPU samples reached 89.3% and remained around 77–79% in subsequent samples. This stop honored the machine-responsiveness constraint; llama.cpp and the runner were terminated, and GPU memory use fell to about 1.3/8.2 GiB. The third condition (`reveal_all_3`) did not run.

## Completed condition

| Condition | Completion | Initial individual accuracy | Final individual accuracy | Initial majority accuracy | Final majority accuracy | Messages | Serialized UTF-8 bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `natural_3` | Complete | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 4,354 |

The official HiddenBench scorer reports zero accuracy at both stages for this one task and seed. This is a single descriptive observation, not evidence that natural discussion generally fails or that another protocol would do better. Discussion-token count is omitted because the server tokenizer endpoint was unavailable after the safe shutdown.

The completed condition used 20 model requests: 6,627 prompt-evaluation tokens and 13,153 generated tokens according to the llama.cpp log, with no truncated completions. The run's 137 resource samples averaged 54.1% host CPU (maximum 89.3%), 34.2% GPU utilization (maximum 72.0%), and 3.3 server CPU cores (maximum 3.7). The two requests initiated for `exchange_decide` are incomplete and excluded from scoring and completed-condition cost totals.

## Interpretation and next step

This run provides no estimate of relative protocol quality, no efficiency frontier, and no scaling evidence. Its useful outcomes are (1) the phase-aware runner completed the natural condition and produced a scoreable result, (2) this difficult episode remained unsolved after three rounds of natural discussion, and (3) the capped local inference setup still caused high host load over a long run.

Do not resume this 14B setup unchanged. Before further model inference, replace the long three-condition run with short, isolated calibration episodes and an explicit lower resource ceiling or smaller model. Keep the preregistered v0.6 result immutable; any changed model or resource configuration is a new pilot. The private raw transcript remains under ignored `.cache/` and is not published.
