# v0.5 Full-Profile Interim Result

**Status:** Full Profile complete; the three Hidden Profile task runs are in progress.

The pre-registered local capability check has passed on all three pinned HiddenBench verification tasks. The official scorer reports perfect initial and final individual accuracy and perfect initial and final majority accuracy for each task:

| Task | Initial individual | Final individual | Initial majority | Final majority |
|---|---:|---:|---:|---:|
| West City | 1.00 | 1.00 | 1.00 | 1.00 |
| North Hill | 1.00 | 1.00 | 1.00 | 1.00 |
| East Town | 1.00 | 1.00 | 1.00 | 1.00 |

Each task produced all 68 planned successful outputs and was persisted independently. The three runs each took about 72 minutes on an RTX 4060 Laptop GPU with Qwen3-14B Q4_K_M and 24 GPU layers plus CPU offload. This is hardware-specific service cost.

This result establishes that the local model can solve these tasks with full information. It does not measure the causal value of communication, and it is not evidence that natural discussion or a new protocol is superior. The Hidden Profile is necessary to assess whether dispersed information is recovered through discussion. This is one model, one seed per task, and three related scenarios; it is a calibration gate, not a generalization claim.

## Reproduction

The run used the frozen configuration in [`preregistration.json`](preregistration.json) and the pinned official scorer. Raw task output is retained under the ignored `.cache/pilot_hiddenbench_v0_5/` directory. The per-task metrics were computed with `hiddenbench.metrics.score_results` from the pinned local HiddenBench source revision.
