# HiddenBench Qwen3-14B reasoning-mode screen v0.4

**Run date:** 2026-09-28  
**Model:** Qwen3-14B Q4_K_M, 24 GPU layers with CPU offload, 8,192 context.  
**Tasks:** three official HiddenBench verification tasks, full-information initial votes only.

## Result

The model answered **12/12** initial votes correctly (4/4 on each task), passing the preregistered 0.8 capability gate. The prior v0.3 screen on the same checkpoint, tasks, and task seeds scored 4/12. v0.4 enabled Qwen3 thinking and changed decoding to temperature 0.6, top-k 20, top-p 0.95, and presence penalty 1.5, following the [official Qwen3-14B model card](https://huggingface.co/Qwen/Qwen3-14B-GGUF). Because reasoning mode and sampling settings changed together, this comparison does not isolate which change accounts for the difference.

The 12 requests generated 7,284 tokens in 930.024 seconds of summed service time (77.5 seconds per completion on average). There were no retries or truncated responses. This shows a viable full-information operating point, with substantial inference cost on this 8 GB laptop GPU plus CPU offload. See [sanitized results](../experiments/hiddenbench_v0_4/RESULTS.json); private rationales and task records remain local under ignored `.cache/pilot_hiddenbench_v0_4/`.

## Decision

Proceed to the official 15-round full/hidden HiddenBench natural-discussion baseline under these exact model and decoding settings. This screen licenses the full baseline only; it does not establish that communication helps or that a new protocol will improve on natural discussion. A protocol comparison requires a separate preregistration after this baseline and local cost analysis.
