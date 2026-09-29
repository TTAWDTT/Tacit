# Cache-to-Cache audit v0.1: a cross-model KV representation channel

**Status:** static review of Fu et al. (ICLR 2026) and the authors' public implementation pinned at `9fa85d35a20e20469af3e282926491222f4f11e0` (shallow clone under this project's ignored `.cache/research/`). No source was executed, no model weights were downloaded, and no training or inference was run.

## What C2C exchanges

Cache-to-Cache (C2C) computes receiver and sharer KV caches for the same input context. A learned fuser projects the sharer's cache into the receiver's cache representation and combines both with a residual path and learned per-layer gate; the receiver then decodes using the fused cache. It aligns tokens across tokenizers and layers across model depths, and trains the fuser while freezing both language models.

This is a learned cross-model representation channel, not a symbolic or textual language. It is a strong systems baseline when both models expose internal caches and a pair-specific fuser can be trained. The paper evaluates Qwen, Llama, and Gemma pairings and model sizes from 0.6B to 14B, which is materially broader than same-backbone-only systems such as TFlow. It still requires access to both models' forward passes, internal KV state, and a trained cache adapter; it does not work through ordinary black-box text APIs.

## Published evidence and limits

The paper reports C2C above its text-to-text (T2T) handoff baseline across its model-pair and benchmark settings, with average gains varying by sharer, and an average latency speedup of about 2.5x. Its core Table 4 pairs a fixed Qwen3-0.6B receiver with three sharers across MMLU-Redux, OpenBookQA, ARC-Challenge, and C-Eval. On one pair, reported C2C times are 0.30–0.40 seconds versus 0.81–1.52 seconds for T2T. It also reports higher LongBench scores than T2T in all three context-length bins (0–4k, 4–8k, and 8k+). These are useful conditional results, but context-bin accuracy is not a communication-complexity or equal-byte scaling law.

The paper's latency experiment counts both model input processing and includes a reported 90 ms for KV fusion on one NVIDIA A100, batch size one. That makes its latency accounting more complete than a receiver-only timing, but it is still an in-process/device-local result: neither the paper tables nor the public runtime measure serialized cache bytes, network transfer, or multi-host synchronization. A full cache payload scales with sequence length, KV layers/heads, and scalar width; “same cache length” does not mean “small communication bandwidth.”

The paper trains fusers on up to 500,000 OpenHermes2.5 examples, while selected analysis experiments use MMLU's auxiliary training split. Its appendix reports approximately 6.95 GPU-hours at 300 steps and 44.72 GPU-hours at 1,929 steps for the Qwen2.5-0.5B/Qwen3-0.6B pair; one 300-step score is comparable to or above the final checkpoint, so training budget and checkpoint selection are part of the result rather than a fixed negligible constant. The paper also reports cases where a weaker sharer misleads a stronger receiver and reduces accuracy.

## Receiver-need and task boundary

In the main C2C design, both models prefill the shared input query/context. The paper's oracle studies show that a model's cache can enrich another model's representation of the same context, and the benchmark results show that this can improve downstream predictions. However, the benchmarks do not establish that communication is necessary because the receiver has no private-information partition: it receives the same question the sharer encoded. The authors identify real agentic, multi-round communication as future work. Tacit should therefore treat C2C as collaborative model-state exchange and test it on receiver-need tasks where sharers receive private evidence or partial traces that the receiver does not see.

## Public implementation boundary

The pinned Rosetta wrapper consolidates the receiver model, sharer models, and projectors in one PyTorch module. The official live-chat example loads all models onto the same device and applies the fuser during generation. This demonstrates a local reference integration, not a distributed transport protocol. The repository's September 2026 README says agent-managed KV-cache serving is forthcoming; the present code does not define a portable wire format or network runtime. Do not score cache handoff as zero bytes because tensors share one process/device.

## Implications for Tacit's comparison

Keep C2C in a distinct **internal KV-state transfer** stratum, separate from black-box text, Interlat's hidden-state message stream, and TFlow's weight-space update. A faithful comparison should:

- keep the same task examples, receiver prompts, and answer budget across C2C and T2T;
- use true, same-shape other-example, neutral/zero-sharer, receiver-only, and same-model versus heterogeneous-sharer controls;
- require a receiver-need split with private sharer evidence before claiming necessary communication;
- measure all sharer/receiver prefill and decode compute, fuser cost, memory, latency, and setup training;
- serialize the actual KV payload at a declared dtype and framing, count transfer bytes and synchronization, and report a separate co-located-device condition;
- plot quality against serialized bytes and full inference cost over context length, model pair, and reuse horizon, not token counts alone.

**Falsifiable prediction:** if the C2C gain comes from useful sharer-specific representations rather than additional encoder computation or training, true-context cache fusion should beat same-shape mismatched/neutral caches on a task where only the sharer sees decisive evidence. If the advantage vanishes after matched receiver-need controls, or its quality advantage disappears on an equal-serialized-byte/full-cost frontier, C2C does not establish a superior communication medium for that setting.

## Reproduction status

The published findings are treated as paper-reported evidence. Tacit has not reproduced them. The public code and paper were audited statically; the code has a local PyTorch integration but no measured remote transfer contract. Mark C2C **paper-supported, locally unreplicated, portable-byte comparison pending**. This is a deployment and measurement boundary, not a code defect finding.

## Sources

- Fu et al. (2026), [ICLR 2026 paper on arXiv](https://arxiv.org/abs/2510.03215) and [full HTML](https://arxiv.org/html/2510.03215).
- Authors' [pinned C2C implementation](https://github.com/thu-nics/C2C/tree/9fa85d35a20e20469af3e282926491222f4f11e0), especially the [Rosetta cache wrapper](https://github.com/thu-nics/C2C/blob/9fa85d35a20e20469af3e282926491222f4f11e0/rosetta/model/wrapper.py), [projector](https://github.com/thu-nics/C2C/blob/9fa85d35a20e20469af3e282926491222f4f11e0/rosetta/model/projector.py), [live-chat loading and transfer example](https://github.com/thu-nics/C2C/blob/9fa85d35a20e20469af3e282926491222f4f11e0/script/playground/live_chat_example.py), and [README](https://github.com/thu-nics/C2C/blob/9fa85d35a20e20469af3e282926491222f4f11e0/README.md).
