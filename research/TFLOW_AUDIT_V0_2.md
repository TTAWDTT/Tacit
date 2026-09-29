# TFlow audit v0.2: token accounting and receiver-need boundary

**Status:** static review of Bao et al. (2026) and the public implementation pinned at `7aef170594336c6bda2f8baa1ac51f503f6096ce` (shallow clone under this project's ignored `.cache/research/`). No source was executed, and no model, checkpoint, or dataset was downloaded or evaluated. This supplements [v0.1](TFLOW_AUDIT_V0_1.md); it does not replace that survey or reproduce/refute the paper's accuracy and wall-time results.

## The transmitted object

TFlow does not send a sentence, token sequence, or receiver-readable latent. It runs two role-prompted senders on the same question, extracts their question-token hidden states across all layers, aggregates layers with learned weights, and feeds those activations to a trained parameter generator. The generator emits rank-4 LoRA factors for a known Qwen3-4B receiver; sender factors are fused, temporarily patched into the receiver's linear layers, and removed after its answer.

This is an instance-conditioned control channel: sender-side state changes the receiver's computation. It is not a general language. The paper formalizes a fixed, known receiver architecture; the official runtime creates three role wrappers over one shared Qwen3-4B model instance. A different receiver needs a compatible architecture mapping and parameter-generator checkpoint/training. The method transfers no generated role plan or private evidence. Since every agent receives the same task question, these benchmarks test useful role-conditioned computation, not whether information unavailable to the receiver can be communicated.

## Published trade-off

The paper evaluates GSM8K, MATH, MMLU-Redux, MBPP+, and HumanEval+ with one Qwen3-4B backbone. Against its three-agent TextMAS condition, TFlow reports 71–83% fewer processed tokens and 2.3–4.6x lower wall time, but accuracy is lower on every benchmark: gaps range from 1.32 points on MBPP+ to 9.76 on HumanEval+. Against a single agent, TFlow improves accuracy by 7.13–8.53 points but has higher wall-clock latency on four of five tasks because transient LoRA injection impairs efficient batching. This is a concrete quality/latency/token trade-off, not a win on every axis.

The paper says the parameter generator is trained on 32,000 examples for one epoch on an RTX PRO 6000 in about eight hours. The released repository is inference-only and provides an approximately 123 MB generator checkpoint; it omits the training implementation. The setup is therefore promising as a released inference artifact, but its training recipe is not end-to-end reproducible from that repository alone.

## Token-accounting defect in the public evaluator

In pinned [`TFlowMethod.solve`](https://github.com/BWR-hhh/TFlow/blob/7aef170594336c6bda2f8baa1ac51f503f6096ce/src/methods/tflow.py), each sender calls `_per_sender_condition`, which invokes `extract_hidden_states_all_layers` on the complete role prompt and question before slicing question positions. The method then returns only `token_usage` from the final receiver's `generate_with_truncation_info`. The batched TFlow path likewise reports the receiver's usage only.

By comparison, pinned [`TextMASMethod.solve_batch`](https://github.com/BWR-hhh/TFlow/blob/7aef170594336c6bda2f8baa1ac51f503f6096ce/src/methods/textmas.py) explicitly sums prompt and completion tokens from both senders and the receiver. The pinned [`Evaluator`](https://github.com/BWR-hhh/TFlow/blob/7aef170594336c6bda2f8baa1ac51f503f6096ce/src/evaluation/evaluator.py) reads `metadata.token_usage` as its token metric. Therefore the official code's TFlow token metric omits the full prompt-token inputs processed by its two sender forwards, while TextMAS counts sender inputs and outputs. The published token-reduction percentages should not be treated as a matched total-compute-token result until recomputed with sender prompt processing charged. This finding does not invalidate the paper's task accuracy or reported wall-clock measurements; it identifies a mismatch in the released token-accounting path.

The same issue matters for system accounting beyond tokens: all three roles share a model instance, and the reference implementation constructs and applies the LoRA factors in-process. It reports wall time but does not serialize a cross-process or network communication payload. A portable deployment must either transmit the sender conditions to the parameter generator or transmit the generated LoRA factors to the receiver, then count binary bytes, framing, dtype, synchronization, generator cost, and checkpoint distribution. The paper's in-process path is a valid co-located systems condition, not evidence of zero-byte remote communication.

## What Tacit should test

Keep TFlow in the **weight-space / co-located shared-backbone** stratum, separate from black-box text protocols and cross-model latent transfer. A faithful comparison should:

- correct total token accounting by including each sender's full prompt processing and receiver input/output;
- compare at matched task quality and separately report total tokens, FLOPs or measured compute, memory, latency, and serialized payload;
- use receiver-need tasks where senders possess private evidence or partial traces that the receiver does not see, so success depends on a transmitted contribution;
- include true-query, other-query, zero-condition, static-LoRA, no-communication, and matched text-message conditions;
- charge parameter-generator training/checkpoint distribution and compare an independent reimplementation or released training recipe;
- keep fixed-backbone same-model results separate from heterogeneous-model transfer and from external API-only agents.

**Falsifiable prediction:** TFlow's main advantage should come from avoiding autoregressive sender text and receiver prefill. When sender forward-pass tokens and generator cost are charged, its processed-token edge should shrink; when accuracy is matched rather than merely reported at a fixed point, the remaining wall-time advantage may also shrink. If the same-question hidden signal does not outperform a matched static/zero or wrong-question condition on a task where only the sender sees decisive evidence, the communication interpretation is falsified for that setting.

## Reproduction status

The published results are treated as reported evidence; Tacit has not reproduced them. Static source audit found that the official evaluator's TFlow token metadata excludes sender prompt processing and that the released repository omits generator training code. Mark the official artifact **paper-reported, local reproduction pending, token totals require correction**. Do not infer that the accuracy or wall-clock results are false.

## Sources

- Bao et al. (2026), [arXiv paper](https://arxiv.org/abs/2605.13839) and [HTML version](https://arxiv.org/html/2605.13839).
- Authors' [official repository](https://github.com/BWR-hhh/TFlow/tree/7aef170594336c6bda2f8baa1ac51f503f6096ce), especially the [TFlow method](https://github.com/BWR-hhh/TFlow/blob/7aef170594336c6bda2f8baa1ac51f503f6096ce/src/methods/tflow.py), [TextMAS token aggregation](https://github.com/BWR-hhh/TFlow/blob/7aef170594336c6bda2f8baa1ac51f503f6096ce/src/methods/textmas.py), [agent hidden-state extraction](https://github.com/BWR-hhh/TFlow/blob/7aef170594336c6bda2f8baa1ac51f503f6096ce/src/models/agent.py), and [evaluator](https://github.com/BWR-hhh/TFlow/blob/7aef170594336c6bda2f8baa1ac51f503f6096ce/src/evaluation/evaluator.py).
