# TFlow audit v0.1

**Audit date:** 2026-09-29  
**Status:** paper and public inference-repository review; no code was installed or run.

## Why this matters

Bao et al., *Good Agentic Friends Do Not Just Give Verbal Advice: They Can Update Your Weights* (arXiv:2605.13839), introduce TFlow, a weight-space communication route. It strengthens the project's alternative-mechanism baseline: collaborators need not express exchanged content as a message that the receiver reads. A sender's hidden states condition a learned parameter generator, which emits receiver-specific LoRA perturbations applied transiently during generation.

Sources: [paper](https://arxiv.org/abs/2605.13839) · [official code](https://github.com/BWR-hhh/TFlow)

## What the work reports

- Experiments use three frozen Qwen3-4B agents; the parameter generator maps sender activations into LoRA factors for a known receiver architecture and target modules.
- Across five reasoning, knowledge, and coding benchmarks, TFlow reports gains of 7.13–8.53 accuracy points over its standalone receiver while consuming fewer processed tokens.
- Against its text-based three-agent comparator, the paper reports 71–83% fewer processed tokens and 2.3–4.6x lower wall-clock inference time, with an accuracy gap on four of five tasks and a larger gap on HumanEval+.
- The official repository publishes inference code and an approximately 123 MB parameter-generator checkpoint. Its README says the training code is not included; the checkpoint was trained on 32k samples from a mixed reasoning corpus.

These are results in the paper's evaluated setup, not evidence of a portable message language or a general cross-model protocol.

## Boundary and missing costs

TFlow changes the communication object and receiver computation together. The channel is an instance-specific tensor/weight perturbation, not merely hidden text. Token reductions therefore cannot establish lower communication bandwidth by themselves. A fair frontier needs the serialized LoRA factors (shape, dtype, framing, and transport boundary), sender activations if they cross a boundary, generator compute and storage, receiver patching/compute, and setup amortization. The paper's token and latency results answer useful systems questions but do not substitute for this encoding-neutral channel ledger.

The demonstrated method is receiver-specific and assumes access to sender hidden states and the receiver's forward pass; the backbone agents share Qwen3-4B. It is therefore not directly available to API-only agents, arbitrary heterogeneous model families, or independently hosted systems without compatible weights/runtime. These limitations make it a strong native-runtime alternative and a poor proxy for a model-agnostic wire protocol.

The paper also notes that TFlow takes longer than its single-agent receiver on four of five tasks because transient LoRA application impedes efficient batching, even though it is faster than the much more token-heavy TextMAS arm. Report both comparisons and total latency; do not summarize the result as unconditional speedup.

## Consequences for Tacit

1. Preserve distinct claims for **text/wire representation**, **latent or weight-space communication**, and **complete collaboration system**. A reduction in model-token count belongs to inference accounting, not automatically to channel bytes.
2. Keep TFlow as a conditional systems comparator when compatible open checkpoints, sufficient local GPU memory, and a measurable serialized boundary are available. Do not download its backbone/checkpoint or run it under the current idle-resource gate.
3. For any shared-model or weight-space condition, compare true instance-specific sender state with a same-shape mismatched-instance perturbation, neutral perturbation, and no-message arm. This tests whether the transferred object carries task-relevant information rather than merely changing receiver behavior.
4. If no feasible weight-space run is available, state the access/compute constraint and avoid claiming superiority over the strongest latent/weight-space alternatives. A discrete protocol may still have value under heterogeneous, API-only, auditability, or low-setup deployment conditions, but that is a conditional hypothesis.

## Falsifiable prediction

At matched task quality, TFlow may dominate text on receiver-prefill and full inference compute in a fixed compatible-backbone setting, while losing or becoming incomparable on portable serialized bytes and setup/reuse cost. If actual tensor transport, generator cost, and setup amortization erase its advantage, token-only savings overstate operational efficiency. In a heterogeneous receiver condition, a model-agnostic text protocol may retain transfer where receiver-specific weight perturbations do not. These predictions should be tested separately; they do not imply that text is universally preferable.
