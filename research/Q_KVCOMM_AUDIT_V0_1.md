# Q-KVComm source audit v0.1

**Date:** 2026-09-29
**Paper:** Kriuk & Ng, [*Q-KVComm: Efficient Multi-Agent Communication Via Adaptive KV Cache Compression*](https://arxiv.org/abs/2512.17914), arXiv v1.
**Scope:** primary paper and search for an author implementation. No model weights, datasets, code, or services were downloaded or run.

## Method described

Q-KVComm proposes four steps around a KV transfer: select 70% of layers using a hybrid attention score/Gaussian depth prior, extract supplemental facts using YAKE/NER/domain patterns, quantize selected KV tensors with 4/6/8-bit mixed precision, and calibrate sender tensors to receiver statistics before concatenating them with the receiver cache. It evaluates TinyLlama-1.1B and Qwen2.5-1.5B on SQuAD, HotpotQA, and NarrativeQA.

This is a relevant system-level compression proposal. It combines latent state with a text facts channel, so it should be compared as a hybrid cache-plus-text representation, not as a pure latent protocol.

## Evidence gaps and technical questions

1. **Reported quality is not task success.** The paper defines contextual relevance, answer completeness, semantic fidelity, coherence, compression, and throughput, but Table I reports only contextual relevance, coherence, compression ratio, and MB saved. It gives no exact QA accuracy/F1 or answer-completeness results, uncertainty, item count, or per-example ledger. The paper's reported coherence score (0.777–0.957) is described as based on response length and structure, so it cannot alone establish semantic fidelity or correct task answers.
2. **The listed baselines are not numerically compared in the presented results table.** Full text, uncompressed KV, and uniform quantization are named as baselines, but Table I shows only Q-KVComm at nominal 4/6/8-bit settings. Thus the table does not substantiate a quality/latency advantage over those baselines.
3. **The cross-architecture calibration is under-specified for tensor shapes.** The method maps per-coordinate means and standard deviations, then suggests scalar statistics for differing hidden dimensions. Elementwise centering/scaling does not map one vector dimension, head layout, or layer count into another. Yet the receiver integration concatenates the calibrated cache into attention. The actual shape-compatible mapping, layer correspondence, and tested sender/receiver pairs are not specified in the experiment table. Do not treat heterogeneous compatibility as verified without the exact implementation and shape-level demonstration.
4. **Compression denominator is unclear.** The paper reports 5.06–6.93× compression with 70% layer selection and 4–8-bit quantization, but the reported table does not state the original cache dtype, whether unselected layers are omitted, tensor/scale/zero-point metadata treatment, or whether the additional fact-summary text is included in compressed payload bytes. This prevents interpreting the ratio as a portable byte saving, especially against BF16/FP16 KV caches.
5. **No official implementation is linked from the arXiv page.** Search did not identify a verified author repository. The paper also lacks a detailed train/test split, input length distribution, exact receiver cache integration pseudocode, or released per-example outcomes in the source inspected.
6. **Communication necessity remains untested.** The QA datasets can involve external context, but the paper does not define a matched private-evidence partition where only the sender sees a decisive fact and compare true versus mismatched/neutral/no-message cache. It therefore does not establish that the cache/fact channel causally transfers receiver-unavailable information.

## Reproduction and comparison requirements

- Pin actual sender/receiver model revisions, tokenizer, cache dtype, layer/head shapes, context lengths, selected-layer map, quantizer settings, and calibration data.
- Publish application-layer serialized byte counts for both KV and extracted facts, including all headers, scales, zero points, shape metadata, and required calibration state. State clearly whether setup and shared dictionaries are charged.
- Report paired exact task outcomes and established dataset metrics (for example, SQuAD EM/F1 and task-appropriate QA metrics), not only a heuristic coherence score. Include uncertainty intervals and item-level outputs.
- Compare the same cache/fact content against full text, compressed text, full-precision selected KV, uniform quantization, and no-message; fix prompts and generation budgets.
- On receiver-need examples, include true-instance cache, other-instance cache, zero/neutral cache, and no-message. Test shape-compatible same-model transfer separately from heterogeneous transfer.
- Measure extraction, sensitivity profiling, quantization, serialization, transport, dequantization, receiver prefill/decode, memory, and end-to-end latency. A compression ratio alone is not an efficiency frontier.

## Project decision

Keep Q-KVComm as a **candidate, not an established strong baseline result** until task utility, baseline comparisons, denominator, tensor compatibility, and implementation become verifiable. Its idea belongs in the search space, but present evidence does not support ranking it above text or other cache methods. Do not implement from the paper's ambiguous heterogeneous mapping or run locally under the frozen model-resource gate.

## Sources

- [arXiv abstract and metadata](https://arxiv.org/abs/2512.17914)
- [arXiv full text](https://arxiv.org/html/2512.17914v1)
