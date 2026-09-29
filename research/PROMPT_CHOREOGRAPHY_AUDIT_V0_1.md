# Prompt Choreography audit v0.1

**Reviewed:** 2026-09-29  
**Purpose:** decide how shared KV-cache execution belongs in Tacit's communication baselines. This is a source audit, not a replication.

## Source and method

Reviewed the full TACL 2026 paper, its reported tables and assumptions, and verified the paper's link to a reference implementation. The code itself was not audited or run. The primary source is Bai and Eisner, [*Accelerating Language Model Workflows with Prompt Choreography*](https://aclanthology.org/2026.tacl-1.13/) (TACL 14, 253–270, DOI [10.1162/TACL.a.643](https://doi.org/10.1162/TACL.a.643)); linked code: [tjbai/choreo](https://github.com/tjbai/choreo).

## What the system actually changes

Prompt Choreography maintains a dynamic global cache of message text and Transformer key/value encodings. Calls select, reorder, and mask cached messages; new messages can be generated in parallel from views of the same cache. This reuses *computation state* inside one model runtime. The logical text message still exists, and the method is not a compressed serialized network payload or a general cross-model language.

Its implementation assumes the cache fits in GPU memory, messages can be reused on the same device, workflows do not leave the GPU cache idle for long, and positions can be updated under relative encodings such as RoPE. The paper's experiments use a shared model/runtime and explicitly study accuracy changes and information leakage caused by altered context visibility. This is a different deployment contract from Tacit's loopback messages between independently addressed endpoints.

## Results and qualifications

- On the three main MATH workflows, the paper reports 2.0–6.2× lower per-step time-to-first-token, but only about 1.027–1.036× end-to-end speedup in those tested configurations; decode time dominates. Broader prefill-bound Tree-of-Thought configurations reach up to 2.2× end-to-end speedup.
- Choreography without adaptation often loses task accuracy. For Llama-3.1-8B, the reported MATH accuracy falls from 39.0 to 24.8 for iterative debate, 39.6 to 30.2 for Tree of Thoughts, and 64.6 to 52.4 for parallel debate. Lightweight fine-tuning largely restores performance in some workflows, but that training and its amortization horizon are additional costs. The Qwen3 appendix includes different outcomes by model size, further arguing against a universal claim.
- The paper identifies information blockage and indirect leakage across agent contexts as concrete failure modes. Cache reuse is therefore not automatically a semantics-preserving or privacy-preserving implementation optimization.

The reported speedups are evidence for a conditional inference-system optimization, not for fewer information bits or a more efficient message language. Do not quote the 2.2× maximum as the typical end-to-end result.

## Consequences for Tacit

1. Keep **logical representation**, **transported bytes**, and **inference-state reuse** as separate coordinates. A zero-new-text-token cache hit must not be recorded as a zero-byte wire message or a new language.
2. If a future study claims total compute/latency efficiency for agents sharing a compatible model runtime, include ordinary prefix caching and a Prompt-Choreography-style shared-cache baseline when technically feasible. Count cache memory, retrieval/masking work, any fine-tuning, output changes, and privacy boundary.
3. For independent or heterogeneous endpoints, mark shared-cache transfer as inapplicable unless both sides can safely consume the same compatible state and the transmitted serialized tensor is measured. Do not compare its in-process shared-memory path to network bytes as if the boundaries matched.
4. The predicted regime is falsifiable: savings should grow with the fraction of repeated prefill work and shrink in decode-bound workflows. Under equal generated outputs, TTFT may improve substantially without a comparable end-to-end latency or channel-byte gain. A benchmark that records only output-token counts cannot test this prediction.

## Decision

Add shared KV reuse to the conditional *system-level* baseline inventory, not to the first Private Match codec ranking. Private Match v0.2 sends a short semantic payload over separately measured loopback endpoints and does not presently implement a model-runtime shared cache. A shared-cache study needs a task/workflow with repeated prompt context, matched outputs, and a runtime that exposes KV state; it is not a reason to weaken current resource gates or launch a model on this host.

## References

- Bai, T. J. and J. Eisner (2026). [*Accelerating Language Model Workflows with Prompt Choreography*](https://aclanthology.org/2026.tacl-1.13/). TACL 14:253–270. [PDF](https://aclanthology.org/2026.tacl-1.13.pdf), [DOI](https://doi.org/10.1162/TACL.a.643), [implementation](https://github.com/tjbai/choreo).
