# DiffMAS source audit v0.1

**Date:** 2026-09-29
**Paper:** Yu et al., [*Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems*](https://arxiv.org/abs/2604.21794), arXiv v1, under review at COLM 2026.
**Scope:** primary paper and public code search only. No model weights, datasets, or implementations were downloaded or run.

## Why it is a strong baseline

DiffMAS is a recent trained KV-trace communication system. A sequential Planner–Critic–Refiner–Solver pipeline appends latent KV segments rather than overwriting a fixed-size carrier. The final agent decodes conditioned on the accumulated trace. The paper describes SFT with task-specific LoRA and reports gains over Single, TextMAS, training-free LatentMAS, and C2C across math/science, coding, and commonsense tasks. For example, Qwen3-8B AIME24 is reported at 76.7% versus 50.0% Single and 50.0% TextMAS; Qwen3-14B HumanEval+ is 87.7% versus 81.5% TextMAS.

This is a serious upper-bound comparator when agents share compatible internals and can be task-adapted. It makes communication jointly learnable with downstream task behavior, beyond fixed KV transfer or hand-written latent adapters.

## Scope and reproduction cautions

- All four roles operate on the same task instance. The system passes the plan and critique as latent state; the Refiner also receives the original question. The benchmark does not test a sender-only fact the receiver cannot see, so its gains do not establish communication necessity under private evidence.
- Training is task-specific: the paper uses 210 Hendrycks MATH examples, 50 HumanEval examples, and 700 CommonsenseQA examples with synthetic traces for its task families. It evaluates on AIME, GPQA, HumanEval+, MBPP+, and OpenBookQA, excluding stated training examples where applicable. Math-family overlap and exact split/hash hygiene should be audited in replication.
- The AIME tables use increments of about 3.3 percentage points, consistent with 30 questions. Headline changes therefore rest on a small test set and are reported without uncertainty intervals in those tables. Replication should use paired item-level outcomes and interval estimates before treating the deltas as stable.
- The paper reports training on A40 GPUs for Qwen3-4B/8B and Ministral3-8B, and H200 GPUs for Qwen3-14B and DeepSeek-R1-Distill-Qwen-32B. This is outside Tacit's current local inference/training gate.
- There is an implementation-description ambiguity to resolve before replication: the Figure 1 caption says only the final agent's LoRA parameters are updated, whereas the method and training appendix describe gradients propagating through the latent trace to adapt how upstream agents encode and downstream agents interpret it. Verify the actual trainable parameter set and gradient path from author code or checkpoints before describing it as joint sender–receiver learning.
- A search found the paper's arXiv landing page but no linked author implementation. A public repository surfaced by search is an arXiv-to-code scaffold, not verified author code. Treat the method as paper-reported and not independently reproduced here.

## Theory and bandwidth boundary

The paper's proposition compares a fixed-dimensional overwriting state under an assumed contractive Jacobian with a concatenated trace. In the concatenated case, the norm of a block's loss gradient is bounded by the norm of the full-trace gradient; this follows from coordinate projection. The no-depth-factor statement is an interface-level observation. It does not by itself prove stronger end-to-end gradients in a particular trained transformer, higher task success, lower compute, or lower communication cost. The contraction assumption is also needed for its overwriting comparison.

The trace grows with the number of stages: if each of K agents appends T blocks of dimension d, then its scalar count is KTd and raw bytes are at least KTd times bytes per scalar, before metadata and transport. Increasing the preserved trace can increase downstream context processing and memory. More preserved state may improve task accuracy while worsening the bandwidth/compute frontier; both sides must be measured at equal serialized budgets.

## Falsifiable Tacit follow-ups

1. Use a receiver-need task with disjoint private records to test true, other-episode, zero/neutral, and no-message KV traces. Establish receiver-alone capability controls first.
2. At a fixed schedule and held-out task family, compare DiffMAS to optimized text, structured messages, and training-free latent transfer over matched serialized-byte budgets. Charge LoRA training, adapter distribution, both endpoints' inference, KV bytes, receiver cache memory, and wall-clock latency.
3. Ablate each sender trace and permute or replace it with a matched trace. If each role contributes transferable information, sender ablations should lower receiver task utility in a role-specific and episode-sensitive way.
4. Compare fixed trace capacity against unbounded concatenation as K and T grow. Prediction: without compression, raw payload and receiver attention context grow at least linearly in K*T; equal-quality bandwidth advantage is not implied by the paper's gradient proposition.
5. Resolve and record which LoRA parameters receive gradients, whether upstream hidden states are recomputed through the differentiable graph, how truncation/cache detach works, and the train/test split hashes.

## Project decision

Keep DiffMAS in the conditional learned-latent baseline family. Its benchmark results make it necessary to challenge text-format claims under a trained, compatible-runtime setting, but the paper does not settle private-evidence necessity, portable bandwidth, or generalization across arbitrary agents. Do not reproduce locally while the frozen resource gate blocks model work.

## Sources

- [arXiv abstract and metadata](https://arxiv.org/abs/2604.21794)
- [arXiv full text](https://arxiv.org/html/2604.21794v1)
- [Search result: public arXiv-to-code scaffold](https://github.com/Arxiv-to-code/arxiv-260421794-learning-to-communicate-toward-end-to-end-optimization-of-mu) (not verified as author code)
