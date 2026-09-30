# Multi-party private sum, frozen IID task bundle (v0.1)

This bundle freezes a narrow calibration task for the existing private-sum SDK example. Each of `m` senders independently receives an integer uniformly sampled from `{0,1,2,3}`; the receiver must return the exact sum. No single role sees the complete input vector. The generator creates keyed, fixed tuples and separate sender, receiver, and evaluator ledgers. It makes no model calls.

## Research question

Does protocol choice change end-task exact-sum success or communication cost when sender count and input entropy grow, once message decoding and arithmetic are separated?

This task is deliberately only a calibration instrument. It cannot establish general-purpose language superiority: the sufficient statistic is the input vector, the task has a trivial exact fixed-width code, and arithmetic may dominate receiver errors.

## Preregistered conditions and outcomes

Within each sender count, run every condition on exactly the same episode IDs and values:

- `no_message`: receiver sees only public metadata and the registered prior. The theory reference is the exact mode probability computed in `research/multiparty_sum_scaling.py`, not a sampled estimate.
- `full_information`: receiver receives all values; this measures receiver arithmetic and answer-format capability.
- `communicate`: each sender independently emits one message to the receiver. Initial protocol conditions should include decimal, JSON, labeled decimal, two-character binary text, and fixed-sentence English from `examples/multiparty_private_sum.py`; all transport and model-call costs must be counted.
- `oracle_sender` / `oracle_receiver`: scorer-side ablations may bypass one component to separate sender encoding/faithfulness from receiver decoding/arithmetic. These are diagnostic upper bounds and are not language competitors.

Primary outcome: strict exact-sum success. Report per-episode generated and delivered application bytes, full transport bytes, sender syntax validity, value fidelity, delivered-message fidelity, model calls, latency, and inference usage where available. Show success-cost frontiers across byte budgets; do not infer superiority from accuracy alone. Cluster uncertainty by independently generated episode tuple, and keep each protocol paired on tuple within `m`. Different `m` values are different task sizes and are not paired observations.

The central prediction from the exact coding reference is that zero-error communication over a noiseless fixed-width binary channel requires and attains `2m` payload bits. In particular, if another condition claims lower cost, verify whether its cost includes framing, identifiers, tokenizer charges, and model computation, and whether it is lossless on all inputs or only successful on this finite sample. A second prediction is that the no-message Bayes reference falls with `m` according to the sum-distribution mode; it must be computed exactly, not estimated from the finite bundle.

## Frozen limits

The generated range is `m=2..11`, with `m+1 <= 12` planned calls for one call per sender plus one receiver call. The existing global local-resource preflight remains binding: CPU mean `<20%`, every sample `<30%`, GPU utilization `<25%`, GPU memory `<1800 MiB`, free RAM `>=6000 MiB`, and ports 8000/8001/8002 idle. A rejected or stale preflight means no inference; do not relax the thresholds. The latest recorded GPU sample was over the limit, so this artifact work does not authorize model execution.

## Generate and validate

Use a secret evaluator key stored only under the ignored project-local `.cache/` directory. Never commit or place it in prompts.

```powershell
python experiments/multiparty_sum_v0_1/generate_tasks.py `
  --task-key-file .cache/multiparty_sum_v0_1/evaluator.key --create-task-key
python experiments/multiparty_sum_v0_1/generate_tasks.py `
  --task-key-file .cache/multiparty_sum_v0_1/evaluator.key `
  --output .cache/multiparty_sum_v0_1/bundle --task-seed 0 --episodes-per-count 32
```

Every sender file contains only that sender's private value; the receiver file contains only the episode ID, sender count, role, and prior; `gold_mNN.jsonl` is evaluator-only. Strip `episode_id` before constructing a model prompt. The manifest binds all role files by SHA-256 and documents the task seed and key ID, but not the secret key. HMAC-SHA256 streams are domain-separated and use rejection sampling for unbiased values; the public seed alone is insufficient to reconstruct the tuple. Generation refuses overwrite by default. Keep the key and bundle under `.cache/` during development.

The bundle is not yet a public benchmark result. Before any model comparison, preregister the protocol cards, episode count/power rationale, statistical analysis, transport accounting, and a passing fresh resource preflight; first pass an independent full-information capability screen under the frozen call cap.
