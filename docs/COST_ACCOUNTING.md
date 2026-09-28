# Communication and inference cost accounting

**Status:** Measurement contract for future protocol experiments. Do not mix channel bandwidth with model inference work or combine them into a scalar unless prices and weights are preregistered.

## Keep three budgets distinct

### 1. Channel budget

Count what crosses the agent boundary: serialized payload bytes, framing/metadata bytes, number of transmissions, and (where relevant) each recipient's tokenizer count for the received payload. For broadcast, report both transmitted bytes and receiver-delivery bytes; a system that physically transmits one shared message differs from one that copies it into each receiver context. Charge any on-channel schema, vocabulary, codebook, or decoder description. State what shared side information is assumed preinstalled.

The communication-constrained frontier uses channel budget (B_{channel}), e.g. UTF-8 bytes or a fixed receiver-tokenizer budget. These units are not interchangeable; report each separately. Never infer that fewer generated tokens means lower channel use without counting actual delivered payloads.

### 2. Model inference cost

For every model call, record the complete API/runtime prompt tokens and generated tokens, including task context, system/developer instructions, protocol instructions, shared history, agent messages, answer generation, retries, and failed/truncated calls. These counts measure inference work; they are not all new communication over the agent boundary. Also record call count, per-call service latency, and the number of generated and received tokens attributable specifically to communication payloads.

Use native tokenizer counts for each model and the serialized bytes for cross-model comparison. For a message received by a heterogeneous model, count it under that receiver's tokenizer as part of its prompt, and separately as channel bytes.

### 3. Runtime and setup cost

Record end-to-end wall latency, critical-path latency when calls run concurrently, summed model-service time, tool/parser/decoder time where observable, retries and repair turns, and local process CPU/GPU time or sampled use when available. Report host and device memory peaks. Distinguish process-level measures from total-machine samples; background activity can contaminate the latter.

If a method requires protocol search, codebook construction, decoder training, calibration, or negotiation, report those costs separately and amortize them only over a declared reuse horizon (H). State whether the setup artifact is shared for free, sent once, or repeated in each prompt. For hosted models, report billed input and output units/cost separately when available.

## Per-episode record

Every run record should be sufficient to reconstruct the aggregates and frontier point. Minimum fields:

```json
{
  "episode_id": "task-seed-condition",
  "protocol": {"policy_id": "...", "code_id": "...", "decoder_id": "..."},
  "outcome": {"joint_success": false, "answer_score": 0.0},
  "transmissions": [
    {"round": 1, "sender": "A", "recipients": ["B"], "payload_utf8_bytes": 0,
     "sender_tokenizer": "...", "sender_tokens": 0, "recipient_tokens": {"B": 0}}
  ],
  "model_calls": [
    {"agent": "A", "stage": "communicate", "input_tokens": 0, "output_tokens": 0,
     "service_seconds": 0.0, "retry": false, "truncated": false}
  ],
  "runtime": {"wall_seconds": 0.0, "tool_seconds": null, "process_cpu_seconds": null,
              "process_gpu_seconds": null, "peak_rss_bytes": null, "peak_vram_bytes": null},
  "setup": {"one_time_bytes": 0, "one_time_tokens": 0, "reuse_horizon": 1}
}
```

Unknown measures must be `null` or omitted with an explicit reason; do not fill them with zero. Preserve raw payloads privately as appropriate, with a sanitized aggregate suitable for public release.

## Comparisons to publish

1. **Quality versus channel budget:** success and semantic fidelity at equal delivered bytes and, separately, equal receiver-tokenizer payload tokens.
2. **Quality versus inference cost:** task quality versus total input tokens, total generated tokens, model calls, service time, and local compute measures.
3. **Quality versus runtime/system cost:** wall latency and memory/compute where measurable.
4. **Matched-quality cost:** interpolation only when observed points support it; report uncertainty and do not extrapolate beyond tested ranges.
5. **Setup horizon:** per-episode and amortized cost at the preregistered reuse horizon, including decoder and repair costs.

Keep a vector-valued Pareto frontier unless a deployment scenario supplies defensible exchange rates among bytes, latency, compute, and quality. An intervention may reduce communication bandwidth while increasing prompt tokens or decoding compute; the record should make that tradeoff visible.

