# Raw KV-state transfer size estimate v0.1

**Status:** model-free architecture calculation; not a benchmark or a measured wire trace.  
**Question:** what bandwidth tradeoff appears if a shared-runtime KV cache is turned into a serialized cross-process message?

## Derivation

For a conventional decoder-only Transformer, a cached token has one key and one value vector per layer and KV head. For `L` layers, `H_kv` KV heads, head dimension `d`, and `b` bytes per element, the dense raw payload per token is

`B_token = 2 * L * H_kv * d * b`.

The official [Qwen3-1.7B configuration](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json) specifies 28 layers, 8 KV heads, head dimension 128, and BF16 model dtype. Assuming the runtime stores each K/V element in BF16 (2 bytes/element),

`B_token = 2 * 28 * 8 * 128 * 2 = 114,688 bytes = 112 KiB`.

The payload scales linearly with the number of message tokens:

| Tokens represented | BF16 raw K/V payload | Ideal INT8 payload* |
|---:|---:|---:|
| 1 | 112 KiB | 56 KiB |
| 8 | 896 KiB | 448 KiB |
| 16 | 1.75 MiB | 0.875 MiB |
| 32 | 3.5 MiB | 1.75 MiB |
| 64 | 7 MiB | 3.5 MiB |
| 128 | 14 MiB | 7 MiB |
| 4,096 | 448 MiB | 224 MiB |
| 8,192 | 896 MiB | 448 MiB |

\*The INT8 column is only the element-count lower estimate at one byte per element. It excludes quantization scales, block metadata, padding, serialization headers, alignment, and any accuracy/reconstruction loss. It is not a tested codec.

## What this does and does not show

- This is the raw dense tensor volume for these architectural dimensions and the stated dtype assumption, not measured network bytes, allocated runtime memory, or a prediction for an engine that prunes layers, compresses values, evicts context, uses a different cache dtype, or changes cache layout.
- A real cross-process message also needs serialization metadata and compatible model/tokenizer/config identity. If the receiver does not share the exact model state and interpretation, raw KV tensors may be unusable or change behavior. If encodings depend on parent context, transmitting only the new message's token span may not reconstruct the original workflow's context semantics.
- Prompt Choreography keeps the cache on the same device as the model; its shared-memory path therefore avoids transferring these tensors over a network. The estimate applies only to an explicitly serialized state-transfer baseline, not to its normal in-process execution.
- The estimate makes the representation/transport tradeoff concrete: a compact text message can be far smaller in wire bytes, while a receiver with compatible internal state may avoid re-encoding work. Neither axis predicts task success, semantic fidelity, total latency, or energy. Measure those independently.

## Falsifiable use in future comparisons

If Tacit adds a Qwen3-1.7B KV-transfer arm, compare the measured serializer output against the raw-size estimate after freezing cache dtype, token count, layers, and tensor shape. Record actual transmitted bytes including framing and scale metadata, receiver reconstruction/application, task success, setup/calibration, and compute. A raw dense BF16 implementation should be close to `114,688 * token_count` bytes before container overhead; a materially smaller result must identify the compression or sparsity mechanism and validate receiver fidelity. Do not substitute this calculation for a runtime trace.

## References

- Qwen Team. [Qwen3-1.7B `config.json`](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json), accessed 2026-09-29. Architecture fields used: `num_hidden_layers`, `num_key_value_heads`, `head_dim`, and `torch_dtype`.
- Bai & Eisner (2026). [Prompt Choreography](https://aclanthology.org/2026.tacl-1.13/). Its standard shared-cache execution keeps K/V state on the same device; see the [separate source audit](PROMPT_CHOREOGRAPHY_AUDIT_V0_1.md).
