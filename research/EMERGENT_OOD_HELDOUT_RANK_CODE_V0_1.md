# Held-out modular-rank code: exact fixed-width oracle bound

**Status:** model-free formalization and reference codec. This is a task-support oracle control, not a proposed general LLM language, a model result, or a claim of communication superiority.

## Question

The v0.4 split holds out a modular slice of a categorical product. The existing `ceil(d log2 V)` fixed-width bound encodes any tuple in the full `V^d` universe. Does the known held-out test support itself admit a smaller exact code, and does that bit reduction survive serialization?

## Result

For `d` axes with `V` values each, the public held-out support contains exactly `V^(d-1)` tuples. If a target is guaranteed to be in that support and sender and receiver share the split and decoder, a fixed-width zero-error code has the exact lower bound:

```text
bits >= ceil(log2(V^(d-1))) = ceil((d - 1) log2(V))
```

This is attained without a lookup table. Rank the values on each axis using the split's public per-axis permutation. Encode the first `d-1` ranks as a base-`V` integer. The last rank is uniquely fixed by the requirement that all ranks sum to zero modulo `V`. This is a bijection over the held-out support.

For the default `d=4, V=4` fixture, the held-out support has 64 elements. Its support-aware floor is 6 bits versus 8 bits for an arbitrary full-universe tuple. Both serialize to one byte, before framing, so the oracle predicts no reduction in application wire bytes in this default setting. For non-power-of-two support sizes, the codec rejects unused fixed-width codewords and nonzero canonical padding bits.

The scaling report now lists ideal bit savings and byte-rounded payload savings separately. For example, with `d=4, V=5`, the full-universe rank needs 10 bits (2 bytes) while the held-out rank needs 7 bits (1 byte); this saves one payload byte before framing. The same one-byte saving occurs for `(d,V)=(5,4)` and `(6,3)`. Such a saving is still only a payload comparison: protocol framing and sharing/setup cost can erase it.

## Limits and accounting

- This code is valid only for tasks whose target is known to be in the held-out support. It is not a code for training targets or arbitrary tuples.
- The split schema, ontology, permutation, and decoder are shared side information. Their distribution/setup cost is excluded from the payload lower bound and must be accounted for, amortized over the number of messages if stateful.
- Framing, serialized bytes, provider tokenization, model instructions, inference compute, latency, and receiver errors are not included in the bit bound.
- The bound concerns fixed-width, worst-case zero-error payloads. A variable-length code with a nonuniform target prior has a different expected-length question.
- The codec being decodable by deterministic software says nothing about whether an LLM can interpret the payload. Treat it as an oracle comparator only.

## Falsifiable prediction

P16 in [Theory §17](../docs/THEORY.md) predicts that exhaustive encode/decode is bijective on every generated held-out support and rejects out-of-support targets, noncanonical padding, and unused fixed-width codewords. It also predicts an ideal saving of exactly `ceil(d log2 V) - ceil((d-1) log2 V)` bits over the corresponding full-universe fixed-width code. It predicts a serialized saving only when byte rounding (and then framing/setup) actually decreases. Any violation falsifies the implementation or a stated split assumption.

## Verification

The standard-library test `tests.test_emergent_ood_v04_heldout_rank_codec` exhaustively enumerates the default 64-element support, checks a non-power-of-two support, rejects train meanings and malformed payloads, and varies split seeds. No model, inference endpoint, GPU operation, or external data is needed.
