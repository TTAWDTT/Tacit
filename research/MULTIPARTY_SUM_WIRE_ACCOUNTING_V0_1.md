# Multi-sender sum wire-accounting diagnostic v0.1

**Status:** deterministic, model-free serialization experiment. It measures only the current SDK text channel; it does not rank protocols by task quality or total LLM cost.

## Question and method

For the five message strings in [`multiparty_private_sum.py`](../examples/multiparty_private_sum.py), how much does a message cost after it crosses the current `LocalTCPMessageChannel` serialization boundary, and how does that cost grow from 2 to 11 senders? Eleven senders plus one receiver call is the example's 12-call ceiling.

The calculator calls `LocalTCPMessageChannel.measure`, not `send`: it opens no socket and contacts no model. Each sender holds a value cycling through 0, 1, 2, 3. It records logical UTF-8 payload bytes, the JSON-serialized payload field, framing/acknowledgment bytes, and total application-layer bytes. It includes two identifier policies: the example's format-specific protocol ID and a common ID shared by all formats, so the second is the controlled representation comparison.

Exact per-message records and all agent-count cells are in the [machine-readable ledger](data/MULTIPARTY_SUM_WIRE_ACCOUNTING_V0_1.json). Recompute from the repository root with:

```powershell
python research/multiparty_sum_wire_accounting.py
```

The ledger includes the SHA-256 of the example source used for measurement. The channel counts a length-prefixed UTF-8 JSON envelope and one-byte acknowledgment; TCP/IP and link-layer headers are excluded.

## Results

Application bytes for `m=2` and `m=11` senders:

| Format | Example IDs, m=2 | Common ID, m=2 | Example IDs, m=11 | Common ID, m=11 |
|---|---:|---:|---:|---:|
| Decimal | 250 | 248 | 1,379 | 1,368 |
| JSON | 268 | 272 | 1,478 | 1,500 |
| Labeled `v=N` | 254 | 252 | 1,401 | 1,390 |
| Base-2 text (`00`–`11`) | 250 | 250 | 1,379 | 1,379 |
| Fixed sentence | 298 | 294 | 1,643 | 1,621 |

With a shared protocol ID, base-2 text costs exactly one extra serialized application byte per sender versus decimal: 2 extra bytes at `m=2` and 11 at `m=11`. The shipped IDs differ in length (`binary` is one character shorter than `decimal`); those repeated header bytes exactly cancel that payload difference, making the two shipped totals tie. Identifier choice is therefore itself a measured setup/framing variable, and should be normalized or charged explicitly in a format comparison.

At `m=11`, the common-ID decimal cell carries 33 serialized payload bytes and 1,335 framing/acknowledgment bytes, for 1,368 total. Its framing accounts for 97.6% of this application-level measure. This is a property of the current per-message loopback envelope and acknowledgment, not a physical-network estimate or a universal protocol overhead. JSON costs 132 bytes more than decimal in this cell; the fixed sentence costs 253 more. These are transport-size differences only.

## Interpretation and limits

- The format's apparent payload compactness does not translate directly into wire savings when per-message routing/framing dominates. For this one-digit task, plain decimal is smaller than its two-character base-2 text rendering after normalizing IDs.
- This does **not** compare model adherence, task success, input/output tokens, full prompts, codebook distribution, inference compute, latency, or an actual raw-bit codec. System-prompt instructions for the formats are deliberately excluded, and each condition's learned/fixed decoder setup is not charged.
- The lower bound is `2m` ideal payload bits for exact summation over `{0,1,2,3}`. The measured application bytes are not directly comparable to that payload-only theorem without also specifying the transport boundary and setup costs.
- No superiority claim follows. A fair end-to-end study must pair model outcomes and receiver fidelity with complete provider usage and this transport accounting, then sweep message budgets. Larger messages and different framing policies may change the relative cost.

## Next falsifiable step

If future tasks carry multi-field meanings, the payload-saving candidate should cross over only when its per-message serialized payload reduction exceeds any extra decoder/setup and token cost at the declared reuse horizon. Measure that point with the exact task's frozen prompts/tokenizer and the same end-to-end wire boundary; do not extrapolate it from this scalar calibration.
