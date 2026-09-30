# Transport costs for the optimal finite sum codes (v0.1)

**Status:** model-free application-byte accounting for the exact finite lossy oracle; no model runs, no sockets, and no protocol-superiority claim.

## Purpose

The [finite lossy-sum frontier](MULTIPARTY_SUM_LOSSY_FRONTIER_V0_1.md) is measured in ideal payload bits. Here we connect its actual optimal codebooks to Tacit's serializers so the result can be read in both information and transport units. The question is whether an ideal payload saving survives envelope framing, byte packing, message identifiers, and the choice to skip senders assigned zero bits.

## Protocol and measurement boundary

For each frontier point with `m≤4`, every sender receives the corresponding optimal partition of `{0,1,2,3}`. A value maps to its partition-cell index. Two equivalent payload renderings are measured:

1. **Text codeword:** the fixed-width binary label as ASCII `0`/`1`, serialized by `LocalTCPMessageChannel` in its UTF-8 JSON envelope.
2. **Packed frame:** the label as an integer packed into `ceil(width/8)` bytes with canonical zero padding, sent as an opaque payload in the existing `tlu.frame.v1` metadata envelope.

Both reports charge application-layer length prefixes, JSON headers, protocol ID, and one-byte callback acknowledgments. They exclude TCP/IP and link-layer headers, final-answer transmission, codebook/card distribution, prompt tokens, and all model computation. The channel boundaries and serializers are real SDK implementations. This is still an oracle: it does not show that an LLM can produce or decode these codewords.

Every codebook is measured with a fixed-length 16-byte code-specific ID and a separate common 16-byte ID control. This keeps identifier length matched while preserving an identifier that can name a codebook in the realistic arm. Codebook distribution/setup remains uncharged and must be amortized in any deployment comparison.

The nonzero-width schedule is reported in two ways:

- `empty_message_for_every_sender`: all `m` sender slots produce one envelope, including an empty zero-width codeword. This holds the schedule fixed and exposes per-message framing cost.
- `omit_zero_width_senders`: the preregistered shared protocol skips senders whose encoder partition is a singleton. The receiver inserts that encoder's one constant transcript symbol, while the sender's actual private value remains unknown and must be handled by the MAP decoder. This changes the communication policy and makes its savings visible separately.

The B=0 row under the second schedule sends no envelopes and exactly matches the no-message transport cost (zero bytes). The B=0 row under the first schedule intentionally measures `m` empty envelopes and is *not* the no-message condition.

## Observed finite results

The exact success column is inherited from, and exhaustively reverified against, each optimal codebook. Byte totals include sender-to-referee envelopes only.

| Senders | At-most payload bits | Exact success | Text app bytes: fixed / skip 0-bit senders | Packed-frame app bytes: fixed / skip 0-bit senders |
|---:|---:|---:|---:|---:|
| 2 | 0 | 4/16 = 0.2500 | 236 / 0 | 548 / 0 |
| 2 | 2 | 8/16 = 0.5000 | 238 / 238 | 550 / 550 |
| 2 | 4 | 16/16 = 1.0000 | 240 / 240 | 550 / 550 |
| 3 | 1 | 14/64 = 0.21875 | 355 / 119 | 823 / 275 |
| 3 | 3 | 24/64 = 0.3750 | 357 / 357 | 825 / 825 |
| 3 | 6 | 64/64 = 1.0000 | 360 / 360 | 825 / 825 |
| 4 | 1 | 48/256 = 0.1875 | 473 / 119 | 1,097 / 275 |
| 4 | 4 | 96/256 = 0.3750 | 476 / 476 | 1,100 / 1,100 |
| 4 | 6 | 128/256 = 0.5000 | 478 / 478 | 1,100 / 1,100 |
| 4 | 8 | 256/256 = 1.0000 | 480 / 480 | 1,100 / 1,100 |

With every sender envelope retained, moving from no-information to exact success changes application totals only slightly because the message/metadata framing dominates this scalar payload. Packed frames cost more than UTF-8 text here because the frame metadata is larger; this is a property of these two SDK envelopes, not an inherent ranking of binary and text. Omitting the known zero-width roles can save whole envelopes at low budgets, but that is a scheduling/policy change. At full budget every sender has a two-bit word, so no sender is omitted.

## Reproduction and falsification

Run `python research/multiparty_sum_transport_frontier.py` to emit the committed [exact ledger](data/MULTIPARTY_SUM_TRANSPORT_FRONTIER_V0_1.json). It includes all frontiers, per-sender bytes, two ID controls, both scheduling policies, and SHA-256 hashes for the transport calculator, exact frontier calculator/result, and channel runtime.

The measurement API [`LocalTCPFrameChannel.measure`](../tacit/channel.py) shares the exact frame builder used by `send`; its focused runtime test asserts `measure(payload) == send(payload)` on the returned cost record. The transport-frontier test patches `socket.create_connection` to fail if any measurement attempts network I/O, replays every emitted codebook over every possible private input vector, and checks the committed JSON artifact. A reproduced byte total or exact success mismatch falsifies the serialization ledger, codebook, or source hashes.

This provides a usable *systems oracle* for one tiny task family. It does not include setup amortization, model prompt/generation tokens, decoding errors, or empirical model behavior, and no LLM superiority claim follows.
