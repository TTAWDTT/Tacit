# Ontology-general compact labeled-fields baseline v0.2

**Status:** frozen prompt-card candidate for a future cross-ontology v0.4 comparison. No model result exists. v0.2 supersedes the default-ontology-only v0.1 card before its first model run. This remains a hand-authored structured-text baseline, not a discovered language or evidence of protocol superiority.

## Research question

Can explicit semantic labels keep a compact text representation decodable across domain vocabularies while approaching the payload size of an opaque code? This is a strong, interpretable baseline against which any LLM-specific protocol must be tested.

## Frozen protocol

The single frozen card is [`examples/compact_labeled_fields_v2.json`](../examples/compact_labeled_fields_v2.json), used through the existing `shared_protocol_card` condition. It does not name the ontology or hardcode fields. Both agents use this grammar:

1. Emit one `attribute=value` pair per field.
2. Sort exact attribute names in ascending Unicode code-point order.
3. Separate pairs with `;`, with no spaces around `=` or `;`.
4. Preserve exact key/value spelling. Spaces inside a key or value are allowed.
5. Internal spaces in names/values are preserved; leading/trailing whitespace is unsupported. The current grammar has no escaping, so keys and values containing `;` or `=` are unsupported.

The sender receives only its private meaning and the shared card. The receiver receives its candidate table and the same card. Neither needs a split seed, episode ID, target ID, candidate order convention, or examples. This uses lexical field labels as the shared compositional primitives. It is explicitly not an opaque token code.

## Model-free payload comparison

For each held-out tuple, compare the exact UTF-8 payload `key=value;...` with compact JSON (`ensure_ascii=False`, separators `(',', ':')`) containing the same keys and values. In each of the three included ontologies, all keys and values avoid the reserved delimiters and the labeled-fields string saves exactly 18 payload bytes per tuple. At seed 17, each held-out support has 64 tuples:

| Ontology | Labeled fields: min / mean / max bytes | Compact JSON: min / mean / max bytes | Payload saving |
|---|---:|---:|---:|
| Default | 49 / 53.00 / 57 | 67 / 71.00 / 75 | 18 bytes per tuple |
| Robotics v1 | 48 / 54.00 / 61 | 66 / 72.00 / 79 | 18 bytes per tuple |
| Music v1 | 52 / 56.75 / 62 | 70 / 74.75 / 80 | 18 bytes per tuple |

The constant difference is purely punctuation overhead: these serializers carry the same exact labels and values. It is a payload-byte result, not a tokenization, application-envelope, latency, inference-cost, or task-success result. Repeated prompt-card costs and all model input/output tokens remain part of any measured frontier.

## Offline scorer

[`research/score_compact_fields.py`](score_compact_fields.py) pins this card's SHA-256, verifies the v0.4 run manifest binds the raw JSONL bytes to this card and to a single `shared_protocol_card` condition, and writes a separate audit artifact. For each raw sender message it reports:

- syntactic parse validity (fields can be split into unique `key=value` pairs);
- exact format validity (the complete expected key set is present in canonical order);
- semantic parse validity (the parsed keys match the target schema);
- canonical label fidelity (all decoded key/value pairs exactly equal the sender's private tuple);
- strict receiver candidate-ID success.

These outcomes stay separate. A receiver can select the right candidate after a malformed or unfaithful message; that does not establish successful use of this grammar. The scorer reads evaluator-only target tuples after the run and never provides them to either model or modifies the source cost ledger.

Example after a frozen model run:

```powershell
python research/score_compact_fields.py --input .cache/emergent_ood_v0_4/compact-fields-raw.jsonl --output .cache/emergent_ood_v0_4/compact-fields-audit.json
```

## Falsifiable predictions and limits

- The exact same card and grammar can be applied unchanged to default, robotics, and music held-out episodes. A cross-ontology comparison must freeze the card before validation/test outcomes and retain ontology as a task stratum.
- For every identical tuple under the included delimiter-free labels, this payload serializer must remain exactly 18 bytes shorter than compact JSON. Any other result indicates serializer drift or a mismatch in content.
- If syntax adherence and exact sender fidelity hold but the receiver's exact success is indistinguishable from `1/k`, a compact payload has not produced usable communication.
- The current three symbolic ontologies do not establish grounding, natural-domain generalization, or transfer to unfamiliar attribute labels beyond lexical copying. The format requires messages to carry all `d` fields; it has no special savings for higher-order held-out support.
- Values or attribute names containing `;` or `=`, or with leading/trailing whitespace, require a separately specified grammar and new card hash. Do not silently normalize or escape them.
- Compare complete cost: repeated instruction tokens, sender and receiver model calls, generated and delivered bytes, application envelope, failed deliveries, decoding failures, latency, and any setup/amortization. The 18-byte payload advantage alone does not establish a frontier gain.

## Verification status

The offline serializer/scorer is exhaustively checked over all 64 held-out meanings in each included ontology. The same frozen card passed dry-runs against default, robotics, and music episode bundles; each one-set plan contains eight calls under the 12-call cap and reports `model_loaded=false`, `inference_started=false`. No real model run, capability pass, or resource-gated inference has occurred. The v0.1 default-only card was never used for model inference and is superseded before outcomes were observed.
