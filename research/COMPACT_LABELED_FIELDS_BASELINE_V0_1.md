# Compact labeled-fields baseline v0.1

**Status:** superseded before inference by [v0.2](COMPACT_LABELED_FIELDS_BASELINE_V0_2.md). This default-ontology-only version was frozen but never run on a model. It remains as a record of the initial baseline and its exact payload calculation.

## Motivation

The current v0.4 `natural_language` arm requests a short sentence that repeats all attribute names. The `json` arm uses a compact JSON object. Neither is a lower-overhead human-readable field serialization. A deterministic `name=value` line provides a stronger compact-text control while preserving explicit labels and exact domain values.

The initial card used `shape`, `color`, `quantity`, and `texture` in a fixed order. It was replaced with one ontology-general card before model outcomes existed, so the default-only card is no longer part of the active comparison. The format did not encode meaning IDs, candidate IDs, or split membership.

## Frozen message grammar

For ordered fields `(shape, color, quantity, texture)`, the sender emits:

```text
shape=<exact value>;color=<exact value>;quantity=<exact value>;texture=<exact value>
```

There are no spaces, quotes, or extra text. The receiver matches all four exact, case-sensitive values to one complete candidate tuple. Any missing, duplicate, reordered, extra, malformed, or wrong-valued field is a format/fidelity failure, even if a language model still chooses the correct candidate.

## Model-free payload calculation

Across the 64 default held-out meanings at split seed 17, this grammar's payloads range from 49 to 57 UTF-8 bytes (mean 53). Compact JSON with the same attribute names and exact values ranges from 67 to 75 bytes (mean 71). For each tuple in this ASCII ontology, the compact labeled-fields serializer saves exactly 18 payload bytes: both carry the same field names and values, while their punctuation overhead differs. This is a deterministic serialization calculation, not provider-token use, application-envelope savings, or an LLM experiment. The compact JSON reference uses `ensure_ascii=False` and separators `(',', ':')`, so whitespace is not inflated to manufacture a gain.

## Falsifiable prediction and run requirements

- On successful format-following outputs, the offline syntax parser must accept exactly the frozen grammar, and parsed fields must equal the sender's private tuple. A different message that succeeds at final candidate selection still fails this representation-fidelity criterion.
- The payload-byte difference against compact JSON must equal 18 bytes per exact same tuple under this ontology. Any other value indicates serializer drift or mismatched contents.
- End-task success is scored separately using the existing strict candidate-ID outcome. Do not infer the format's value from parser validity alone.
- Count card bytes and repeated prompt tokens, generated message bytes, delivered application-envelope bytes, receiver tokens, latency, and failures/retries. The static payload calculation does not establish a frontier advantage.
- The frozen scorer is [`research/score_compact_fields.py`](score_compact_fields.py). It reads a single-condition raw `shared_protocol_card` result JSONL, verifies the exact frozen card hash and the runner sidecar's source hash/condition/card binding, and writes a separate audit file without changing the cost ledger. It reports syntax, vocabulary decoding, label fidelity, and exact task selection independently. The generic runner still leaves its inline `semantic_parse_valid` and `exact_format_valid` fields null for generic cards; use the offline report for this card.
- Preserve the separate receiver capability gate and fresh resource preflight. Do not use validation/test outcomes to rewrite this card or select a syntax variant.

Example after a frozen run (replace the output path only within the project):

```powershell
python research/score_compact_fields.py --input .cache/emergent_ood_v0_4/compact-fields-raw.jsonl --output .cache/emergent_ood_v0_4/compact-fields-audit.json
```

The scorer refuses to overwrite the source ledger or card, refuses unpinned card bytes, and refuses to replace existing output unless `--force` is supplied.

## Interpretation boundary

This baseline still communicates the exact semantic fields in visible text. It does not test learned composition, hidden symbols, cross-model codebook transfer, error correction, or a language emerging among agents. Its research value is to challenge any apparent compact-code gain against a simple, legible, low-payload reference.
