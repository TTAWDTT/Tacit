# Compact labeled-fields baseline v0.3: prompt-cost audit

**Status:** active frozen candidate for a future v0.4 comparison; no LLM result exists. The protocol card is [`examples/compact_labeled_fields_v3.json`](../examples/compact_labeled_fields_v3.json). v0.3 keeps the cross-ontology grammar from v0.2 and shortens its repeated instructions after an offline cost audit.

## Observation that changed the protocol

The first ontology-general card had a real communication-economy risk: the v0.2 sender and receiver instructions totaled 1,574 UTF-8 bytes per episode in the current runner, compared with 665 bytes for the built-in JSON instructions and 630 for built-in natural-language instructions. That estimate counts only the two role instruction strings supplied on each episode's sender/receiver calls. It is not model-token accounting, but it reveals that an 18-byte output-payload saving could be overwhelmed on a broader request-cost axis if input instructions are longer.

The active v0.3 card reduces its total role-instruction strings to 926 bytes per episode (sender 424, receiver 502) for the default task. JSON remains 665 bytes and natural language 630 under the same runner. The card therefore still adds 261 instruction bytes over JSON and 296 over natural language per episode. The generic baseline must not be described as a total-cost improvement on this evidence.

These byte counts are deterministic from `_Protocol` and UTF-8 serialization. They are not input-token counts, since tokenization depends on the model/tokenizer and possibly provider-side prefix caching. The exact input-token cost will be measured from each frozen model call if/when the resource and receiver-capability gates allow an experiment. The output application message remains a separate cost axis; do not subtract prompt bytes from payload bytes as if they used the same channel.

## Frozen v0.3 message grammar

The card is ontology-general across the included default, robotics, and music fixtures. It transmits every exact field as `key=value`, sorted by key, joined with `;`. The same card is used for sender and receiver across all three domains. The deterministic codec, post-run audit, and cross-ontology exact payload comparison are documented in [v0.2](COMPACT_LABELED_FIELDS_BASELINE_V0_2.md); the card-specific scorer is [`score_compact_fields.py`](score_compact_fields.py).

The three-ontology serializer calculation remains exactly 18 fewer payload bytes than compact JSON for the same tuple. That is only an application-payload reduction. Whether it survives full input/output token use, receiver errors, and total inference costs is unknown.

## Falsifiable experiment and accounting

- Keep one protocol-card hash unchanged across all three ontology strata and held-out test rows.
- On every episode, report exact candidate success, syntax validity, complete-schema parsing, sender-value fidelity, output bytes, application-envelope bytes, complete provider-reported input/output tokens, model calls, latency, and failures. The offline scorer reads evaluator traces after a run and must not filter or rewrite the cost ledger.
- Compare paired tasks against compact JSON, concise NL, no-message, full-information, and other eligible strong baselines. Plot task success against input tokens and output wire bytes separately; also publish complete-cost summaries rather than collapsing these quantities into a payload-only score.
- If the compact representation saves output bytes but spends more input tokens or loses receiver success, it has not improved the overall frontier at that operating point. If it improves only after repeated-prefix caching, evaluate and label that runtime condition separately.
- Preserve the independent receiver capability gate and fresh resource preflight. A runner dry-run only checks that the card and episode schema can be planned; it does not satisfy either gate.

## Current evidence and limits

The offline encoder/scorer covers all 192 held-out tuples across the three included ontologies, and the frozen card loads in dry-run for each. No model was loaded and no inference result exists. The ontology set is synthetic and hand-authored, the grammar has no escaping for `;` or `=`, and the instruction-byte comparison cannot substitute for model-specific tokenizer usage. This is a stronger baseline candidate, not a newly discovered LLM language.
