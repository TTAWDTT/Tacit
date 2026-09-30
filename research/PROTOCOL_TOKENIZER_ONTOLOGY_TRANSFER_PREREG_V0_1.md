# Mistral content-token costs across held-out ontologies v0.1

**Status:** preregistered deterministic cost replication; no model-performance or statistical-inference claim.

## Motivation

The Qwen tokenizer audit over three held-out ontologies found compact-fields v0.3 more expensive than JSON in every domain (`+46.938` to `+47.594` known content tokens per episode). A later synthetic Mistral-tokenizer sweep used generic `axisNN`/`vNN_NN` labels and found the same direction in its small-dimensional fixture. This frozen follow-up tests whether that negative cost result transfers to the repository's actual default, robotics, and music attribute vocabularies and keyed episode ledgers under the Mistral tokenizer.

## Frozen data and method

- Reuse, without regeneration or edits, the project-local test bundles used in the prior Qwen audit: default split seed 23 and robotics/music split seed 31; 64 test episodes per ontology, 192 total.
- Use the same exact v0.4 runner `_Protocol`, sender/receiver/user JSON construction, protocol card v0.3, candidate tables, and JSON, compact-fields, and fixed-digit conditions as `research/audit_protocol_token_costs.py`.
- Tokenizer: official `mistralai/Mistral-7B-Instruct-v0.3` `tokenizer.json` at immutable revision `adadfb3fbae87ecc77cd5bf2c3318434d5da04cf`; expected SHA-256 `e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49`; `add_special_tokens=False`.
- Primary outcome: per-ontology mean/min/max of each protocol's known content-token total and its paired delta from JSON. Known total includes both role inputs, the echoed message in the receiver input, and ideal sender output; it excludes receiver output, chat templates, provider caches/usage, actual model behavior, inference compute, billing, and latency.
- Compare the Mistral results with the hash-pinned existing Qwen three-ontology artifact. Report ontology-specific differences; do not pool or reweight domains after seeing the data.
- Bind preregistration, bundle manifests, both tokenizer artifacts, runner, codec, card, and source scripts in machine-readable output. Retain all three domains even if a prediction fails.

## Predictions

**P22a — ontology-general negative cost result.** Compact-fields known content-token cost exceeds JSON in each of the three Mistral-tokenized ontologies. If any domain is non-positive, reject this cross-tokenizer direction prediction for that ontology and report its paired distribution.

**P22b — explicit-map burden.** The fixed-symbol arm costs more than JSON in each Mistral-tokenized ontology. Any domain-specific reversal falsifies this prediction.

No prediction is made that Mistral margins equal Qwen margins. Differences between tokenizers and ontologies are descriptive and cannot be converted into model quality, paid-token billing, or a reusable setup break-even without provider/model outcomes. No inference calls are part of this study.
