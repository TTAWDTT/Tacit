# Cross-tokenizer protocol cost transfer v0.1

**Status:** preregistered deterministic tokenizer comparison; no model-performance claim.

## Motivation and source pin

The previous Qwen3-4B content-token sweep found a dense dimensional crossover for compact labeled fields against JSON at `d=24`. That result depends on a tokenizer and does not establish cross-model robustness. This follow-up measures the identical messages and prompt-construction code with the Mistral-7B-Instruct-v0.3 tokenizer as a second, architecturally distinct vocabulary.

Use only `tokenizer.json` from the official [Mistral-7B-Instruct-v0.3 Hugging Face repository](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/adadfb3fbae87ecc77cd5bf2c3318434d5da04cf/tokenizer.json), pinned to revision `adadfb3fbae87ecc77cd5bf2c3318434d5da04cf`. The repository exposes this file as a 1.96 MB tokenizer artifact and identifies the model repository license as Apache-2.0. Do not download or load any model weights. Record the downloaded file's SHA-256 before counting.

## Frozen design

- Dimensions `d ∈ {4,10} ∪ {20,21,…,40}` and cardinalities `V ∈ {2,4,8,10}` (92 settings).
- Sixteen deterministic prompt fixtures per `(d,V)`, four receiver candidates per fixture, and the same identity-rank modular-holdout constructor as the Qwen sweeps.
- Same v0.4 prompt constructors, compact-fields v0.3 card, and JSON, card, and fixed-digit conditions. No prompt, message, candidate table, or episode fixture may be adapted to Mistral tokenization.
- Count with the Rust `tokenizers` library from `tokenizer.json`, `add_special_tokens=False`. Record UTF-8 bytes separately. This is content tokenization only: exclude Mistral chat-template tokens, receiver output, caching, inference compute, latency, billing, and task outcomes.
- Primary measure: mean known content tokens per episode and each alternative's delta against JSON, with the same sender/receiver inputs, receiver-side delivered-message echo, and ideal sender output boundary used previously.
- Preserve every setting and bind this plan, Mistral tokenizer revision/hash, prompt/card/runner/codec hashes, and preceding Qwen confirmation JSON hash into the output.

## Falsifiable predictions

**P21a — crossover portability.** For every tested `V`, the compact-fields minus JSON difference is positive at `d=20` and negative at `d=40`, replicating the direction of the Qwen bracket. A sign failure at either endpoint falsifies transfer of the bracket to this tokenizer.

**P21b — location transfer.** For every tested `V`, the first non-positive integer dimensionality is in `{23,24,25}`. This deliberately tests the observed Qwen location rather than assuming that a sign change anywhere counts as a successful prediction. If the sign changes outside this set, report the alternative location and reject P21b.

**P21c — codebook acquisition burden.** The fixed-symbol arm remains more expensive than JSON at every tested `(d,V)`, and its excess increases strictly with `V` at each fixed `d`. Any violation is reported cell-by-cell.

No model-level inference, endpoint calls, or accuracy claims are part of this study. Even if P21a–P21c hold, the result generalizes only this deterministic content-token accounting to one additional tokenizer; model behavior, chat templates, quality-cost frontiers, and other tokenizer families remain untested.
