# Private Match model-free scaling sweep v0.1

## Question

How do exact symbolic payload bounds, simple text serializers, persistent compression, receiver-visible task bytes, and the no-message reference scale with (a) the record's feature/vocabulary complexity and (b) the receiver's candidate count?

The sweep is motivated by benchmarks that vary communication complexity rather than reporting one aggregate task score; Silo-Bench, for example, organizes exact-answer tasks into levels by optimal communication complexity ([ACL 2026 paper](https://aclanthology.org/2026.acl-long.1354/)). This local sweep is much narrower: it varies one-way private-record matching only.

## Frozen run

- Runner: experiments/private_match_v0_1/scaling_sweep.py
- Episodes per valid cell: 32
- Seed start: 9000; cell seeds are deterministically separated by 100,000.
- Feature counts: 1, 2, 4, 8
- Vocabulary sizes per feature: 2, 4, 16, 256
- Receiver candidate counts: 2, 8, 32, 128
- Valid cells: 49; skipped cells: 15 where the tuple space could not contain the requested number of distinct candidates.
- Total task episodes: 1,568.
- Aggregate task-sequence SHA-256: 99e973b5840794c413b6c64991d58f3cd377a9df1926f3608f47ba6c32c12d8c
- Output: research/data/PRIVATE_MATCH_SCALING_V0_1.json

Each cell runs all four deterministic codec arms and the zlib comparisons from codec frontier v0.2. Sender and receiver context size is measured as compact UTF-8 JSON for the task view only; no system prompt, tokenizer, or model is involved.

## Results

### Record complexity at two receiver candidates

| Features d | Values V | Exact rank floor (bits) | Packed rank (bytes) | Delimited tuple (bytes) | Compact JSON (bytes) | Fixed labeled sentence (bytes) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1 | 1 | 5 | 14 | 50 |
| 2 | 4 | 4 | 1 | 11 | 27 | 63 |
| 4 | 16 | 16 | 2 | 23 | 53 | 89 |
| 8 | 256 | 64 | 8 | 47 | 105 | 141 |

Across every valid grid cell, the rank arm's stated worst-case bit floor matched \(\lceil d\log_2 V\rceil\), and all deterministic codec round trips scored 100%. Packed byte width is \(\lceil d\log_2 V/8\rceil\), so it changes in byte steps. This is a verification of the known theorem over the grid, not an empirical LLM scaling law.

### Candidate-count growth at fixed schema

For d=4 and V=16, the schema is held constant while n changes:

| Candidates n | No-message Bayes accuracy | Mean receiver-view bytes | Rank payload (bytes) | Delimited tuple (bytes) |
|---:|---:|---:|---:|---:|
| 2 | 1/2 | 295 | 2 | 23 |
| 8 | 1/8 | 823 | 2 | 23 |
| 32 | 1/32 | 2,935 | 2 | 23 |
| 128 | 1/128 | 11,383 | 2 | 23 |

As predicted by the task definition, candidate count changes the receiver's table/context and the no-message reference, while it does not change these sender-only schema encodings. This is conditional on the declared boundary: the sender does not know the receiver table or row IDs. The measured view bytes are not model input tokens.

## Interpretation and limits

Within this task family, worst-case symbolic message length scales as \(d\log_2 V\) bits, while receiver task-view serialization grows with the number of candidates. These are separate scaling axes: table search size does not enter this sender-independent code's message floor. The sweep gives a concrete machine-readable frontier for a task where exact deterministic controls are available.

The result does not establish a language scaling law. It has one sender and one receiver, one message, exact tuple semantics, a uniform synthetic prior, shared schema/vocabulary, and deterministic decoders. It does not vary agent count, communication rounds, model capability, compositional transfer, or latent channels. Text byte lengths are not tokenizer lengths. Persistent zlib context may exploit corpus order and requires synchronized stream state. No inference or paid API calls were used.

Next, keep this analytical scaling result separate from the LLM claim. A later model study must first pass a fresh receiver-capability and local-resource gate, then test held-out feature combinations and heterogeneous receivers with model-native token and full-cost accounting.
