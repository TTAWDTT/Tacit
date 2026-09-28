# Private Match model-free codec frontier v0.1

## Question

What is the exact serialized payload cost of several reversible one-way encodings on a fixed synthetic private-record task, when sender and receiver share the schema and vocabulary?

This is a codec and accounting calibration. It does not test LLM comprehension, tokenizer cost, or protocol superiority.

## Frozen run

- Runner: experiments/private_match_v0_1/compare_codecs.py
- Task: Private Match v0.1, generated deterministically with the task generator
- Episodes: 128, seeds 5000 through 5127
- Receiver candidates: 8
- Record schema: 5 categorical features, 16 values per feature
- Task-sequence SHA-256: afd635dcd279f170577edc4d13e3f7382bb173cae69c93bb5d03372f9d41fe5b
- Output: research/data/PRIVATE_MATCH_CODEC_FRONTIER_V0_1.json
- All sender encoders received only sender views; all decoders received only receiver views; the scorer was called after decoding.

## Result

| Encoding | Exact decoded matches | Mean payload bytes | Mean serialized bits |
|---|---:|---:|---:|
| Fixed-width mixed-radix rank, packed binary | 128/128 | 3 | 24 |
| Ordered delimited tuple | 128/128 | 29 | 232 |
| Compact JSON object | 128/128 | 66 | 528 |
| Fixed labeled natural-language template | 128/128 | 102 | 816 |

Under this idealized shared-schema boundary, a complete record has \(16^5\) possible values. The exact worst-case zero-error information floor is 20 bits. Packing that rank into whole bytes transmits 24 bits because the final byte has four padding bits. The rank payload is 34 times smaller than the fixed labeled sentence by byte count in this configuration. This is a wire-format fact for deterministic encoders, not a claim about language-model tokens or usefulness.

All arms score 100% because each is a reversible deterministic encoder/decoder. The no-message Bayes reference is 1/8 under the task's uniform target-index prior; the centralized exact-information reference is 1.0. Neither is an LLM measurement.

## Interpretation and limits

The run confirms that explicit schema sharing and encoding choice can change the serialized channel size substantially even on a trivial exact task. It also shows why bit counts, whole-byte transport, and text-byte counts must be reported separately. It does not establish that an LLM can use a rank code, that the labeled sentence is an optimized natural-language baseline, or that the codebook/schema setup is free in a deployment.

The natural-language condition is one fixed descriptive template. The JSON and delimited conditions are also fixed serializers. No model-native tokenization, prompt overhead, receiver parsing errors, setup distribution, retry behavior, latency, or inference cost was measured. This comparison cannot pass the full-information capability gate and must not be entered as a language-performance result.

The next meaningful comparison requires a fresh LLM capability screen and model-native token accounting. If resources remain outside the frozen gate, continue only with analytic/model-free baselines and do not load a local model.
