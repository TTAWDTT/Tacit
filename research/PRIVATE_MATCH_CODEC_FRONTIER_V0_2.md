# Private Match codec and compression frontier v0.2

## Question

Does ordinary lossless compression materially change the model-free wire-size ordering once tiny messages are sent independently, in a persistent stream, or with a shared dictionary whose setup bytes are charged?

This is a codec and accounting calibration. It does not test LLM comprehension, tokenizer cost, or protocol superiority.

## Frozen run

- Runner: experiments/private_match_v0_1/compare_codecs.py
- Task: Private Match v0.1, generated deterministically
- Episodes: 128, seeds 5000 through 5127
- Receiver candidates: 8
- Record schema: 5 categorical features, 16 values per feature
- Task-sequence SHA-256: afd635dcd279f170577edc4d13e3f7382bb173cae69c93bb5d03372f9d41fe5b
- Compression: zlib level 9, compile/runtime zlib 1.2.13
- Persistent mode: one continuous DEFLATE stream; newline record delimiter included; Z_SYNC_FLUSH after each message and Z_FINISH at the end.
- Shared dictionary: fixed schema/phrase/value dictionary, 198 raw bytes, SHA-256 512d5ff1ca5cdedcd5f19e32f7db0d06580bf8624e4c3f17a9a06ea9ebff87f2.
- Output: research/data/PRIVATE_MATCH_CODEC_FRONTIER_V0_2.json

The zlib format wraps DEFLATE with stream metadata and integrity checking; DEFLATE itself combines LZ77-style references with Huffman-coded symbols. The codec is a strong generic lossless baseline, but its small-message framing, flush, and shared-dictionary costs belong in the measured boundary. See [RFC 1950](https://www.rfc-editor.org/rfc/rfc1950.html) and [RFC 1951](https://www.rfc-editor.org/rfc/rfc1951.html).

## Results

All deterministic encoders and decoders returned the exact candidate in 128/128 episodes. The table reports bytes per episode; independent zlib bytes are summed across separately terminated streams, while persistent-stream sizes include the in-stream newline and per-message sync flush.

| Encoding | Raw payload | Independent zlib per message | Persistent zlib, no dictionary | Persistent zlib with dictionary | With dictionary amortized at 128 messages |
|---|---:|---:|---:|---:|---:|
| Fixed labeled natural-language template | 102.00 | 77.30 | 17.33 | 17.02 | 18.57 |
| Compact JSON object | 66.00 | 45.30 | 16.75 | 16.73 | 18.27 |
| Ordered delimited tuple | 29.00 | 25.59 | 13.98 | 13.95 | 15.49 |
| Fixed-width mixed-radix rank, packed binary | 3.00 | 11.00 | 10.06 | 10.09 | 11.64 |

At this horizon, transmitting the 198-byte dictionary makes every dictionary-coded stream larger overall than the same persistent stream without that dictionary. A linear extrapolation from the measured per-message savings gives rough break-even horizons of 650 messages for the fixed labeled sentence, 8,448 for JSON, and 6,336 for the delimited tuple. These are extrapolations from one seed sequence, not measured long-run thresholds; zlib stream block decisions and message distributions can change with a longer run. The dictionary does not reduce the rank stream here.

Independent zlib streams are still large for these short messages because every message pays a new wrapper/trailer and adaptive coding overhead. Persistent compression amortizes that overhead and exploits repeated field names and template text. It remains more expensive than sending the 3-byte rank directly in this experiment. Applying zlib to the already packed random-looking rank expands it substantially.

The ideal worst-case zero-error information bound remains 20 bits. The rank's 3-byte payload is 24 transmitted bits after byte alignment. The no-message Bayes reference is 1/8 and the centralized exact-information reference is 1.0; neither is an LLM measurement.

## Interpretation and limits

This comparison strengthens the baseline boundary: a new text protocol must be compared with standard compression, persistent stream state, and dictionary amortization, not just uncompressed JSON and prose. On this small, uniformly sampled task, a known task-specific rank beats these general-purpose compressors in raw wire bytes. That says nothing about whether an LLM can emit or decode that rank or whether a shared schema/rank code is practical across tasks.

The labeled sentence is one fixed template, not optimized natural language. No tokenizer, prompt, model, receiver failure, setup distribution beyond the explicit dictionary bytes, latency, or inference cost is measured. Persistent streams also assume synchronized compressor/decompressor state and ordered delivery; loss recovery and random access are out of scope. This is not a language-performance result.

The experiment makes the coding baseline sharper while leaving the key research gate unchanged: compare the best viable formats on a receiver-capable LLM task with native tokenization, complete inference costs, matched budgets, and held-out transfer.
