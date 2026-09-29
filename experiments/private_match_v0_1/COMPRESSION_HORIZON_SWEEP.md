# Compression horizon sweep v0.1

## Question and prior

The v0.2 codec report estimated dictionary break-even horizons by linearly
extrapolating savings observed over 128 episodes. This sweep replaces that
extrapolation with exact cumulative stream sizes at each preregistered horizon
across five deterministic task sequences. It is a model-free wire-cost control,
not an LLM communication result.

The predictions, seeds, task dimensions, serializer/compressor settings,
horizons, primary metric, and limits are frozen in
[`compression_horizon_prereg_v0_1.json`](compression_horizon_prereg_v0_1.json).

## Reproduce

```powershell
python experiments/private_match_v0_1/compression_horizon_sweep.py --output research/data/PRIVATE_MATCH_COMPRESSION_HORIZON_V0_1.json
```

The runner compares persistent zlib level 9 with and without the same 198-byte
shared dictionary used in codec frontier v0.2. Each message includes an LF
record delimiter and is made visible with `Z_SYNC_FLUSH`; the stream is finished
at each measured horizon. The primary result is
`no_dictionary_stream_bytes - (dictionary_bytes + dictionary_stream_bytes)`.
Positive values mean dictionary transfer saved total bytes at that horizon.
The dictionary is charged once per independent stream and receiver. It is not
charged once per episode.

Every stream is decompressed and byte-compared with its source messages before
its size is recorded. The report includes a task-sequence hash for every seed,
exact per-seed cumulative byte counts, pooled descriptive summaries, and the
first measured horizon where each dictionary arm is strictly smaller. It does
not interpolate a break-even between measured horizons.

## Scope

Only payload and one-time dictionary bytes are compared. Compression-library
installation, parser/runtime, tokenizer use, prompt cost, model inference,
latency, and model decoding ability are not measured. The single-receiver setup
does not determine the cost of distributing a dictionary to many agents.
