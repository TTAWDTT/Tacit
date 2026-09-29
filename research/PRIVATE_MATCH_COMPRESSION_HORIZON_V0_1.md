# Private Match compression horizon v0.1

## Question

Does sending the fixed zlib preset dictionary once repay its setup bytes over
longer persistent message streams? The previous v0.2 report projected a
break-even horizon by treating the average byte saving over 128 messages as a
constant per-message saving. This run measures cumulative bytes directly.

The codec and horizon grid were preregistered in
[`compression_horizon_prereg_v0_1.json`](../experiments/private_match_v0_1/compression_horizon_prereg_v0_1.json).

## Frozen design and execution

- Five independent deterministic task seeds: 12,000, 22,000, 32,000, 42,000,
  and 52,000.
- 16,384 messages per seed; exact cumulative stream sizes measured at 14
  horizons from 1 through 16,384.
- Eight-candidate Private Match, five categorical features, 16 values per
  feature; labeled text, compact JSON, delimited tuple, and fixed-width rank.
- zlib 1.2.13, level 9, one persistent stream, LF delimiter per message,
  `Z_SYNC_FLUSH` after each message, `Z_FINISH` at each measured horizon.
- 198 dictionary bytes are charged once per independent stream and one receiver.
- Every compressed stream was decompressed and compared byte-for-byte with its
  source messages before it was counted.

The primary metric is

`no_dictionary_stream_bytes - (198 + dictionary_stream_bytes)`.

Positive values mean the dictionary arm used fewer total bytes. The 70
seed-by-horizon combinations contain 280 codec rows in the
[machine-readable results](data/PRIVATE_MATCH_COMPRESSION_HORIZON_V0_1.json).

## Results

Across all five seeds, the dictionary never beat the no-dictionary stream at
any measured horizon. Mean compressed-stream savings (no dictionary stream
minus dictionary stream) barely changed between 128 and 16,384 messages:

| Codec | Stream bytes saved at H=128 | Stream bytes saved at H=16,384 | Net after 198-byte dictionary at H=16,384 | Seeds where dictionary won |
|---|---:|---:|---:|---:|
| Labeled text template | 38.2 | 38.4 | -159.6 | 0/5 |
| Compact JSON | 2.2 | 2.4 | -195.6 | 0/5 |
| Delimited tuple | 3.8 | 3.8 | -194.2 | 0/5 |
| Packed rank bytes | -4.0 | -4.0 | -202.0 | 0/5 |

The rank-arm negative saving means that the dictionary stream itself was four
bytes larger on average. Every stream remained a successful lossless round
trip.

### Direct extension of the v0.2 task sequence

The first 128 messages from seed 5,000 exactly reproduce the four persistent
stream-size pairs in v0.2: labeled text 2,218/2,179 bytes, JSON 2,144/2,141,
delimited tuple 1,789/1,785, and packed rank 1,288/1,292 (baseline/dictionary
stream). Extending that same seed sequence to 16,384 messages produces
241,156/241,117; 230,117/230,114; 202,731/202,727; and 163,729/163,733 bytes.
The dictionary's stream-only byte advantage is exactly unchanged for each arm.
See the [legacy-seed extension data](data/PRIVATE_MATCH_COMPRESSION_HORIZON_LEGACY_SEED_V0_1.json).

## Interpretation and correction

The v0.2 values of 650 messages for labeled text, 8,448 for JSON, and 6,336 for
the delimited tuple were linear extrapolations, not observed crossover points.
They are withdrawn. In the matching legacy sequence, the apparent 128-message
per-message gains came almost entirely from the initial stream state; they did
not accumulate as the stream grew. Across all five new seeds, total dictionary
benefit was already near its observed plateau by 128 messages and remained far
below the 198-byte setup transfer. The data therefore reject a constant
per-message extrapolation for this codec/task setting.

The general setup-cost condition remains `A + C_dict(H) <= C_base(H)`. The
simpler `A + H*c_dict <= H*c_base` crossover formula requires stationary
per-episode costs. Stateful streaming codecs can violate that assumption; use
measured cumulative curves unless a steady-state marginal saving is established.

This result does not prove that no dictionary can ever repay its cost. It shows
that this fixed dictionary did not do so through 16,384 messages under this
uniform synthetic workload and point-to-point transfer boundary. A dictionary
preinstalled at both endpoints has a different setup assumption. The library,
parser, tokenizer, inference, decoder-training, and multi-recipient distribution
costs are not measured. This is not an LLM or language-superiority result.
