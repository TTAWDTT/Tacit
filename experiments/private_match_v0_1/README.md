# Private Match v0.1

**Status:** model-free task and scorer calibration. This is a local, synthetic adaptation of the private-record matching task family; it is not a reproduction of MT-PingEval and contains no upstream records.

## Task

Two agents share a schema. The sender sees one target record with `d` categorical fields. The receiver sees `n` candidate records, each with a public candidate ID. Exactly one candidate is an exact match to the sender's target. The receiver must return that candidate ID. The sender is not given the receiver's table or candidate IDs; the receiver is not given the target or gold ID. There is no action oracle, shared target index, task label, or other privileged hint.

`generate_tasks.py` uses only the Python standard library and produces three separate JSONL files:

- `sender.jsonl`: sender view only;
- `receiver.jsonl`: receiver view only;
- `gold.jsonl`: scorer-only target candidate IDs.

The generator creates distinct synthetic tuples from a finite vocabulary, shuffles them, and draws the target position from a separately domain-separated deterministic random stream. Under the task's independent-uniform sampling definition, conditional on any fixed receiver table, the target ID is uniform over its `n` rows. The exact no-message Bayes accuracy is `1/n`; the centralized exact-information oracle scores 1.0. These are distributional controls, not LLM results. Episode IDs do not expose generation seeds.

With `d` fields and `V` values per field, the record space has `V^d` possible tuples. A zero-error one-way sender that does not know the receiver's table must encode every tuple distinctly, so the exact worst-case payload floor is `ceil(log2(V^d)) = ceil(d log2 V)` bits. A fixed-width shared tuple-rank code attains it. The proof and assumptions are in [`docs/THEORY.md`](../../docs/THEORY.md#10-exact-one-way-coding-floor-for-private-record-matching). This bit floor is a symbolic control, not an LLM-token prediction.

Example generation:

```powershell
python experiments/private_match_v0_1/generate_tasks.py `
  --output .cache/private_match_v0_1/dev `
  --episodes 32 --seed 1000 --candidates 8 --features 5 --vocabulary-size 16
```

Existing files are preserved unless `--force` is supplied. The manifest stores task parameters, Python runtime/randomness metadata, the analytic no-message reference, and SHA-256 hashes of all role-separated files. Keep sender/receiver views physically separate in every runner; never construct a combined model prompt from `gold.jsonl` or provide the manifest's seed to either agent.

## Model-free codec calibration

Run compare_codecs.py to compare a labeled natural-language template, compact JSON, an ordered delimited tuple, and a fixed-width mixed-radix rank code over the same role-separated task generator. For example:

    python experiments/private_match_v0_1/compare_codecs.py --output .cache/private_match_v0_1/codec_frontier.json --episodes 128 --seed 5000 --candidates 8 --features 5 --vocabulary-size 16

The report gives exact match accuracy, mean/median/p95/max UTF-8 or binary payload bytes, serialized payload bits, and the ideal worst-case zero-error bit bound. For a 5-field, 16-value schema, the theorem's lower bound is 20 bits, while a byte-aligned rank payload occupies 3 bytes (24 transmitted bits); this explicitly exposes final-byte padding. It also compares zlib level 9 on independent messages, one persistent stream with a flush at each message boundary, and a persistent stream with a fixed shared dictionary. The latter reports dictionary bytes/hash, setup-inclusive wire cost at the evaluation horizon, and a clearly labeled linear break-even extrapolation. zlib follows the standardized lossless DEFLATE method described in [RFC 1950](https://www.rfc-editor.org/rfc/rfc1950.html) and [RFC 1951](https://www.rfc-editor.org/rfc/rfc1951.html); stream state, line framing, flushing, dictionary distribution, Python/zlib versions, and all transmitted bytes are recorded. All codec arms are deterministic and should decode perfectly. Their purpose is to verify the cost boundary and give reference wire sizes, not to measure an LLM's ability to follow or understand a format. The labeled sentence is a fixed template, not a prompt-optimized natural-language baseline. The report explicitly marks LLM token and inference cost as unmeasured.

Each encoder receives only the sender view; each decoder receives only the receiver view. The scorer is called after decoding. A task-sequence SHA-256 identifies the exact generated role inputs and scorer references used in the aggregate. The mixed-radix arm assumes a shared feature order and vocabulary, just like the theoretical bound; shared-state setup is not included in its payload count. The zlib preset dictionary is not assumed to arrive free: both its raw size and its effect on the comparison horizon are shown. The v0.2 result is frozen in [the compression frontier report](../../research/PRIVATE_MATCH_CODEC_FRONTIER_V0_2.md).

## Model-free scaling sweep

The scaling runner varies record width, vocabulary size, and receiver candidate count on 32 fresh tasks per valid configuration. Its defaults cover feature counts 1/2/4/8, vocabulary sizes 2/4/16/256, and candidate counts 2/8/32/128; impossible tables are recorded as skipped. Run it with:

    python experiments/private_match_v0_1/scaling_sweep.py --output .cache/private_match_v0_1/scaling_sweep.json --episodes 32 --seed 9000

It reports the exact rank-bit floor and packed bytes, every codec/compression arm, no-message Bayes reference, and compact JSON byte size of each agent's task view. These are separate axes: candidate count grows receiver context and reduces no-message accuracy while sender-only schema encodings keep the same payload length. Task-view bytes are not model tokens. The frozen sweep and interpretation are in [PRIVATE_MATCH_SCALING_V0_1.md](../../research/PRIVATE_MATCH_SCALING_V0_1.md).

## Research role and limits

The motivating [MT-PingEval paper](https://arxiv.org/abs/2602.24188) uses private personal-record tables and automatically generated instances. It reports that multi-turn accuracy gains can largely follow a random guess-and-check baseline. Private Match v0.1 deliberately uses a fixed one-way sender-to-receiver schedule first, so it can calibrate semantic payload representation without crediting extra turns or lucky guesses. It is a deliberately small task family: exact tuple matching does not test broad scientific reasoning, planning, long-horizon coordination, or natural-language grounding.

Before any LLM format comparison, require a receiver full-information capability gate on fresh tasks. Compare no message, correct target message, a target message deranged from another episode, a fixed-width shared tuple-rank code, and a centralized oracle; keep task schedule and prompts fixed. Candidate representation arms include optimized concise natural language, JSON, delimited tuples, and a compact code only if its setup/codebook cost is declared. Measure exact answer, semantic decoding, protocol adherence, actual serialized payload bytes, receiver-native input/output tokens, full prompt/completion costs, latency, and failure/retry counts. Equal-budget curves and cross-model transfer are required before a protocol claim. The model-free codec calibration is only a wire-size and scorer check; it cannot pass the LLM capability gate.

The task manifest is not a preregistration. No model run, language comparison, scaling result, or superiority claim is included in this artifact.
