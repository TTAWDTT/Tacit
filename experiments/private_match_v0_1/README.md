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

## Research role and limits

The motivating [MT-PingEval paper](https://arxiv.org/abs/2602.24188) uses private personal-record tables and automatically generated instances. It reports that multi-turn accuracy gains can largely follow a random guess-and-check baseline. Private Match v0.1 deliberately uses a fixed one-way sender-to-receiver schedule first, so it can calibrate semantic payload representation without crediting extra turns or lucky guesses. It is a deliberately small task family: exact tuple matching does not test broad scientific reasoning, planning, long-horizon coordination, or natural-language grounding.

Before any format comparison, require a receiver full-information capability gate on fresh tasks. Compare no message, correct target message, a target message deranged from another episode, a fixed-width shared tuple-rank code, and a centralized oracle; keep task schedule and prompts fixed. Candidate representation arms include optimized concise natural language, JSON, delimited tuples, and a compact code only if its setup/codebook cost is declared. Measure exact answer, semantic decoding, protocol adherence, actual serialized payload bytes, receiver-native input/output tokens, full prompt/completion costs, latency, and failure/retry counts. Equal-budget curves and cross-model transfer are required before a protocol claim.

The task manifest is not a preregistration. No model run, language comparison, scaling result, or superiority claim is included in this artifact.
