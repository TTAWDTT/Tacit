# Private Match v0.3: three-agent complementary evidence

This task extends Private Match from one complete-record sender and one
receiver to three agents: two source agents each know one coordinate of the
hidden target, and a receiver sees a shuffled table of all candidate pairs.
Neither source can identify the target by itself. Their two messages compose
to identify exactly one row. This is a model-free task definition and control;
it contains no claim that any message representation works better.

## Frozen task family

For a power of two `q >= 2`, the receiver sees the full Cartesian table
`[q] × [q]` with independently shuffled candidate order and candidate IDs.
The hidden target is uniform over the `q²` rows. Sender X sees only the
target's X coordinate; sender Y sees only its Y coordinate. The receiver sees
neither value. Role views and gold are serialized separately. Episode seeds
control the candidate shuffle/IDs and target draw through domain-separated
deterministic random streams; the seed is never placed in model-visible data.

Generate a small task shard without loading a model:

```powershell
python experiments/private_match_v0_3/generate_tasks.py `
  --output .cache/private_match_v0_3/q4 `
  --episodes 32 --seed 3000 --q 4
```

The task generator validates each episode before writing and publishes SHA-256
hashes for every role file in `manifest.json`. It refuses to overwrite an
existing shard unless `--force` is passed.

An offline fake-client integration test routes the two oracle coordinates to
the receiver over the SDK's loopback channel, then collects a sealed answer
for scoring. Both delivered messages appear in the wire ledger; the terminal
answer is counted as inference only and is not sent to another agent.

## Exact task and communication references

Let `X,Y` be independent uniform elements of `[q]`. Candidate row `(X,Y)` is
the target. The receiver's candidate table is the same in every condition and
the target index is uniform conditional on that table. Therefore the exact
Bayes accuracies are:

| Evidence visible to receiver | Exact Bayes accuracy |
| --- | ---: |
| No message | `1/q²` |
| X source only | `1/q` |
| Y source only | `1/q` |
| Both source values | `1` |

For a zero-error simultaneous protocol, each source must distinguish all `q`
possible values of its own coordinate. If two X values share an encoded
message, fix any Y value: the receiver would see the same two-message
transcript for two different target rows and could not answer both correctly.
The same argument applies to Y. Thus each source needs at least `log₂(q)` bits
in a fixed-width binary code, for a total of `2 log₂(q) = log₂(q²)` payload
bits. The two coordinates attain this bound. This is a bound for this
finite, noiseless, simultaneous task model; it says nothing about LLM token
cost, decoding errors, instructions, or setup cost.

### Exact fixed-width bit-budget frontier

`bit_frontier.py` extends the zero-error endpoint to every integer total
payload-bit budget. Let sender X and sender Y use `b_x` and `b_y` fixed-width
bits, with `b_x + b_y <= B`; the fixed turn schedule identifies the source
slot. Under the task assumptions above, the optimal Bayes accuracy is

\[
  A^*(q,B) = \max_{b_x+b_y\le B}
  \frac{\min(q,2^{b_x})\min(q,2^{b_y})}{q^2}.
\]

Since `q=2^w`, this reduces exactly to

\[
  A^*(q,B)=\frac{2^{\min(B,2w)}}{q^2}.
\]

**Proof.** Each deterministic sender maps coordinate values into message
classes. A received pair identifies the Cartesian product of one X-class and
one Y-class. If these classes have sizes `a` and `b`, the receiver has `ab`
equally likely rows and can be correct with probability `1/(ab)`. Averaging
over the `ab` target rows in that cell contributes exactly `1/q²`; summing
over nonempty cells gives `K_x K_y/q²`, where `K_x` and `K_y` are the numbers
of distinct message classes. A `b_x`-bit fixed-width message has at most
`min(q,2^{b_x})` classes, and likewise for Y. These bounds are attained by
partitions with that many classes. Maximizing their product under the integer
bit budget yields the displayed frontier. Randomized encoders cannot improve
it: conditioning on their randomness gives a mixture of deterministic encoder
pairs, whose average accuracy is no greater than the best deterministic pair. ∎

For `q=4`, the frontier is `1/16, 1/8, 1/4, 1/2, 1` at total payload
budgets `0` through `4` bits. This assumes simultaneous fixed-width payloads,
uniform independent coordinates, a complete receiver table, and a free fixed
schedule. It excludes envelope bytes, prompt/codebook cost, tokenization,
inference compute, and model errors. It is an ideal communication reference,
not an LLM performance result. Reproduce it with:

```powershell
python -m experiments.private_match_v0_3.bit_frontier --q 4 --max-bits 8
```

## Frozen feasibility protocols and runner (model calls not run)

`protocols.py` freezes prompt revision `pmt3-prompts-1` and four representation
IDs: `concise_nl`, `compact_kv`, `strict_json`, and `fixed_binary`. The parser
accepts only canonical compact KV/JSON strings and exactly `log2(q)` binary
digits. Natural-language format and semantic fidelity are recorded as unknown
until a separate blinded judging procedure is preregistered. `concise_nl` is a
feasibility arm; it has not been optimized and must not be described as the
strong natural-language baseline.

The runner is dry-run by default:

```powershell
python -m experiments.private_match_v0_3.runner --condition both_sources `
  --protocol compact_kv --episodes 4 --q 4
```

Execution requires the explicit `--execute` flag, a passing recent local
resource preflight, endpoint/model/tokenizer identifiers, and a disjoint
perfect full-information calibration ledger for evaluation batches. It uses
loopback endpoints only, disables redirects, caps requests at 30 seconds, and
rejects batches above 12 planned calls before opening output or contacting an
endpoint. Full-information calibration is run separately; its receiver sees
both exact coordinates and only its sealed final submission is scored. The
no-message arm performs no inter-agent transmission and still records one
receiver inference call.

The exact endpoint/model and tokenizer settings remain pending. No inference
has been authorized or run by this preregistration. The default run path is a
dry-run summary and makes no model request.

### Analyze completed ledger batches

After collecting every evaluation arm into its own JSONL file, pass all of
those files to the v0.3 report command:

```powershell
python -m experiments.private_match_v0_3.report `
  .cache/private_match_v0_3/no_message.jsonl `
  .cache/private_match_v0_3/sender_x_only.jsonl `
  .cache/private_match_v0_3/sender_y_only.jsonl `
  .cache/private_match_v0_3/both_nl.jsonl `
  .cache/private_match_v0_3/both_kv.jsonl `
  .cache/private_match_v0_3/both_json.jsonl `
  .cache/private_match_v0_3/both_binary.jsonl `
  --output .cache/private_match_v0_3/report.json
```

The report reconstructs each task from its seed, re-scores the exact answer,
checks prompt revision, schedule, route, transmitted text-size metadata, parser
diagnostics, and model-call counts, then delegates cost aggregation, paired
uncertainty, and Pareto-frontier calculations to the repository's shared
`tlu.costs.v3` tools. It keeps calibration and evaluation strata separate,
preserves model strata, and reports decoder alignment explicitly. Natural-
language format/fidelity is marked missing by design until an independently
frozen semantic-judging method exists. No diagnostic pools distinct task/model
strata.

## Planned model comparison (not run)

The future paired comparison should retain every generated episode and include
no-message, X-only, Y-only, and full-information diagnostic conditions. Only
conditions where both source agents send are representation comparisons. Keep
the task, fixed schedule (`X → receiver`, then `Y → receiver`), model
population, decoding, and receiver answer contract fixed while comparing:

- concise natural language (feasibility only until a disjoint optimization
  stage establishes a stronger baseline);
- compact key-value text;
- strict JSON;
- a fixed-width binary code with its schema/codebook setup charged;
- any candidate representation discovered in later work.

The two source messages must be separately attributable in the receiver's
transcript and cost ledger. Compare task success against both delivered
application bytes and complete model input/output tokens. A shorter channel
payload is not a win if the complete inference cost or error rate erases the
gain. Preserve syntax validity, coordinate fidelity, and final task success as
separate outcomes.

Use a seed-disjoint full-information calibration gate, but never filter
evaluation episodes by calibration or by an evaluation arm's outcome. At the
default `q=4`, the receiver table has only 16 rows and the ideal references are
6.25% without messages, 25% with one source, and 100% with both. The reference
does not guarantee that a language model can parse the task. Any model run
must pass the project's frozen host-resource and automatic-stop gates first;
this README does not authorize inference by itself.

## Limits

The benchmark is deliberately small and algorithmic. It tests whether two
independent messages compose under controlled information partitioning; it
does not measure long-horizon planning, strategic disclosure, noisy transport,
adaptive routing, or general scientific reasoning. The complete Cartesian
structure gives clean exact priors and a communication lower bound, but also
makes the task easier than realistic collaboration. Treat it as a three-agent
calibration family that complements, rather than replaces, HiddenBench,
Pointer Chasing, and richer task families.
