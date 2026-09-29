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
neither value. Role views and gold are serialized separately. An evaluator-only
random 256-bit key drives domain-separated HMAC-SHA256 counter streams for
candidate order, candidate IDs, and target selection. Bounded draws use
rejection sampling to avoid modulo bias. Public episode seeds alone cannot
reconstruct a task. The key is not stored in the manifest or passed into model
context; keep the key file private and out of version control. The benchmark's
formal prior is ideal independent-uniform sampling; a fixed-key shard is a
deterministic pseudorandom realization under the HMAC PRF assumption.

Create the same key once for task generation, every runner shard, and reporting:

```powershell
python experiments/private_match_v0_3/generate_tasks.py `
  --create-task-key --task-key-file .cache/private_match_v0_3/task.key
```

Generate a small task shard without loading a model:

```powershell
python experiments/private_match_v0_3/generate_tasks.py `
  --task-key-file .cache/private_match_v0_3/task.key `
  --output .cache/private_match_v0_3/q4 `
  --episodes 32 --seed 3000 --q 4
```

The task generator validates each episode before writing and publishes SHA-256
hashes for every role file in `manifest.json`. It refuses to overwrite an
existing shard unless `--force` is passed. Shard episode IDs use the same
seed-derived canonical IDs as runner ledgers, so generated role views can be
joined directly to model-run records for the same key and seed.

`episode_id` and the public episode seed are evaluator metadata. Raw role
records retain IDs for joining ledgers, but the runner strips them with
`model_visible_view(...)`; custom integrations must do the same. The secret
task key stays evaluator-side and is never included in prompts. The manifest
contains only a short key identifier so runs made with different keys cannot
be accidentally pooled.

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

`protocols.py` freezes prompt/parser/context revision `pmt3-prompts-6` and seven
representation arms: `concise_nl`, `short_nl`, `autoform`, `compact_kv`,
`decimal_index`, `strict_json`, and `fixed_binary`. Six have deterministic
reference encoders. `autoform` is a separate prompt-selected open-format arm
adapted from Chen et al.'s AutoForm prompt and official-source audit: the model
chooses a concise structured medium or code at runtime. It has no fixed syntax
or deterministic encoder, so mechanical format validity and semantic fidelity
are recorded as unknown; exact receiver success and complete costs remain
scored. This is not a reproduction of the paper's HotpotQA benchmark. The
parser accepts canonical compact KV/JSON strings, exactly `log2(q)` binary
digits, canonical zero-based decimal indices, and the exact frozen sentences
for both English templates. Other English paraphrases remain unknown pending a
separate blinded semantic judge. The English baseline is selected from the two
frozen templates using development outcomes only; neither development result
is evaluation evidence.

The deterministic encoder can be used directly without a model:

```python
from experiments.private_match_v0_3.protocols import encode_coordinate_message

encode_coordinate_message("compact_kv", q=4, sender="sender_x", value="x0002")
# 'x=x0002'
encode_coordinate_message("fixed_binary", q=4, sender="sender_y", value="y0003")
# '11'
encode_coordinate_message("decimal_index", q=4, sender="sender_y", value="y0003")
# '3'
encode_coordinate_message("short_nl", q=4, sender="sender_y", value="y0003")
# 'y is y0003.'
```

All formats encode the same coordinate meaning; only their surface channel
representation changes. `decimal_index` is a task-aware shorthand control,
not a claim that decimal digits constitute a general-purpose language. Its
sender-slot and zero-based-index schema are setup information and must be
included in complete model-input cost. These functions are reference codecs
and test fixtures. `autoform` intentionally has no deterministic encoder; its
message representation is selected by the prompted model. Model conditions
still ask the LLM to emit and interpret each format.

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
dry-run summary and makes no model request. Before execution, create the key
above; the runner reads it from `--task-key-file` (defaulting to the same path).

The development shard is seeds 302000–302007. Run it in two four-episode
`both_sources` batches (12 model calls each), using the passing calibration
ledger and `--split development`; collect one ledger per English candidate.
Once both ledgers exist, select and freeze the candidate without any model call:

```powershell
python -m experiments.private_match_v0_3.select_nl_baseline `
  --task-key-file .cache/private_match_v0_3/task.key `
  --concise-ledger .cache/private_match_v0_3/development_concise_nl.jsonl `
  --short-ledger .cache/private_match_v0_3/development_short_nl.jsonl `
  --output .cache/private_match_v0_3/nl_selection.json
```

The selector verifies exact development IDs, pairing, task key and shape,
model/tokenizer population, and complete token, service-time, and serialized
transmission telemetry. It maximizes exact joint success, then minimizes the
per-tokenizer input-plus-output token vector in tokenizer-ID order, logical
UTF-8 payload bytes, and protocol ID. It never adds unlike tokenizer counts.
Its manifest hashes the ledgers, preregistration, and protocol implementation
and totals both candidates' calls, serialized payload/framing bytes, tokens by
tokenizer, service time, and wall time as optimizer setup. The preregistered
pilot amortization horizon is eight evaluation episodes; this is a pilot
accounting convention, not a deployment-lifetime estimate. The held-out
evaluation shard is seeds 304000–304007 and must not be used by the selector.

### Analyze completed ledger batches

After collecting every evaluation arm into its own JSONL file, pass all of
those files to the v0.3 report command:

```powershell
python -m experiments.private_match_v0_3.report `
  .cache/private_match_v0_3/no_message.jsonl `
  .cache/private_match_v0_3/sender_x_only.jsonl `
  .cache/private_match_v0_3/sender_y_only.jsonl `
  .cache/private_match_v0_3/both_concise_nl.jsonl `
  .cache/private_match_v0_3/both_short_nl.jsonl `
  .cache/private_match_v0_3/both_autoform.jsonl `
  .cache/private_match_v0_3/both_compact_kv.jsonl `
  .cache/private_match_v0_3/both_decimal_index.jsonl `
  .cache/private_match_v0_3/both_strict_json.jsonl `
  .cache/private_match_v0_3/both_fixed_binary.jsonl `
  --task-key-file .cache/private_match_v0_3/task.key `
  --nl-selection-manifest .cache/private_match_v0_3/nl_selection.json `
  --output .cache/private_match_v0_3/report.json
```

Supplying the selection manifest is required for a complete selected-English
efficiency comparison. The report validates its hashes, key, prompt revision,
development IDs, model/tokenizer signature, and evaluation coverage. It charges
the one-time selector's serialized channel bytes and per-tokenizer model tokens
as a setup artifact on the selected English arm, and publishes raw plus
eight-episode-amortized calls, bytes, tokens, service time, and wall time.
Fixed setup is shown separately from episode bootstrap uncertainty. Without the
manifest, both English candidate points are omitted from the Pareto frontier
instead of being treated as zero-setup.

The report reconstructs each task from its seed and evaluator key, re-scores
the exact answer, checks prompt revision, schedule, route, transmitted text-size metadata, parser
diagnostics, and model-call counts, then delegates cost aggregation, paired
uncertainty, and Pareto-frontier calculations to the repository's shared
`tlu.costs.v3` tools. It also attaches the exact Bayes controls and ideal
fixed-width bit-budget curve separately for each q in the input. It keeps
calibration and evaluation strata separate,
preserves model strata, and reports decoder alignment explicitly. Natural-
language paraphrase fidelity and AutoForm free-format fidelity are marked
missing by design; they are not mislabeled as malformed. No diagnostic pools
distinct task/model strata.

For exact ideal references under independent non-uniform coordinate priors,
`bit_frontier.py` also exposes
`optimal_nonuniform_success_probability(probabilities_x=..., probabilities_y=..., total_payload_bits=...)`.
It returns a `Fraction`; the assumptions and proof are in [Theory §15](../../docs/THEORY.md#15-exact-frontier-for-independent-non-uniform-coordinate-priors).
`optimal_nonuniform_codebook(probabilities=..., payload_bits=...)` constructs an
attaining one-coordinate fixed-width encoder and frozen MAP decoder. Its
`success_probability(evaluation_prior)` method quantifies exact transfer while
keeping both fixed. The method assumes that encoder and decoder already share
the codebook; discovery and setup costs are excluded.
`FixedWidthCodebook.worst_case_success_probability(prior, tv_radius=...)`
returns the exact worst-case success over all priors within the stated total
variation distance, with the encoder and decoder frozen.
The current episode generator still samples uniform coordinates, so this is a
theoretical calculator and future experiment control, not a claim about the
present generated shards.

### Exact prior-shift episode control (model-free)

`prior_shift.py` builds a deterministic, exactly stratified cohort for rational
independent target-coordinate priors. It uses real role-separated candidate
tables and scorer records, permutes target order with a keyed HMAC stream, and
compares a frozen training-prior codebook with a codebook retrained for the
evaluation prior at identical per-sender bit widths. The API enforces a maximum
cohort size before constructing episodes. This is an exact finite design, not a
random sample and not a population confidence interval.

For a fresh local task key, then the four-value example from Theory §15:

```powershell
python -m experiments.private_match_v0_3.generate_tasks `
  --create-task-key --task-key-file .cache/private_match_v0_3/task.key
python -m experiments.private_match_v0_3.prior_shift `
  --task-key-file .cache/private_match_v0_3/task.key `
  --train-x '1/2,1/4,1/8,1/8' --train-y '1/4,1/4,1/4,1/4' `
  --eval-x '1/8,1/8,1/2,1/4' --eval-y '1/4,1/4,1/4,1/4' `
  --bits-x 1 --bits-y 2
```

The exact cohort has 32 episodes. The frozen codebook scores 8/32 and the
evaluation-adapted codebook scores 24/32, matching their exact expected rates
of 1/4 and 3/4. Codebook/prior setup is free in this oracle control; no LLM
calls, tokenization, framing, or inference costs are included. Keep the task key
outside version control and model-visible context.

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
