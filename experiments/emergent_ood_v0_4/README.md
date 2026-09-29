# Emergent OOD transfer fixture v0.4

This standard-library-only artifact provides a higher-order meaning split and keyed, role-separated communication episodes. It is not a model runner, protocol, benchmark result, or language-performance claim.

## Meaning split

The default universe is `shape × color × quantity × texture`, with four values per attribute (`4^4 = 256` meanings). For each seed, the generator deterministically orders each axis using domain-separated SHA-256 hashes. A meaning is held out when the sum of its four permuted ranks is `0 mod 4`.

This yields 64 held-out four-way combinations and 192 training combinations. Every value combination of one, two, or three attributes appears in training. Exhaustive enumeration confirms 16 unary assignments, 96 pair assignments, and 256 triple assignments. The split digest and lower-order coverage are recomputed by the validator.

```powershell
python experiments/emergent_ood_v0_4/split.py --seed 17 --output .cache/emergent_ood_v0_4/seed_17.json
```

Omitting `--output` prints JSON to stdout. File output is rejected unless it stays inside the project. The split seed is public metadata, not a secret or a task key.

## Keyed communication episodes

Create a local evaluator-only 256-bit key once, then generate role ledgers:

```powershell
python -m experiments.emergent_ood_v0_4.episodes keygen --key .cache/emergent_ood_v0_4/evaluator.key
python -m experiments.emergent_ood_v0_4.episodes generate --key .cache/emergent_ood_v0_4/evaluator.key --seed 17 --task-seed 0 --k 4 --sets-per-stage 16 --output-dir .cache/emergent_ood_v0_4/calibration-17
python -m experiments.emergent_ood_v0_4.episodes generate --key .cache/emergent_ood_v0_4/evaluator.key --seed 23 --task-seed 0 --k 4 --sets-per-stage 16 --output-dir .cache/emergent_ood_v0_4/evaluation-23
```

The private key and generated episodes stay under ignored `.cache/`. HMAC domain-separated streams allocate 16 of the 64 held-out meanings to validation and the other 48 to test; training targets come only from the 192 training meanings. Each candidate set has exactly `k` episodes, one per candidate target, with a target-independent candidate table and ordering. Therefore the no-message Bayes accuracy is exactly `1/k` for every set. The manifest reports candidate co-occurrence coverage both on observed meanings and against the stage's full target support.

Output consists of separate `sender_{train,validation,test}.jsonl`, `receiver_...`, and evaluator-only `gold_...` ledgers. Sender rows contain only a private attribute tuple; receiver rows contain only candidate IDs and tuples. Model-facing rows have no episode, target, meaning, candidate-set, or stage identity. A trusted runner must preserve row order to align the ledgers. Never give a model access to the output directory, manifest, key, or gold ledger. Manifest file hashes support artifact integrity checks; they do not prove that evaluation was conducted correctly.

The default `sets-per-stage=16` is a small plumbing fixture (64 targets per stage), not a sample-size recommendation. Candidate sets and meanings are nested/repeated observations. Any inferential run needs a preregistered sample plan, more clusters, tool-boundary leakage checks, and a passing model-resource gate. Key and output paths outside the project are rejected.

## Evaluator runner

The runner uses the repository's protocol-neutral runtime and leaves model server startup to the operator. It verifies the split digest, all nine ledger hashes, role separation, target membership, and balanced candidate sets before any model request. It selects complete balanced candidate-set clusters, omits all evaluator IDs/stage labels from prompts, uses one sender-to-receiver turn followed by a sealed final answer, and writes raw results plus an evaluator-only manifest under the project.

Available conditions are `no_message`, `full_information`, `natural_language`, `autoform`, `json`, `symbolic`, `shared_protocol_card`, and `usage_only_transfer`. The English arm is a plain baseline, not development-optimized natural language. `autoform` is an AutoForm-style prompt-selected open-format baseline adapted from the existing Private Match prompt: the sender chooses a concise medium and the receiver gets no fixed grammar. We score exact end-task selection and complete measured cost; semantic parsing and fixed-format validity remain unknown, so this is not labeled a new language. It does not claim an exact reproduction of the published AutoForm setup. JSON is fixed structured text. The symbolic condition is a fixed four-axis digit code and a strong handcrafted control; it is not Tacit's learned language. `shared_protocol_card` evaluates a previously frozen sender/receiver instruction artifact using this JSON schema: `{"schema":"tlu.shared_protocol_card.v1","protocol_id":"...","sender_instruction":"...","receiver_instruction":"..."}`. The runner hashes and applies the card unchanged. The separate `induce_protocol_cards.py` tool can propose compositional-symbolic or plain-English instruction candidates from training meanings; candidates remain hypotheses until evaluated, and there is no real-model induction result yet. The receiver must return an exact candidate ID. Malformed/truncated answers remain failures. Reported token totals are `null` when any call lacks provider usage, with per-call missingness preserved. The run records tokenizer/model IDs, generated versus delivered bytes, the full loopback application envelope, call/service/wall costs, sender-format audits, and an exact small-batch conflict-graph coloring floor. `--wire-budget-bytes` caps the complete serialized application message for the episode (default 4096 bytes); the budget and actually delivered bytes are both recorded. A sender completion that exceeds the cap is not delivered, but its model call and generated output remain charged. This is a wire-byte budget, not a model-token budget. Temperature is fixed at zero; sender/receiver completion caps are 160/48 tokens.

### Usage-only protocol transfer

The `usage_only_transfer` condition tests in-context receiver onboarding. The sender receives the sender half of a frozen protocol card. A new receiver gets its candidate table and train-only meaning/message exemplars, but receives neither the card nor its decoder instruction. It must infer the convention from use. This is a one-message transfer comparison, not online adaptation or evidence that the protocol is compositional.

Supply the frozen card with `--protocol-card` and a `--usage-examples` JSON artifact. The artifact has exactly these fields: `schema` (`tlu.usage_examples.v1`), `protocol_id`, `protocol_card_sha256`, `training_split_sha256`, `training_episode_manifest_sha256`, `acquisition`, and `examples`. Each example has exactly `meaning_id`, `meaning`, and `message`. The `acquisition` object has `method` (`model_generated`, `human_authored`, or `programmatic`), `model_id`, `tokenizer_id`, `generation_calls`, `input_tokens`, `output_tokens`, `service_seconds`, and `wall_seconds`; unknown measurements are `null`. For model-generated artifacts, `generation_calls` counts all attempts—including failed-format retries—and must be at least the number of successful examples.

Build the pairs only from training meanings under the frozen sender convention. The runner binds their claims to the exact card, split, and episode-manifest hashes, and checks every meaning against the sender training ledger. It strips IDs and provenance fields from the receiver prompt. These hashes validate artifact binding, not authorship: preserve the generation trace and audit it before making a transfer claim. Supply `--usage-reuse-horizon H`, the preregistered number of evaluation episodes that will reuse this artifact. Each result carries the artifact in the standard `setup` array, so `tools.cost_report`, `tools.frontier_report`, and `tools.paired_report` deduplicate it by hash and amortize its one-time payload bytes, model calls, service/wall time, and tokenizer-indexed generation tokens over `H`. Setup bytes measure the serialized example payload for one conceptual onboarding transfer; HTTP framing and actual artifact transport are not measured by this runner. The run manifest also preserves acquisition input/output totals. `usage_example_bytes_per_receiver_request` reports repeated prompt context; provider input tokens already include it. Setup-distribution bytes and repeated inference-prompt costs are distinct accounting dimensions.

Example artifact (replace each placeholder with the exact digest/ID and observed message):

```json
{
  "schema": "tlu.usage_examples.v1",
  "protocol_id": "frozen-compositional-code-v1",
  "protocol_card_sha256": "<sha256 of the exact card file>",
  "training_split_sha256": "<split digest>",
  "training_episode_manifest_sha256": "<sha256 of the exact episode manifest.json>",
  "acquisition": {
    "method": "model_generated",
    "model_id": "local-sender-id",
    "tokenizer_id": "local-sender-tokenizer-revision",
    "generation_calls": 1,
    "input_tokens": null,
    "output_tokens": null,
    "service_seconds": null,
    "wall_seconds": null
  },
  "examples": [
    {"meaning_id": "<train meaning id>", "meaning": {"shape": "circle", "color": "amber", "quantity": "one", "texture": "smooth"}, "message": "<observed training message>"}
  ]
}
```

Run it with `--conditions usage_only_transfer --protocol-card ... --usage-examples ... --usage-reuse-horizon 100`. It also requires the independent receiver capability ledger and fresh resource preflight used by other message conditions. It cannot run in the reserved `--stage train --conditions full_information` calibration batch.

#### Generate examples from training meanings

`generate_usage_examples.py` creates the artifact by querying the frozen card's sender instruction once per selected training meaning. Selection is reproducible and keyed, each sender prompt contains one private tuple and no evaluator IDs, and every completed request is checkpointed. The hard cap is 12 total request attempts per batch, including failed attempts and retries across resumes; a retry consumes a slot. Select no more than 12 examples per batch. Dry-run is the default:

```powershell
python -m experiments.emergent_ood_v0_4.generate_usage_examples `
  --input-dir .cache/emergent_ood_v0_4/evaluation-23 `
  --split-seed 23 `
  --task-key .cache/emergent_ood_v0_4/evaluator.key `
  --protocol-card .cache/emergent_ood_v0_4/frozen-card.json `
  --examples 4 `
  --output-dir .cache/emergent_ood_v0_4/usage-examples-4
```

Execution is separate and requires a fresh passing resource preflight, a locally running loopback sender endpoint, and explicit model/tokenizer IDs. The script never starts a service or loads a model itself:

```powershell
python -m experiments.emergent_ood_v0_4.generate_usage_examples `
  --input-dir .cache/emergent_ood_v0_4/evaluation-23 `
  --split-seed 23 `
  --task-key .cache/emergent_ood_v0_4/evaluator.key `
  --protocol-card .cache/emergent_ood_v0_4/frozen-card.json `
  --examples 4 `
  --output-dir .cache/emergent_ood_v0_4/usage-examples-4 `
  --model local-sender-id `
  --tokenizer-id local-sender-tokenizer-revision `
  --resource-preflight .cache/emergent_ood_v0_4/resource_preflight.json `
  --execute
```

The runner emits `usage-examples.json`, a `generation-trace.jsonl` with raw successes and failed attempts, and a `generation-manifest.json` with per-attempt prompt/completion hashes and usage. Request errors and malformed completions are checkpointed and charged if a later retry succeeds. An interrupted batch can resume only with an identical selection/card/model/endpoint configuration and a newly passing preflight; use `--resume`. Keep the task key and generated role artifacts in ignored `.cache/`. This utility does not prove the card is leak-free or that the sender follows it; audit the saved trace and score the examples before claiming protocol acquisition.

### Independent receiver capability screen

Before any message condition (`natural_language`, `autoform`, `json`, `symbolic`, `shared_protocol_card`, or `usage_only_transfer`) executes, the same receiver must pass a 12/12 full-information screen on three complete candidate sets from the **training meaning partition of a separate calibration split seed**. Calibration and evaluation bundles must use different split seeds but the same task key, task seed, candidate count, attribute names/values, receiver model/tokenizer, and model population. Supply the calibration bundle path and seed separately; the runner verifies its hashes and manifest, exact calibration episode/candidate identities, exact answers, and model stratum. It also verifies calibration episode IDs do not occur in evaluation. Meaning tuples may recur across independently generated split seeds: the local completion adapter sends each call as a fresh request with no shared conversation history, and calibration answers are never inserted into evaluation prompts. The screen is a strict model-eligibility check for basic task execution; it does not establish performance on unseen compositions or estimate a stable success rate. Evaluation split seeds and rows are never retained or removed according to their own full-information outcomes. Full-information runs on validation/test remain descriptive controls and cannot replace the independent train-only screen.

The training-stage command is intentionally limited to this 12-call calibration batch:

```powershell
python -m experiments.emergent_ood_v0_4.runner --input-dir .cache/emergent_ood_v0_4/calibration-17 --split-seed 17 --stage train --sets 3 --conditions full_information --output .cache/emergent_ood_v0_4/runs/capability.jsonl --execute --resource-preflight .cache/emergent_ood_v0_4/resource_preflight.json
```

Every message-condition batch then supplies that ledger and the independent calibration bundle. The evaluation split seed must differ from the calibration seed:

```powershell
python -m experiments.emergent_ood_v0_4.runner --input-dir .cache/emergent_ood_v0_4/evaluation-23 --split-seed 23 --stage validation --conditions natural_language --capability-ledger .cache/emergent_ood_v0_4/runs/capability.jsonl --capability-input-dir .cache/emergent_ood_v0_4/calibration-17 --capability-split-seed 17 --output .cache/emergent_ood_v0_4/runs/validation-natural-language.jsonl --execute --resource-preflight .cache/emergent_ood_v0_4/resource_preflight.json
```

Use `--set-offset` to process later candidate sets without reusing the first batch. For example, the validation fixture's 16 sets can be divided into offsets `0,1,...,15` with `--sets 1`; each single-condition message batch uses eight calls and stays under the 12-call cap. Give every batch a unique output path, then concatenate only the result JSONL rows for analysis. The output manifests preserve each batch's offset; one split seed remains one independent inference cluster regardless of the number of candidate sets processed.

The current local resource gate has not passed; these commands document the workflow and are not permission to launch inference. Run the dry-run form by omitting `--execute` before a future model batch.

Dry-run is the default and makes no model request:

```powershell
python -m experiments.emergent_ood_v0_4.runner --input-dir .cache/emergent_ood_v0_4/episodes --stage validation --conditions no_message natural_language
```

After a protocol has been developed on training episodes and frozen, evaluate it with `--conditions shared_protocol_card --protocol-card .cache/emergent_ood_v0_4/frozen-card.json`. Validation can select development choices; freeze the card before touching the test stage.

Execution requires a passing resource report no older than five minutes, obtained before starting the configured local endpoints. The runner only accepts loopback URLs and does not start the service:

```powershell
$env:TLU_SENDER_MODEL = "local-sender-model"
$env:TLU_RECEIVER_MODEL = "local-receiver-model"
$env:TLU_SENDER_TOKENIZER_ID = "sender-tokenizer-revision"
$env:TLU_RECEIVER_TOKENIZER_ID = "receiver-tokenizer-revision"
python experiments/emergent_ood_v0_3/resource_preflight.ps1 -Output .cache/emergent_ood_v0_4/resource_preflight.json -Ports 8000,8001
# Start the local model endpoint only after a passing report, then:
python -m experiments.emergent_ood_v0_4.runner --input-dir .cache/emergent_ood_v0_4/episodes --stage validation --conditions no_message natural_language --sender-base-url http://127.0.0.1:8000/v1 --receiver-base-url http://127.0.0.1:8001/v1 --execute --resource-preflight .cache/emergent_ood_v0_4/resource_preflight.json
```

During execution the runner atomically replaces `<output>.checkpoint.json` after each completed episode-condition result. If the process stops early, rerun the identical command with `--resume` and a newly passing resource preflight; the runner checks hashes for the task bundle, protocol card, capability inputs, code, model/tokenizer configuration, and batch settings, then verifies checkpoint rows form the exact requested prefix before reusing them. A mismatch is rejected before creating endpoint clients. The checkpoint includes evaluator-only labels and traces, so keep it inside ignored `.cache/` and never expose it to either model. The checkpoint is removed after the final result and manifest are written. Resume recovers completed rows, not a partially completed model request; an interrupted request may be repeated.

To trace an application-byte efficiency frontier, rerun all compared conditions on the same frozen episode bundle, model/tokenizer pair, and protocol at predeclared caps, writing each cap to its own result file. For example, add `--wire-budget-bytes 256` and an output path such as `frontier-256.jsonl`; repeat at the other caps only after the resource gate passes. `tools.frontier_report` retains each budget as a separate task stratum and compares conditions only within a matched budget; `tools.paired_report` supplies cluster-paired uncertainty for those same-budget contrasts. Do not compare runs that changed prompts, protocol cards, episode bundles, or model population. A byte frontier does not substitute for a separate tokenizer-token or inference-cost axis.

The runner computes the exact chromatic number only for batch graphs with at most 20 observed meanings; larger graphs are marked non-exact rather than approximated as a theorem. Message conditions require the verified independent train-stage capability ledger before any endpoint client is constructed. Validation results are development data and test-stage protocols must be frozen before they are run. The 12-call hard batch cap is a feasibility safeguard, not a confirmatory-study design. No protocol superiority claim follows from this runner.

Run focused model-free checks:

```powershell
python -m unittest discover -s tests -p "test_emergent_ood_v04_*.py" -v
```

## Scope and research role

The task asks a sender to communicate a private meaning so a receiver can select its match from a balanced candidate set. Training exposes lower-order combinations while validation/test targets are disjoint held-out compositions. The proposed CLSR-inspired method may induce a reusable dialect from training episodes, select/profile only on validation, then freeze before final evaluation.

The artifact provides a strict task scorer and local runner, plus a model-free tested gate for independent receiver-capability calibration. A train-only candidate-card induction runner now exists, but it has not been executed; there is still no learned dialect, cross-model transfer experiment, or protocol-performance evidence. The [`select_protocol_frontier.py`](select_protocol_frontier.py) turns complete validation runs of at least two supplied cards into a hash-bound Pareto shortlist; it never chooses a single winner and reads only validation bundle roles. It can verify induction manifests, bind the selected cards to their hashes, and include measured generator calls/tokens/service time in separately reported setup costs. Without those manifests, discovery cost remains explicitly unknown. The selector still cannot prove that a card's meaning is free of hidden-target leakage. Canonical-label fidelity for English is a conservative string audit, not a general semantic judge. See the [CLSR transfer experiment design](../../research/CLSR_TRANSFER_EXPERIMENT_DESIGN_V0_1.md), [CLSR prior audit](../../research/CLSR_AUDIT_V0_1.md), and [GlossoGen boundary audit](../../research/GLOSSOGEN_AUDIT_V0_1.md).

### Propose candidate protocol cards from training data

The induction utility defaults to dry-run. It chooses examples only from the split's lower-order training partition, using the private task key to order meaning IDs. The generator sees the attribute vocabulary and sampled training examples; it does not receive validation/test tuples, gold candidate labels, split seed, or task key. The output is a set of instruction-card hypotheses, not evidence of an emergent language or useful protocol. Select `--protocol-family compositional_symbolic` for a compact symbolic-code hypothesis set or `--protocol-family plain_english` for semantically explicit ordinary-English message instructions. Validation can select among either family, but this is a one-shot candidate workflow rather than OPRO-style iterative optimization; all candidate-generation and validation costs must be counted.

```powershell
python -m experiments.emergent_ood_v0_4.induce_protocol_cards `
  --split-seed 23 `
  --task-key .cache/emergent_ood_v0_4/evaluator.key `
  --output-dir .cache/emergent_ood_v0_4/induced-cards
```

To propose a plain-English instruction family for a stronger natural-language prompt control, use a separate output directory and select the family explicitly:

```powershell
python -m experiments.emergent_ood_v0_4.induce_protocol_cards `
  --split-seed 23 `
  --task-key .cache/emergent_ood_v0_4/evaluator.key `
  --protocol-family plain_english `
  --output-dir .cache/emergent_ood_v0_4/plain-english-candidates
```

Execution is a separate, explicit `--execute` operation. It requires an operator-started loopback endpoint, model and tokenizer IDs, and a fresh passing resource report for port 8002; it never starts a service itself. Keep the successful `induction-manifest.json` and list its project-relative path in the candidate spec's optional `induction_manifests` array. The selector checks each induced card's bytes and hash against both the trace and evaluated candidate, and reports generator setup separately from validation-selection setup. It amortizes the measured setup values over the declared reuse horizon but does not treat prompt/completion file bytes as network-wire bytes.

### Freeze a validation-selected protocol frontier

Use only training information to propose and freeze every candidate card first, then run each card across the **same complete validation episode set**, at the same byte cap and model/tokenizer population. With the default 16 validation sets, use 16 one-set batches (`--sets 1 --set-offset 0` through `15`) per card to stay under the 12-call limit. Preserve each JSONL ledger and its runner-written `.manifest.json` sidecar.

Create a project-local candidate spec, for example `.cache/emergent_ood_v0_4/protocol-candidates.json`:

```json
{
  "schema": "tlu.emergent-ood-protocol-frontier-candidates.v1",
  "input_dir": "episodes",
  "split_seed": 23,
  "reuse_horizon_evaluation_episodes": 1000,
  "induction_manifests": ["induced-cards/induction-manifest.json"],
  "candidates": [
    {
      "protocol_id": "candidate-a-v1",
      "card": "cards/candidate-a.json",
      "validation_ledgers": [
        "runs/a-offset-0.jsonl",
        "runs/a-offset-1.jsonl"
      ]
    },
    {
      "protocol_id": "candidate-b-v1",
      "card": "cards/candidate-b.json",
      "validation_ledgers": [
        "runs/b-offset-0.jsonl",
        "runs/b-offset-1.jsonl"
      ]
    }
  ]
}
```

Paths in the spec are relative to its directory. List every offset ledger needed to cover all validation sets for each candidate; the two entries shown are abbreviated examples, not complete coverage for the default 16-set bundle. Run the freeze step only after all candidate cards and validation runs are complete:

```powershell
python -m experiments.emergent_ood_v0_4.select_protocol_frontier `
  --spec .cache/emergent_ood_v0_4/protocol-candidates.json `
  --output .cache/emergent_ood_v0_4/protocol-freeze.json
```

The selector verifies card/result hashes, candidate-set offset coverage, paired episode IDs, exact validation-stage labels, matching byte cap, and matching model/tokenizer IDs. It deliberately opens only the bundle manifest and the three validation role files; absent or damaged test role files do not affect selection. Its Pareto objectives are exact validation success, complete application wire bytes, and—only when every call reports them—per-agent/model/tokenizer input and output tokens. It reports service time but excludes it from dominance because host load can vary. All candidate-validation inference is counted as selection setup and divided by the declared reuse horizon. When induction manifests are supplied, the selector verifies their split/task-key identity, prompt/completion hashes, model usage, and every evaluated card's content hash. It reports and amortizes discovery cost separately; missing provider token usage stays unknown, and artifact bytes are not mislabeled as network traffic. Without manifests, the report explicitly leaves discovery cost unaccounted. Keep the protocol-generation trace and all train-only inputs if an autonomously generated language is later claimed.
