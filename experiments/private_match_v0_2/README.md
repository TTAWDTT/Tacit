# Private Match v0.2: local model pilot runner

This milestone turns the role-separated calibration task into a runnable,
paired communication experiment. It compares no-message, concise natural
language, compact key-value text, JSON, ordered tuple, a fixed hex-nibble code,
and a sender-selected form. The hex condition is a task-specific coded baseline, not a claim of a
new language. Conditions share generated episode IDs, candidate tables, and
hidden target records.

The runner delivers the sender's exact response through Tacit's measured
length-prefixed loopback TCP channel. The recorded boundary counts the UTF-8
JSON payload field, envelope, length prefix, and one-byte callback
acknowledgement at the application layer; it excludes TCP/IP and link-layer
headers. Receiver message-only token counts remain null because model-reported
prompt usage includes the receiver's task context.

Each row also preserves the generated message and answer in a diagnostics
object, reports syntax validity and exact semantic fidelity for mechanically
decodable formats, and keeps receiver success separate. Natural-language and
sender-selected semantic fidelity are left null instead of being guessed.
This makes failed messages auditable while keeping the synthetic target out of
the scorer fields.

The execution path is deliberately staged. A batch has a hard ceiling of 12
planned model requests and each HTTP request has a 30-second timeout. At four
episodes, run the no-message baseline and one message condition together (12
requests), then run one additional message condition per batch (8 requests).
Use the same `--seed` and task parameters for every batch, write to a distinct
output path, and pass every ledger to the report. Episode IDs derive from the
generation seed, so the report can pair conditions collected in separate runs.
The default dry-run previews all eight conditions (32 rows and 56 planned
calls); execution rejects that oversized batch before creating an output file
or calling an endpoint. The `full_information` stage must pass for all four
episodes before any protocol-comparison condition may execute.

For the first batch, run the preflight while the endpoint port is free. After
it passes, start the local endpoint and submit the report to the runner:

```powershell
./experiments/emergent_ood_v0_3/resource_preflight.ps1 -Ports 8000
python experiments/private_match_v0_2/runner.py --execute --protocols full_information --episodes 4 --seed 20260929 --resource-preflight .cache/emergent_ood_v0_3/resource_preflight.json --output .cache/private_match_v0_2/full_information.jsonl
```

After all four full-information answers pass, rerun the resource preflight and
use the private capability ledger for the paired no-message and first
representation batch:

```powershell
./experiments/emergent_ood_v0_3/resource_preflight.ps1 -Ports 8000
python experiments/private_match_v0_2/runner.py --execute --protocols no_message concise_nl --episodes 4 --seed 20260929 --resource-preflight .cache/emergent_ood_v0_3/resource_preflight.json --capability-ledger .cache/private_match_v0_2/full_information.jsonl --output .cache/private_match_v0_2/baseline_concise.jsonl
```

An additional protocol uses the same seed and episode count but a new ledger:

```powershell
./experiments/emergent_ood_v0_3/resource_preflight.ps1 -Ports 8000
python experiments/private_match_v0_2/runner.py --execute --protocols compact_kv --episodes 4 --seed 20260929 --resource-preflight .cache/emergent_ood_v0_3/resource_preflight.json --capability-ledger .cache/private_match_v0_2/full_information.jsonl --output .cache/private_match_v0_2/compact_kv.jsonl
```

Repeat that additional-condition form for each registered protocol. Before
every batch, stop the local endpoint, run the read-only resource preflight with
every configured endpoint port while those ports are free, then restart the
endpoint before running the CLI. `--execute` fails closed on a
missing, rejected, stale, other-host, port-incomplete, or threshold-mismatched
report before creating output or initializing an endpoint client. Every
non-full-information run also requires a passing same-task/model/tokenizer
capability ledger before output creation. For separate
sender and receiver endpoints, include both ports, e.g. `-Ports 8000,8001`,
and use the same report path in `--resource-preflight`. Never use `--force` to
reuse a ledger when combining staged runs.

The full-information ledger contains verbatim model answers and candidate
ordering diagnostics. Keep it local in the ignored `.cache` directory; do not
publish it with experiment artifacts.

Run the paired fidelity/task/cost analysis after collecting the ledgers:

```powershell
python experiments/private_match_v0_2/report.py .cache/private_match_v0_2/baseline_concise.jsonl .cache/private_match_v0_2/compact_kv.jsonl --output .cache/private_match_v0_2/report.json
```

Task parameters contain fixed task-scale settings; episode seeds are stored
under diagnostics so independent episodes pool into the same task/model
stratum and paired bootstrap intervals use the full episode set. The report
CLI accepts one or more ledgers and rejects duplicate condition/episode rows.

## Safe behavior

The CLI is dry-run by default and never launches a model process. Model
requests require `--execute`, explicit model IDs and tokenizer IDs, a fresh
passing same-host resource report, and for protocol comparisons a 100%
full-information capability ledger. Both gates validate before output or
endpoint-client creation. `full_information` must be run as its own first
stage. The CLI uses a loopback OpenAI-compatible endpoint.
Cloud endpoint URLs and HTTP redirects
are rejected. This runner was implemented and tested with fake clients only;
no model was run.

Before executing inference, satisfy the resource thresholds and procedure in
[`../index_v0_3/preregistration.json`](../index_v0_3/preregistration.json).
The prior local-model gate failed, so this artifact does not authorize a pilot
run by itself.

From the repository root, inspect the plan without making any model calls:

```powershell
python experiments/private_match_v0_2/runner.py
```

Once the separately documented resource and capability gates have passed and
a local endpoint is already running, configure `TLU_BASE_URL` (or separate
`TLU_SENDER_BASE_URL` and `TLU_RECEIVER_BASE_URL`), `TLU_SENDER_MODEL`,
`TLU_RECEIVER_MODEL`, `TLU_SENDER_TOKENIZER_ID`, and
`TLU_RECEIVER_TOKENIZER_ID`, then add `--execute`. The separate loopback
endpoints support heterogeneous local model servers. Sender and receiver token
counts stay in separate tokenizer strata. Results write as `tlu.costs.v3`
episode rows. An existing output is protected unless
`--force` is explicit; `--force` overwrites the old ledger rather than
appending duplicate episode IDs.

## Limits

This is a small task with exact matching and a strong code baseline. It tests
communication utility under hidden information, but not long-horizon planning,
heterogeneous transfer, robustness, learned protocols, or broad generalization.
The `autoform` condition is prompt-based sender-selected formatting and is not
an optimized protocol search. See [`preregistration.json`](preregistration.json)
for the paired design and claim limits.
