# Private Match v0.2: local model pilot runner

This milestone turns the role-separated calibration task into a runnable,
paired communication experiment. It compares no-message, concise natural
language, JSON, ordered tuple, a fixed hex-nibble code, and a sender-selected
form. The hex condition is a task-specific coded baseline, not a claim of a
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

Run the paired fidelity/task/cost analysis after collecting a ledger:

```powershell
python experiments/private_match_v0_2/report.py .cache/private_match_v0_2/pilot.jsonl --output .cache/private_match_v0_2/report.json
```

Task parameters contain fixed task-scale settings; episode seeds are stored
under diagnostics so independent episodes pool into the same task/model
stratum and paired bootstrap intervals use the full episode set.

## Safe behavior

The CLI is dry-run by default and never launches a model process. Model
requests require `--execute`, explicit model IDs and tokenizer IDs, and a
loopback OpenAI-compatible endpoint. Cloud endpoint URLs and HTTP redirects
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

Once the separately documented resource gate has passed and a local endpoint
is already running, configure `TLU_BASE_URL`, `TLU_SENDER_MODEL`,
`TLU_RECEIVER_MODEL`, `TLU_SENDER_TOKENIZER_ID`, and
`TLU_RECEIVER_TOKENIZER_ID`, then add `--execute`. Sender and receiver token
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
