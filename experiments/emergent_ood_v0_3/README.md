# Emergent OOD v0.3: staged local capability controls

This runner connects the corrected v0.2 role-separated task to local chat models. Its purpose is to find out whether the receiver can solve the task when given the answer, how it behaves with no message, and whether a plain-English description can transfer the private target. It is a feasibility screen, not a protocol comparison and not a test of an invented language.

The default batch has five paired episodes. They share one fixed ordered candidate set and rotate every candidate through the private target, so target position is balanced. The conditions are:

- `full_information`: one receiver call receives a direct English description of the target. This is a capability ceiling control; it has no channel transmission.
- `no_message`: one receiver call sees only candidates. Across this five-episode block, a fixed positional guess is correct once; the analytic Bayes accuracy is 20%.
- `natural_language`: one sender call produces ordinary English, then the exact text passes through Tacit's measured loopback channel to one receiver call.

The first stage is full-information only. Proceed to no-message and English runs only if the receiver gets all five full-information cases exactly right. This conservative gate prevents interpreting format behavior when the receiver cannot do the task even with the relevant information. Five episodes do not establish a stable success rate or a protocol ranking.

## Staged dry runs

The CLI is dry-run by default and reports prompts' planned call counts without contacting an endpoint:

```powershell
python experiments/emergent_ood_v0_3/runner.py --conditions full_information --seed 17 --output .cache/emergent_ood_v0_3/full_information.jsonl
python experiments/emergent_ood_v0_3/runner.py --conditions no_message --seed 17 --output .cache/emergent_ood_v0_3/no_message.jsonl
python experiments/emergent_ood_v0_3/runner.py --conditions natural_language --seed 17 --output .cache/emergent_ood_v0_3/natural_language.jsonl
```

Each batch has a hard maximum of 12 model requests and 30-second request timeouts. These three commands plan 5, 5, and 10 calls. The same `--seed` generates identical paired episode IDs in every ledger. Dry-runs do not require a capability ledger. For execution, the full-information stage must first write the ledger shown above; the runner validates that all five exact, format-valid rows use the same receiver model, tokenizer, model population, seed, episode IDs, and candidate order before any later-stage endpoint is contacted. Supply it to each follow-up execution with `--capability-ledger .cache/emergent_ood_v0_3/full_information.jsonl`. A missing or failed gate is rejected before output creation or network requests. Never pass `--execute` until a fresh read-only host preflight satisfies the unchanged frozen gate in [`index_v0_3/preregistration.json`](../index_v0_3/preregistration.json).

After the gate passes and a local OpenAI-compatible server is already running, configure `TLU_BASE_URL` or separate sender/receiver loopback URLs, model IDs, tokenizer IDs, and `TLU_MODEL_POPULATION_ID`, then add `--execute` to exactly one staged command. Non-loopback endpoints and HTTP redirects are rejected. This runner never starts a model server, downloads weights, or calls a cloud endpoint. It has only been exercised with fake clients.

Results use `tlu.costs.v3`. The natural-language transmission measures the exact loopback JSON payload, envelope, length prefix, and acknowledgment, excluding TCP/IP headers. Provider token counts include complete prompts; they are not message-only token counts. Full-information and no-message controls have no transmission. Receiver output must be exactly one JSON object containing a valid `candidate_id`; malformed output is task failure.

The English condition is intentionally a feasibility reference, not an optimized natural-language baseline. A matched protocol study requires a larger held-out set, an optimized English instruction frozen on separate development data, valid JSON and compact structured-text baselines, a fixed symbolic code with codebook setup charged, independent receiver/model-family transfer, equal-budget sweeps, and multiple seeds. See [`preregistration.json`](preregistration.json) for the frozen scope and stopping rule.
