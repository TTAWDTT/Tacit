# Emergent OOD v0.3: staged local capability controls

This runner connects the corrected v0.2 role-separated task to local chat models. Its purpose is to find out whether the receiver can solve the task when given the answer, how it behaves with no message, and whether a plain-English description can transfer the private target. It is a feasibility screen, not a protocol comparison and not a test of an invented language.

The default batch has five paired episodes. They share one fixed ordered candidate set and rotate every candidate through the private target, so target position is balanced. The theoretical four-bit reference is specifically a fixed-width payload bound; prefix-free coding, external packet boundaries, and their serialized framing costs are distinguished in [`docs/THEORY.md` §12](../../docs/THEORY.md). The conditions are:

- `full_information`: one receiver call receives a direct English description of the target. This is a capability ceiling control; it has no channel transmission.
- `no_message`: one receiver call sees only candidates. Across this five-episode block, a fixed positional guess is correct once; the analytic Bayes accuracy is 20%.
- `natural_language`: one sender call produces ordinary English, then the exact text passes through Tacit's measured loopback channel to one receiver call.

The first stage is a full-information calibration block on seed 17. Proceed to the seed-18 evaluation conditions only if the receiver gets all five calibration cases exactly right. The independent gate checks basic task eligibility; it does not filter the evaluation episodes. Five calibration cases do not establish a stable success rate or a protocol ranking.

## Staged dry runs

The CLI is dry-run by default and reports prompts' planned call counts without contacting an endpoint:

```powershell
python experiments/emergent_ood_v0_3/runner.py --conditions full_information --seed 17 --output .cache/emergent_ood_v0_3/full_information.jsonl
python experiments/emergent_ood_v0_3/runner.py --conditions full_information --seed 18 --output .cache/emergent_ood_v0_3/full_information_eval.jsonl
python experiments/emergent_ood_v0_3/runner.py --conditions no_message --seed 18 --output .cache/emergent_ood_v0_3/no_message.jsonl
python experiments/emergent_ood_v0_3/runner.py --conditions natural_language --seed 18 --output .cache/emergent_ood_v0_3/natural_language.jsonl
```

Each batch has a hard maximum of 12 model requests and 30-second request timeouts. The first command is a calibration-only full-information screen on seed 17. The remaining three commands evaluate a separate seed-18 block; those evaluation ledgers share paired episode IDs. Keep the calibration ledger out of the evaluation report. A perfect calibration result qualifies the fixed receiver for the follow-up, but must not choose which evaluation episodes are retained. Dry-runs do not require either gate ledger. Before launching the endpoint, create a fresh read-only host report while the intended endpoint ports are still free:

```powershell
./experiments/emergent_ood_v0_3/resource_preflight.ps1 -Ports 8000
```

For separate sender and receiver endpoints, list both ports, for example `-Ports 8000,8001`. The script records three CPU samples, GPU use/memory, free system memory, checked ports, host name, and UTC time. It writes a report under `.cache/` even when the gate rejects, and exits nonzero on rejection. It reads no model files and starts no service. After a passing report, start the local endpoint within five minutes and supply `.cache/emergent_ood_v0_3/resource_preflight.json` with `--resource-preflight` to the executing runner command; the runner verifies freshness, host, exact frozen thresholds, measurements, and endpoint ports before output creation or network requests. The unchanged thresholds are in [`index_v0_3/preregistration.json`](../index_v0_3/preregistration.json).

For execution, first write the seed-17 calibration ledger and require all five exact, format-valid rows. Use that seed-disjoint ledger for seed-18 follow-ups with `--capability-ledger .cache/emergent_ood_v0_3/full_information.jsonl`; the runner checks the task family, receiver model/tokenizer/population, complete calibration block, and that calibration and evaluation episode IDs do not overlap. The separate seed-18 full-information ledger is an evaluation control, not the capability gate. A missing, failed, or overlapping capability ledger is rejected before output creation or network requests.

After the preflight passes and the local OpenAI-compatible server is running, configure `TLU_BASE_URL` or separate sender/receiver loopback URLs, model IDs, tokenizer IDs, and `TLU_MODEL_POPULATION_ID`, then add `--execute --resource-preflight .cache/emergent_ood_v0_3/resource_preflight.json` to exactly one staged command. Follow-up commands also require `--capability-ledger .cache/emergent_ood_v0_3/full_information.jsonl`. Non-loopback endpoints and HTTP redirects are rejected. The preflight is a launch gate, not a live resource monitor: keep the externally managed model server's own stop safeguards active. This runner never starts a model server, downloads weights, or calls a cloud endpoint. It has only been exercised with fake clients.

Results use `tlu.costs.v3`. The natural-language transmission measures the exact loopback JSON payload, envelope, length prefix, and acknowledgment, excluding TCP/IP headers. Provider token counts include complete prompts; they are not message-only token counts. Full-information and no-message controls have no transmission. Receiver output must be exactly one JSON object containing a valid `candidate_id`; malformed output is task failure.

The English condition is intentionally a feasibility reference, not an optimized natural-language baseline. A matched protocol study requires a larger held-out set, an optimized English instruction frozen on separate development data, valid JSON and compact structured-text baselines, a fixed symbolic code with codebook setup charged, independent receiver/model-family transfer, equal-budget sweeps, and multiple seeds. See [`preregistration.json`](preregistration.json) for the frozen scope and stopping rule.
