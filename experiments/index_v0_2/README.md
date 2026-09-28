# INDEX_m deterministic controls v0.2

This preregistered, model-free control checks that the `INDEX_m` task manifest, evaluator roles, deterministic oracle policies, and `tlu.costs.v2` accounting path agree. It does **not** evaluate an LLM, compare languages, or support a protocol-superiority claim.

## Frozen task shard

The preregistration pins 16 synthetic episodes: eight each at vector lengths `m=4` and `m=8`, generated with split `pilot` and master seed `20260929`. The manifest SHA-256 is `637e79254f4dc8f10e65c184a5653ffbf784ab86ecfa98795073e2c68ddecab`. The raw task file is kept under ignored `.cache/` and can be regenerated with:

```powershell
python experiments/index_v0_1/generate_tasks.py `
  --split pilot --seed 20260929 --lengths 4 8 --episodes-per-length 8 `
  --output .cache/index_v0_2/tasks.jsonl
```

## Controls

The frozen conditions are: a receiver returning fixed zero without a message; a centralized full-information oracle; a one-way oracle transmitting the full ASCII bit vector; and an interactive oracle transmitting a fixed-width binary index followed by the selected bit. The preregistration records expected success and payload sizes. Transmissions include payload UTF-8 bytes; framing is currently zero because this deterministic harness has no transport envelope. Recipient model-token counts are explicitly marked not applicable, and each record must contain zero model calls.

## Run and aggregate

After generating the frozen shard, run:

```powershell
python experiments/index_v0_2/run_oracle_controls.py `
  --tasks .cache/index_v0_2/tasks.jsonl `
  --output .cache/index_v0_2/cost_records.jsonl
python tools/cost_report.py .cache/index_v0_2/cost_records.jsonl `
  --output .cache/index_v0_2/cost_report.json
```

The runner checks the preregistered task hash and per-length counts before producing one `tlu.costs.v2` record per episode and condition. The cost report keeps task size and oracle population strata separate. Runtime fields describe only the local deterministic Python runner and are not estimates of model inference latency or system-wide CPU load.

## Interpretation

These controls can catch plumbing errors and demonstrate the expected separation between channel policies on `INDEX_m`. They cannot show that an LLM can perform the sender/receiver operations, that a learned or designed language helps, or that byte counts predict model-token cost. Any LLM communication experiment requires its own frozen model/resource plan and capability gate; no model is loaded or called by this v0.2 runner.
