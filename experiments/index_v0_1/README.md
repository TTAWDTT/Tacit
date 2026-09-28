# INDEX communication task generator v0.1

This is a synthetic task-construction and scoring artifact derived from the deterministic `INDEX_m` communication-complexity control in [`research/INDEX_PROTOCOL_DESIGN.md`](../../research/INDEX_PROTOCOL_DESIGN.md). It runs no LLM and proposes no message language.

Each episode gives the sender a private random bit vector and the receiver a private 1-based index. The receiver must return the selected bit. The JSONL record is an evaluator-side object that contains the gold answer; an eventual runner must construct each role's prompt from the corresponding view and must never send the full record to either agent.

Generate a small private shard under the ignored `.cache/` directory:

```powershell
python experiments/index_v0_1/generate_tasks.py `
  --split pilot --seed 20260928 --lengths 8 16 32 --episodes-per-length 8 `
  --output .cache/index_v0_1/pilot.jsonl
```

The generator uses independently derived deterministic seeds per split, vector length, and episode index. Its validator checks the gold bit, vector length, index range, and role-view separation. `test_generate_tasks.py` covers deterministic regeneration, separation of split streams, no cross-role leakage in the role views, and strict answer parsing.

## Controls required before any format comparison

- One agent sees both inputs and computes the selected bit (model/task capability).
- The receiver gets only its index and no message (chance baseline: 1/2 in expectation for uniform independent bits).
- An oracle channel returns the exact selected bit (transport/evaluator control).
- A language-model sender and receiver use disjoint views (actual communication condition).

First compare one-way full-vector messages to receiver-query / sender-answer interaction as a **policy** comparison. Only then compare optimized natural language, structured text, and a compositional typed code under the *same* communication policy and identical semantic payloads. Charge shared definitions, negotiation, both message directions, token/byte cost, decode failures, retries, and compute. This task is a channel control; positive results do not establish transfer to general agent collaboration.

## Runner and measurement contract

Any future model runner for this task must emit one [`tlu.costs.v2`](../../docs/COST_ACCOUNTING.md) JSONL record per task episode. Set the stratum fields to the preregistration/experiment ID, `INDEX_m` version, split, exact `m`, scorer version, and pinned sender/receiver model revisions. Keep one-way and interactive policies as distinct `policy_id` values; keep each representation and decoder under separately pinned IDs. The aggregator will then keep different vector lengths and model pairings out of the same success/cost point.

For each actual transmission, measure UTF-8 payload and framing bytes from the serialized wire representation, and count the payload with each recipient's exact tokenizer revision. Record every prompt and completion call, including failed, truncated, retry, and final-answer calls; use stages such as `communicate`, `receive`, and `answer` consistently. Keep the evaluator-only gold bit out of all agent messages. Until that runner exists and its records are audited, generator outputs are not protocol experiment results.
