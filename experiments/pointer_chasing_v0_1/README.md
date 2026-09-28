# Pointer-chasing task generator v0.1

This model-free artifact generates deterministic two-party pointer-chasing episodes from the standard communication-complexity task. It loads no model and performs no inference. The task definition and theoretical scope are documented in [`research/INTERACTIVE_POINTER_CHASING_DESIGN.md`](../../research/INTERACTIVE_POINTER_CHASING_DESIGN.md).

Each evaluator-side JSONL record contains independent private function maps for Agent A and Agent B plus the gold answer. An eventual runner must construct each prompt from only that agent's view and the shared task definition; it must never pass the complete record or gold bit to either agent.

Generate a small local pilot shard:

```powershell
python experiments/pointer_chasing_v0_1/generate_tasks.py `
  --split pilot --seed 20260929 --sizes 8 16 --depths 2 3 `
  --episodes-per-condition 4 --output .cache/pointer_chasing_v0_1/pilot.jsonl
```

Seeds are derived with SHA-256 from the master seed, split, size, depth, and episode index. The validator checks the task version, exact schema, role-view shape, function range, and independently recomputed answer. The scorer accepts only an exact `0` or `1` final answer. `oracle_relay(episode)` executes the k-message pointer relay using fixed-width binary offsets of `ceil(log2(n))` bits per pointer and checks each decode. Its reported payload bit count excludes transport framing and is not a `tlu.costs.v3` wire-cost record.

Run the offline integrity checks with:

```powershell
python -m unittest discover -s experiments/pointer_chasing_v0_1 -p "test_*.py" -v
```

These checks validate deterministic data plumbing only. They are not a model-capability result, a measured round-complexity result, or evidence for a message format. Before a model pilot, first verify the full-information capability gate, participant-specific no-message controls, and an oracle relay through the same transport. Do not treat asymptotic bit bounds as token budgets.
