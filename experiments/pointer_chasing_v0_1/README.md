# Pointer-chasing task generator v0.1

This model-free artifact generates deterministic two-party pointer-chasing episodes from the standard communication-complexity task. It loads no model and performs no inference. The task definition and theoretical scope are documented in [`research/INTERACTIVE_POINTER_CHASING_DESIGN.md`](../../research/INTERACTIVE_POINTER_CHASING_DESIGN.md).

Each evaluator-side JSONL record contains independent private function maps for Agent A and Agent B plus the gold answer. Task sizes are restricted to even `n >= 2`, with both parity classes equal in cardinality; this does not guarantee a balanced answer distribution at every depth, so no-message baselines are measured separately. An eventual runner must construct each prompt from only that agent's view and the shared task definition; it must never pass the complete record, gold bit, `master_seed`, split, or `episode_id` to either agent. The seed metadata would let a participant reconstruct the other map. Both agents must submit a final bit after communicating, and the primary task score is joint exactness: both final answers must be correct. Collect the two answers as sealed outputs; do not reveal one agent's final answer to the other, since that would add an unbudgeted message. This prevents the last pointer holder from solving the task alone without sharing the result during the measured protocol.

Generate a small local pilot shard:

```powershell
python experiments/pointer_chasing_v0_1/generate_tasks.py `
  --split pilot --seed 20260929 --sizes 8 16 --depths 2 3 `
  --episodes-per-condition 4 --output .cache/pointer_chasing_v0_1/pilot.jsonl
```

Seeds are derived with SHA-256 from the master seed, split, size, depth, and episode index; each role/map slot is sampled with SHA-256 rejection sampling rather than a runtime-version-dependent PRNG. This makes a deterministic pseudorandom instantiation of the target uniform distribution, not a proof that the finite shard is an IID sample. A fixed regression vector protects the task version's sampling contract. The validator checks the task version, exact schema, role-view shape, function range, and independently recomputed answer. The scorer accepts only exact `0` or `1` outputs from both agents and separately reports individual and joint exactness. `oracle_relay(episode)` executes the k-message pointer relay using fixed-width binary offsets of `ceil(log2(n))` bits per pointer and checks each decode. `exact_no_message_diagnostic(n,k)` exhaustively computes per-agent Bayes accuracies for `n` in `{2,4}` under the full uniform function prior. It also provides feasible joint baselines and an upper bound on any no-message joint accuracy, but does not solve for the globally optimal pair of local decision rules; enumeration is capped because cost grows quadratically in `n^n`. The reported relay payload bit count excludes framing and is not a `tlu.costs.v3` wire-cost record.

Run the offline integrity checks with:

```powershell
python -m unittest discover -s experiments/pointer_chasing_v0_1 -p "test_*.py" -v
```

These checks validate deterministic data plumbing only. They are not a model-capability result, a measured round-complexity result, or evidence for a message format. Before a model pilot, first verify the full-information capability gate, participant-specific no-message controls, and an oracle relay through the same transport. Do not treat asymptotic bit bounds as token budgets.
