# CRAFT bounded-history natural-language baseline v0.23

v0.23 reruns the three-level CRAFT natural-language baseline after v0.22 exceeded the available 4,096-token context and terminated on a faulty substring check.

All roles, task structures, model settings, and public wording remain the same. Two channel/runtime rules change and are fixed across the entire study:

- Directors receive the most recent 16 shared conversation lines as history. Each Director in a turn receives the same snapshot.
- Backend errors are recorded as failed message attempts and omitted from shared history and the Builder transcript. They are never substituted with synthetic natural-language messages.

The parser extracts the `<message>` region, removes its format tags and one outer square-bracket envelope, and preserves the remaining generated prose. The run covers structure indices 0, 7, and 2 (medium, simple, complex), eight turns each, with a 96-call cap. This is the bounded-history natural-language reference condition for later preregistered protocol comparisons.

Run with the pinned Qwen3-8B 4K single-slot server from v0.21 already loaded:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/craft_v0_23/run_craft_baseline.py
```

Raw CRAFT traces stay in ignored `.cache/pilot_v0_23/`.
