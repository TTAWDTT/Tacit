# CRAFT local communication feasibility v0.17

This is a pre-registered, non-comparative feasibility run on three CRAFT structures. It asks whether a pinned local Qwen3-14B setup can execute the task when every Director gets exactly one private-view message slot and the Builder receives all three messages. It does not compare languages or claim a protocol advantage.

The adapter derives from the public CRAFT repository at commit `f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430`. It makes one narrow scheduling change: sample the three distinct roles without replacement instead of sampling three calls with replacement. The upstream loop already gives all Directors the same previous-turn conversation snapshot; the Builder transcript is assembled from the response map. Unique roles therefore make the delivered set exactly equal the generated set while preserving the snapshot/simultaneous-message structure. A fixed Python seed and fixed `PYTHONHASHSEED` make scheduling and role archetypes repeatable.

The upstream prompt, model-generated public message, Builder, oracle candidate generator, move scorer, and environment remain as in the pinned source. Oracle assistance is enabled only for the Builder, as in the paper's communication-focused setup. We use 3 structures (indices 0, 10, 19), 5 turns per game, no tool calls, and one run. This is a small capability calibration, not a paper replication.

## Run

Start the already pinned llama.cpp b11202 server with the Qwen3-14B Q4_K_M file and 24 GPU layers on `127.0.0.1:8000`, as described in `experiments/pilot_v0_16/README.md`. Then run:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/craft_v0_17/run_craft_feasibility.py
```

The script verifies the upstream Git revision, model endpoint alias, and unique-role source patch before running. The patch copy and raw CRAFT output go under ignored `.cache/`; only aggregate, anonymized run-level outcomes are intended for publication.

## Interpretation gate

Report all failures and partial progress. No language-format comparison follows automatically. A later comparison requires a fresh pre-registration, exact generated/delivered message accounting, model-input and payload token counts, paired structure seeds, and capability evidence that the selected Builder/Director setup has room to improve.
