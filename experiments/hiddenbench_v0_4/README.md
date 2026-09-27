# HiddenBench Qwen3-14B reasoning-mode capability screen

The v0.3 14B screen used `--reasoning off` and greedy decoding (`temperature=0`) yet missed the full-information gate. The official Qwen3-14B model card recommends enabling its thinking mode for reasoning tasks and using sampled decoding (temperature 0.6, top-k 20, top-p 0.95); it explicitly cautions against greedy decoding in thinking mode. v0.4 tests this documented configuration before treating the model/task capability limit as established.

The task list, full-information profile, official HiddenBench initial-vote function, 12 planned votes, and task seeds remain fixed. Only inference mode and its official sampling settings change. Raw records stay in ignored `.cache/pilot_hiddenbench_v0_4/`.

The model service settings are frozen in [preregistration.json](preregistration.json). Run `python experiments/hiddenbench_v0_4/run_capability_screen.py --prepare-only` for preflight; omit the flag for the 12-vote screen. A passing score only licenses a separate complete HiddenBench study.
