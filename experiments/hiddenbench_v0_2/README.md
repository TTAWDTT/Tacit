# HiddenBench local calibration v0.2

This is a preregistered repair of the v0.1 local feasibility run. v0.1 failed at the 4,096-token context ceiling before it wrote a complete score file; see [the failure report](../hiddenbench_v0_1/FAILURE_REPORT.md). No v0.1 score is used here.

The experiment repeats the same three official verification tasks, profiles, seed schedule, four agents, 15 discussion rounds, Qwen3-8B artifact, and temperature. The llama.cpp context is raised to 8,192. Each task runs in its own official CLI process, and its raw result is written immediately so a later task failure cannot erase earlier tasks. The task-index seed offsets match the official runner's original `base_seed + task_index * 10_000` rule.

Raw results and per-task logs remain under ignored `.cache/pilot_hiddenbench_v0_2/`. Publish only aggregate metrics; do not expose task-private facts, prompts, rationales, or transcripts.

Run `experiments/hiddenbench_v0_2/run_hiddenbench_v0_2.py --prepare-only` to verify the source, model, data, and idle 8,192-context server. Omit the flag to run full then hidden profiles.

The completed v0.2 calibration failed its preregistered full-profile gate (initial average accuracy 0.167 against a 0.8 threshold). It therefore does not support protocol ranking. See the sanitized [results](RESULTS.json) and [analysis](../../research/HIDDENBENCH_LOCAL_CALIBRATION_V0_2.md); raw conversations remain local.
