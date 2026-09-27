# HiddenBench local capability calibration v0.2

**Run date:** 2026-09-28  
**Status:** complete; calibration only.  
**Model:** Qwen3-8B Q4_K_M, llama.cpp b11202, 8,192 context, 99 GPU layers.  
**Task source:** HiddenBench's three-task verification split, pinned MIT repository revision `3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1` ([paper](https://arxiv.org/abs/2505.11556), [code and dataset](https://github.com/Yassellee/HiddenBench_ICML)).

## Result

The preregistered full-information capability gate **failed**. On three tasks, the mean fraction of correct individual initial votes was **0.167** (2/12 agents); the majority was correct on **0/3** tasks. After 15 discussion rounds under full information, average individual accuracy was 0.333 and group majority accuracy was 1/3. These are descriptive outcomes from one small-model run, not evidence that discussion improved reasoning.

Under the hidden profile, initial and final individual accuracy were both **0/12**, and no task had a correct final majority. Discussion did not recover the correct answer in these three runs. Since the full-information gate was already far below its frozen 0.8 threshold, the model/task pair is not eligible for a communication-protocol ranking.

| Profile | Initial individual accuracy | Final individual accuracy | Initial majority | Final majority |
|---|---:|---:|---:|---:|
| Full information | 0.167 (2/12) | 0.333 (4/12) | 0/3 | 1/3 |
| Hidden information | 0.000 (0/12) | 0.000 (0/12) | 0/3 | 0/3 |

## Communication and inference accounting

The run completed all **408/408** nominal model completions (204 per profile), with no retries or truncated outputs. Across the six episodes, each profile generated 180 discussion messages (60 per task). Tokenizing the serialized `Agent: message` transcript with the pinned local llama.cpp tokenizer yielded 9,308 wire tokens / 47,192 UTF-8 bytes for full information and 7,340 tokens / 37,017 bytes for hidden information. This is transcript size only; it is not total prompt cost or a protocol efficiency comparison.

The server reported 112,019 prompt-evaluation tokens, 18,075 generated tokens, 468.377 seconds summed service time (1.148 seconds per completion on average), a maximum observed context length of 4,676 tokens, and zero truncations. Prompt-evaluation counts are cache-aware llama.cpp work counters, so they may be lower than the total prompt tokens transmitted by the client.

## Interpretation

This run confirms that the local adapter can execute the 15-round HiddenBench flow with durable task-by-task output at 8,192 context. It does not establish that the model can solve the tasks: performance is poor even when every hidden fact is visible to every agent. That prevents a fair test of whether a communication representation helps. The negative result narrows the next step to local capability calibration: either find a locally feasible model with substantially higher full-profile accuracy, or reduce task complexity and validate a matched hidden-information task where the same model reliably solves the full-information control.

No new language, message format, or protocol is proposed from these results. The full sanitized aggregates and analysis code are [RESULTS.json](../experiments/hiddenbench_v0_2/RESULTS.json) and [analyze_hiddenbench_v0_2.py](analyze_hiddenbench_v0_2.py). Raw prompts, private fact assignments, rationales, and transcripts remain in ignored `.cache/pilot_hiddenbench_v0_2/` and are not part of the public result.
