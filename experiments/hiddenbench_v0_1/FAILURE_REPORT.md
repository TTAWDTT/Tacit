# HiddenBench v0.1 local calibration failure

**Run date:** 2026-09-28  
**Status:** incomplete; no benchmark scores are reported.

The pinned Qwen3-8B service matched the preregistered model artifact and 4,096-token context. The official HiddenBench runner reached progress callbacks for the first two of three full-profile tasks, then failed during final-vote JSON parsing on the third task. The server log shows the failing completion reached the 4,095-token context limit and was truncated. The upstream CLI only writes its results JSON after every task finishes, so the two finished in-memory task records were lost when the process exited. The runner log preserves the exception and progress callbacks; the raw model output was not persisted separately.

There were 204 server completions between the pre-run task ID and the final truncated response. This equals the nominal full-profile budget, but includes any vote-level retry calls. Do not treat it as a valid 3-task score or combine it with a future run. The hidden-profile condition did not start.

This is a context-capacity and partial-persistence failure, not evidence about communication quality. Before any rerun, v0.2 increases the local context to 8,192 tokens, executes one official task per CLI process while preserving the original per-task seeds, and persists each task output immediately. The task file, prompts, 15 rounds, temperature, model, and scoring remain fixed. This amendment is published before new model calls.
