# v0.24 implementation audit

The first v0.24 execution found a runner bug in the random-oracle condition. Its initial branch used `if oracle_moves:`; when the game oracle returned an empty list, the Builder fell through to the model call. Server timing records show 14 unintended completions. Therefore the first three-structure random-oracle run was not a pure no-inference condition.

The raw files from that first random-policy run were overwritten when the corrected condition reused the same ignored output path. Its per-turn traces are no longer available, and none of its task outcomes are used in the primary report. The server log retains the count and token/service totals for the 14 implementation-deviation calls.

Before rerunning the deterministic condition, the runner was corrected so that `oracle_moves=[]` returns a local clarification action and can never fall through to inference. The corrected three-structure rerun is the primary control. It independently completed `structure_008` in 15 turns: every action came from a non-empty oracle list, all 15 actions were oracle-followed, and the condition made zero model calls. The natural-language and zero-message LLM conditions were unaffected by this bug.
