# v0.24 implementation audit

The first v0.24 execution found a runner bug in the random-oracle condition. Its initial branch used `if oracle_moves:`; when the game oracle returned an empty list, the Builder fell through to the model call. Therefore the complete three-structure random-oracle run was not a pure no-inference condition.

One result remains directly auditable: `structure_008` completed in 16 turns. Its trace shows at least one oracle candidate on every turn through completion, every selected move followed the candidate list, and no Builder confirmation text was generated. The deterministic policy branch therefore made all 16 moves without invoking a model. This single episode is a valid counterexample to strict communication necessity under the oracle-assisted task setup, but it is not a complete evaluation of the preregistered condition across all three structures.

Before rerunning the deterministic condition, the runner is corrected so that `oracle_moves=[]` returns a local clarification action and can never fall through to inference. The initial run remains preserved in the ignored `.cache/pilot_v0_24/` directory and will be labeled as an implementation-deviation run; the corrected rerun is the primary three-structure control. The natural-language and zero-message LLM conditions were unaffected by this bug.
