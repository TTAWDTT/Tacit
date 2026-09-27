# CRAFT v0.24: does the oracle make communication unnecessary?

## Question

The CRAFT Builder receives up to five actions already checked by the game oracle as valid local progress. Does the bounded-history natural-language team outperform an otherwise identical Builder that gets no Director messages, and can a message-free policy complete the same tasks by choosing among oracle actions?

This is a benchmark-validity and causal ablation, not a language-format comparison. The public v0.23 reference stopped at eight turns; v0.24 reruns the natural-language condition to the paper's 20-turn limit so the two model conditions have matched horizons.

## Frozen conditions

All conditions use CRAFT revision `f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430`, structure indices `[0,7,2]`, run seed `323`, the same Qwen3-8B-Q4_K_M artifact and llama.cpp b11202 server configuration as v0.23, `oracle_n=5`, and the existing upstream move validator and completion scorer. All start from the upstream `partComplete=True` initialization; record the realized starting board and partial-completion category.

1. **Natural-language team:** three distinct Directors and one Builder per turn, recent-16-line Director history, same parser and backend-error handling as v0.23, 20-turn maximum.
2. **Zero-message LLM Builder:** skip every Director generation and deliver an empty discussion to the same Builder. Preserve the five oracle candidates, Builder prompt, parser, decoding settings, and 20-turn maximum. A clarification or invalid action counts as the Builder's outcome; do not inject a replacement message or retry.
3. **Zero-message oracle policy:** no model calls and no Director messages. On each turn, use the exact game oracle to enumerate/sample the same up-to-five legal forward-progress candidates, then choose one uniformly with an independent fixed RNG. Run up to 20 turns using the same engine and scorer. This deliberately measures what the benchmark permits with oracle assistance; it is not a capability claim about an ordinary LLM.

## Outcomes and predictions

Report per structure and turn: completion, progress, executed moves, oracle-following, failed/clarification actions, candidate count, starting state/category, Director messages delivered, Builder-visible wire tokens, and model calls/token/service totals. Compare the two model conditions first within structure, not by pooling turns as independent samples.

Prediction: the oracle-only policy will usually make local progress, but may fail to complete in 20 turns because locally correct actions need not form an efficient global construction order. If it completes any task, CRAFT with this oracle does not satisfy a strict “communication is necessary for success” criterion for that instance. If it does not complete, this weak policy alone cannot establish necessity. The zero-message LLM comparison estimates the effect of removing Director messages under the existing oracle and prompts; it does not isolate a protocol effect.

No language superiority, generalization, scaling, or efficiency-frontier claim will be made from this single-seed ablation. If the no-message condition is competitive or the oracle policy completes, revise task eligibility before any protocol study. If communication appears necessary, run repeated paired seeds and stronger optimized no-message policies before treating the benchmark as communication-essential.

## Preregistered limits

One seed, three structures, and one local model give descriptive evidence only. The 20-turn natural-language rerun may encounter context/backend errors; report them as missing public messages and preserve all partial trajectories. Do not silently substitute fallback messages. No new protocol is introduced in this experiment.
