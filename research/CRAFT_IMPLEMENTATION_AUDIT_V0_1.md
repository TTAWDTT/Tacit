# CRAFT implementation audit v0.1

This is a static source audit, not an execution result. The inspected upstream snapshot is `csu-signal/CRAFT` commit `f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430`, checked out under the ignored project cache. CRAFT remains a candidate benchmark; no Tacit model call has used it.

## Why it is a strong candidate

The paper specifies three Directors with complementary private projections and one Builder that sees the shared board and their messages. No Director sees the full target. Oracle-verified candidate moves let the benchmark isolate communication from the Builder independently discovering a valid move. Its turn-level construction task is materially closer to distributed coordination than PrefixSum.

## Source discrepancy to resolve before reuse

The paper's v2 description says each turn samples one to three **unique** Directors. In the inspected [`run_craft.py`](https://github.com/csu-signal/CRAFT/blob/f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430/run_craft.py), `random.choices(["D1", "D2", "D3"], k=3)` samples three entries **with replacement**. Repeated IDs can therefore be queried multiple times in one turn.

The runner stores answers in `director_responses[did]`, so a later response for the same ID overwrites the earlier one in the dictionary; it also appends every response to `conversation_history`. The Builder transcript is subsequently built from the dictionary, while later Directors have seen the append-only history. If an ID repeats, the Builder and Directors may receive different message sets. This is a potential information-path and reproducibility defect, not evidence that the published results are invalid: the inspected source snapshot may differ from the paper's executed revision.

## Adoption conditions

Before using CRAFT for protocol experiments, pin the precise upstream revision and reconcile this scheduling discrepancy against the paper's released logs/configuration. If the current source is the intended runtime, prepare a minimal, documented adapter that samples a fixed set of distinct Directors and passes exactly the same ordered public transcript to all recipients. Preserve the original task engine, oracle candidate policy, role prompts, model, turn limit, and random seeds. First reproduce one Natural Language condition and validate that each recipient's observed transcript is identical; only then preregister message-format or bandwidth interventions.

The first Tacit result should be a benchmark/runtime fidelity audit. CRAFT's own natural-language results are not a baseline we have reproduced, and its optional LLM-judge scores should not replace the executable game outcome as the primary endpoint.
