# CRAFT implementation audit v0.1

The source-and-trace analysis is extended by the full public dataset audit in [v0.2](CRAFT_TRACE_DATASET_AUDIT_V0_2.md). The inspected upstream snapshot is `csu-signal/CRAFT` commit `f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430`, checked out under the ignored project cache. CRAFT remains a candidate benchmark; no Tacit model call has used it.

## Why it is a strong candidate

The paper specifies three Directors with complementary private projections and one Builder that sees the shared board and their messages. No Director sees the full target. Oracle-verified candidate moves let the benchmark isolate communication from the Builder independently discovering a valid move. Its turn-level construction task is materially closer to distributed coordination than PrefixSum.

## Confirmed source and trace discrepancy to resolve before reuse

The paper's v2 description says each turn samples one to three **unique** Directors. In the inspected [`run_craft.py`](https://github.com/csu-signal/CRAFT/blob/f174fa5ae80c20ce5ccc7bb7cbf4aeab69efe430/run_craft.py), `random.choices(["D1", "D2", "D3"], k=3)` samples exactly three entries **with replacement**. Repeated IDs are queried multiple times in a turn.

The runner stores answers in `director_responses[did]`, so a later response for the same ID overwrites the earlier one in the dictionary; it also appends every response to `conversation_history`. The Builder transcript is subsequently built from the dictionary, while later Directors have seen the append-only history. Therefore duplicate-ID outputs can influence subsequent Directors but be omitted from the Builder's current-turn transcript.

This is visible in all five checked-in 5-turn sample traces at the inspected revision: each `conversation_snapshot` has 15 Director posts (three per turn), while the sum of unique `director_responses` keys is 8, 9, 12, 9, and 9. Thus 3–7 generated posts per game are collapsed before the Builder transcript. These are sample traces, not the paper's full evaluation. The inspected repository snapshot may differ from the revision used for the paper's results, so this finding does not establish that the paper's reported experiments used this exact behavior.

## Adoption conditions

Before using CRAFT for protocol experiments, pin the precise upstream revision and reconcile this scheduling discrepancy against the paper's released logs/configuration. If the current source is the intended runtime, prepare a minimal, documented adapter that samples a fixed set of distinct Directors and passes exactly the same ordered public transcript to all recipients. Preserve the original task engine, oracle candidate policy, role prompts, model, turn limit, and random seeds. Count generated and delivered messages separately. First reproduce one Natural Language condition and validate the information path; only then preregister message-format or bandwidth interventions.

The first Tacit result should be a benchmark/runtime fidelity audit. CRAFT's own natural-language results are not a baseline we have reproduced, and its optional LLM-judge scores should not replace the executable game outcome as the primary endpoint.
