# Interaction calibration v0.2: one-instance result

**Status:** exploratory diagnostic, one fixed Global Max instance, one greedy run per arm. No estimate here is a population effect.

| Condition | Success | Input + output tokens | Message bytes | Send / receive / submit | Self-sends | Wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Unconstrained | 0.50 | 9,181 | 46 | 8 / 2 / 2 | 1 | 63.818 |
| Scaffold only | 1.00 | 7,842 | 46 | 3 / 2 / 2 | 1 | 18.293 |
| Concise NL + scaffold | 0.00 | 9,432 | 46 | 4 / 4 / 0 | 2 | 20.952 |
| No communication | 0.50 | 7,606 | 0 | 7 / 2 / 2 | 1 | 41.565 |

All message-bearing arms transmitted the exact same 46-byte payload (`My local maximum is 827`). The concise-format instruction therefore did not compress the payload; in this run it added prompt tokens and failed to complete either agent's submission. The scaffold-only arm completed both submissions after one self-send; the unconstrained arm completed one correct and one incorrect answer after repeated messages. The no-communication score of 0.50 follows from agent 0 already holding the global maximum in this fixed instance.

This result supports only a narrow diagnosis: on this receiver/task, a generic interaction scaffold changed behavior and was associated with fewer calls, lower measured total tokens, and complete task success in one run. The natural-language format hint did not improve encoding or success in the paired scaffold condition. Since greedy outputs are deterministic, repeating an identical prompt would add no independent evidence; replication must vary task instances, receiver conditions, or decoding seed where stochastic sampling is used.

Raw traces are retained under the ignored `.cache/pilot_v0_2/`. The prompt, model, task, and accounting boundary are pinned in [`policies.json`](../experiments/pilot_v0_2/policies.json). No new language is proposed from this result.
