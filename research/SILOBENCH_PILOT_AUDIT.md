# Pilot v0.1: diagnostic findings and stop decision

**Status:** exploratory run interrupted after discovering invalid task instances and a receiver-side interaction failure. These rows are diagnostics only; they are not a representation comparison and support no superiority claim.

## What ran

Pinned Silo-Bench task files `I-01_n2.json` and `I-03_n2.json` with Qwen3-1.7B, a four-round cap, P2P message tool, and a maximum response length of 256 tokens. The local endpoint reported tokenizer-measured prompt and completion tokens. Full traces remain in the ignored `.cache/pilot_v0_1/` directory and can be regenerated using the pinned runner.

### `I-01_n2.json` — Global Max

| Condition | Success rate | Input + output tokens | Message payload bytes | Wall time (s) |
|---|---:|---:|---:|---:|
| Unconstrained | 0.00 | 8,664 | 46 | 15.583 |
| Concise NL | 0.50 | 8,938 | 46 | 15.841 |
| AutoForm-style instruction | 0.50 | 9,002 | 46 | 15.735 |
| Fixed JSON | 0.00 | 9,920 | 0 | 30.764 |
| No communication | 0.50 | 7,606 | 0 | 39.276 |

The no-communication score is expected for this *instance*: agent 0's local maximum is already the global maximum (827), while agent 1's local maximum is 780. The independent audit records this local-answer sufficiency explicitly.

The trace shows a separate receiver failure. Agent 0 sent `827` to agent 1; agent 1 received it, then repeatedly sent a message to itself and polled again instead of submitting `827`. Agent 0 also failed to terminate. Therefore these outcomes primarily expose interaction-policy/tool-following limitations of this receiver, not a semantic-encoding result. Concise NL and the AutoForm-style prompt happened to get agent 0 to submit but did not fix agent 1.

### `I-03_n2.json` — Distributed Vote

The no-communication arm achieved 1.00 success: both local shards independently select `Candidate_D`, which is also the global winner. The upstream generator overwrites every second vote with a shared winner; with 25 votes per agent, that instance makes the answer locally recoverable by both agents. This violates the intended communication-necessity criterion, so the task was excluded from further comparison. Some other format arms ran before the discovery, but are not evidence about representation quality.

## Dataset audit

`audit_silobench_private_information.py` evaluates exact local operations on all 60 fixed Paradigm-I instances across agent counts 2, 5, 10, 20, 50, and 100. Eight instances let every agent reproduce the global answer by applying the task operation to only its own shard: all six Distributed Vote instances and two Any Match instances. The detailed n=2 outputs are in [`SILOBENCH_N2_AUDIT.md`](SILOBENCH_N2_AUDIT.md); the compact 60-instance table is in [`SILOBENCH_PARADIGM_I_AUDIT.md`](SILOBENCH_PARADIGM_I_AUDIT.md).

This is a local-answer sufficiency check, not proof that all other tasks are communication-necessary: a model can sometimes guess from the task prior, and Paradigms II/III need their own checks. Future inclusion requires (a) an oracle check that each agent's local observation is insufficient for the shared target and (b) an empirical no-communication condition over multiple instances/seeds.

## Decision

Stop representation comparisons on these fixed rows. Retain Silo-Bench as a candidate infrastructure, but curate or generate instances that pass the information-sufficiency gate, then first calibrate a receiver/prompt/tool policy that can complete valid multi-round exchanges. Only after that should bandwidth or format sweeps be interpreted.
