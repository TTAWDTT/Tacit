# Silo-Bench local-answer sufficiency audit

This instance audit evaluates each agent's exact local result under the task's stated operation. It does not model prior-based guessing or claim that a task family is always solvable without communication.

| Instance | Agents | Task | Global answer | Locally sufficient agents | All locally sufficient? |
|---|---:|---|---:|---|:---:|
| `I-01_n2.json` | 2 | Global Max | `827` | agent 0: `827`, agent 1: `780` | False |
| `I-02_n2.json` | 2 | Word Frequency | `14` | agent 0: `7`, agent 1: `7` | False |
| `I-03_n2.json` | 2 | Distributed Vote | `"Candidate_D"` | agent 0: `"Candidate_D"`, agent 1: `"Candidate_D"` | True |
| `I-04_n2.json` | 2 | Any Match | `true` | agent 0: `false`, agent 1: `true` | False |
| `I-05_n2.json` | 2 | Range Count | `22` | agent 0: `11`, agent 1: `11` | False |
| `I-06_n2.json` | 2 | Checksum | `251` | agent 0: `87`, agent 1: `172` | False |
| `I-07_n2.json` | 2 | Average Value | `49.24` | agent 0: `52.52136737471233`, agent 1: `45.94988254269882` | False |
| `I-08_n2.json` | 2 | Set Union Size | `26` | agent 0: `22`, agent 1: `18` | False |
| `I-09_n2.json` | 2 | Top-K Select | `[996, 968, 956, 930, 926, 921, 921, 906, 905, 896]` | agent 0: `[968, 930, 926, 905, 867, 850, 790, 760, 723, 708]`, agent 1: `[996, 956, 921, 921, 906, 896, 882, 855, 816, 799]` | False |
| `I-10_n2.json` | 2 | Standard Deviation | `30.58` | agent 0: `31.579446695469002`, agent 1: `29.360896369525744` | False |

Instances where every agent's exact local answer equals the global answer: `I-03_n2.json`.

This audit matched 10 fixed Paradigm-I instances using pattern `I-*_n2.json`. It checks exact local-task operations only; other paradigms and prior-based guessing are outside this audit.
