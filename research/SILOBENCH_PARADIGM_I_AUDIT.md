# Silo-Bench local-answer sufficiency audit

This instance audit evaluates each agent's exact local result under the task's stated operation. It does not model prior-based guessing or claim that a task family is always solvable without communication.

| Instance | Agents | Task | Global answer | Locally sufficient agents | All locally sufficient? |
|---|---:|---|---:|---|:---:|
| `I-01_n10.json` | 10 | Global Max | `996` | 1/10 agents (IDs: 4) | False |
| `I-01_n100.json` | 100 | Global Max | `1000` | 1/100 agents (IDs: 35) | False |
| `I-01_n2.json` | 2 | Global Max | `827` | 1/2 agents (IDs: 0) | False |
| `I-01_n20.json` | 20 | Global Max | `1000` | 1/20 agents (IDs: 10) | False |
| `I-01_n5.json` | 5 | Global Max | `988` | 1/5 agents (IDs: 0) | False |
| `I-01_n50.json` | 50 | Global Max | `1000` | 1/50 agents (IDs: 48) | False |
| `I-02_n10.json` | 10 | Word Frequency | `80` | 0/10 agents | False |
| `I-02_n100.json` | 100 | Word Frequency | `696` | 0/100 agents | False |
| `I-02_n2.json` | 2 | Word Frequency | `14` | 0/2 agents | False |
| `I-02_n20.json` | 20 | Word Frequency | `133` | 0/20 agents | False |
| `I-02_n5.json` | 5 | Word Frequency | `28` | 0/5 agents | False |
| `I-02_n50.json` | 50 | Word Frequency | `340` | 0/50 agents | False |
| `I-03_n10.json` | 10 | Distributed Vote | `"Candidate_A"` | 10/10 agents (IDs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9) | True |
| `I-03_n100.json` | 100 | Distributed Vote | `"Candidate_A"` | 100/100 agents (IDs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99) | True |
| `I-03_n2.json` | 2 | Distributed Vote | `"Candidate_D"` | 2/2 agents (IDs: 0, 1) | True |
| `I-03_n20.json` | 20 | Distributed Vote | `"Candidate_B"` | 20/20 agents (IDs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19) | True |
| `I-03_n5.json` | 5 | Distributed Vote | `"Candidate_C"` | 5/5 agents (IDs: 0, 1, 2, 3, 4) | True |
| `I-03_n50.json` | 50 | Distributed Vote | `"Candidate_D"` | 50/50 agents (IDs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49) | True |
| `I-04_n10.json` | 10 | Any Match | `true` | 1/10 agents (IDs: 8) | False |
| `I-04_n100.json` | 100 | Any Match | `false` | 100/100 agents (IDs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99) | True |
| `I-04_n2.json` | 2 | Any Match | `true` | 1/2 agents (IDs: 1) | False |
| `I-04_n20.json` | 20 | Any Match | `true` | 1/20 agents (IDs: 13) | False |
| `I-04_n5.json` | 5 | Any Match | `false` | 5/5 agents (IDs: 0, 1, 2, 3, 4) | True |
| `I-04_n50.json` | 50 | Any Match | `true` | 1/50 agents (IDs: 49) | False |
| `I-05_n10.json` | 10 | Range Count | `117` | 0/10 agents | False |
| `I-05_n100.json` | 100 | Range Count | `1199` | 0/100 agents | False |
| `I-05_n2.json` | 2 | Range Count | `22` | 0/2 agents | False |
| `I-05_n20.json` | 20 | Range Count | `226` | 0/20 agents | False |
| `I-05_n5.json` | 5 | Range Count | `50` | 0/5 agents | False |
| `I-05_n50.json` | 50 | Range Count | `621` | 0/50 agents | False |
| `I-06_n10.json` | 10 | Checksum | `123` | 0/10 agents | False |
| `I-06_n100.json` | 100 | Checksum | `165` | 0/100 agents | False |
| `I-06_n2.json` | 2 | Checksum | `251` | 0/2 agents | False |
| `I-06_n20.json` | 20 | Checksum | `93` | 0/20 agents | False |
| `I-06_n5.json` | 5 | Checksum | `132` | 0/5 agents | False |
| `I-06_n50.json` | 50 | Checksum | `5` | 0/50 agents | False |
| `I-07_n10.json` | 10 | Average Value | `51.62` | 0/10 agents | False |
| `I-07_n100.json` | 100 | Average Value | `49.52` | 0/100 agents | False |
| `I-07_n2.json` | 2 | Average Value | `49.24` | 0/2 agents | False |
| `I-07_n20.json` | 20 | Average Value | `49.68` | 0/20 agents | False |
| `I-07_n5.json` | 5 | Average Value | `53.6` | 0/5 agents | False |
| `I-07_n50.json` | 50 | Average Value | `50.33` | 0/50 agents | False |
| `I-08_n10.json` | 10 | Set Union Size | `125` | 0/10 agents | False |
| `I-08_n100.json` | 100 | Set Union Size | `1302` | 0/100 agents | False |
| `I-08_n2.json` | 2 | Set Union Size | `26` | 0/2 agents | False |
| `I-08_n20.json` | 20 | Set Union Size | `258` | 0/20 agents | False |
| `I-08_n5.json` | 5 | Set Union Size | `67` | 0/5 agents | False |
| `I-08_n50.json` | 50 | Set Union Size | `647` | 0/50 agents | False |
| `I-09_n10.json` | 10 | Top-K Select | `[1000, 1000, 999, 997, 995, 994, 994, 993, 993, 984]` | 0/10 agents | False |
| `I-09_n100.json` | 100 | Top-K Select | `[1000, 1000, 1000, 999, 998, 998, 998, 998, 997, 997]` | 0/100 agents | False |
| `I-09_n2.json` | 2 | Top-K Select | `[996, 968, 956, 930, 926, 921, 921, 906, 905, 896]` | 0/2 agents | False |
| `I-09_n20.json` | 20 | Top-K Select | `[997, 992, 986, 985, 984, 983, 980, 980, 979, 977]` | 0/20 agents | False |
| `I-09_n5.json` | 5 | Top-K Select | `[996, 992, 972, 965, 960, 959, 945, 939, 937, 934]` | 0/5 agents | False |
| `I-09_n50.json` | 50 | Top-K Select | `[1000, 999, 999, 996, 996, 995, 995, 994, 994, 994]` | 0/50 agents | False |
| `I-10_n10.json` | 10 | Standard Deviation | `29.04` | 0/10 agents | False |
| `I-10_n100.json` | 100 | Standard Deviation | `29.32` | 0/100 agents | False |
| `I-10_n2.json` | 2 | Standard Deviation | `30.58` | 0/2 agents | False |
| `I-10_n20.json` | 20 | Standard Deviation | `29.39` | 0/20 agents | False |
| `I-10_n5.json` | 5 | Standard Deviation | `29.0` | 0/5 agents | False |
| `I-10_n50.json` | 50 | Standard Deviation | `29.19` | 0/50 agents | False |

Instances where every agent's exact local answer equals the global answer: `I-03_n10.json`, `I-03_n100.json`, `I-03_n2.json`, `I-03_n20.json`, `I-03_n5.json`, `I-03_n50.json`, `I-04_n100.json`, `I-04_n5.json`.

This audit matched 60 fixed Paradigm-I instances using pattern `I-*_n*.json`. It checks exact local-task operations only; other paradigms and prior-based guessing are outside this audit.
