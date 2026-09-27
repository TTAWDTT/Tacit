# CRAFT v0.24 oracle communication-necessity ablation

Single-seed benchmark-validity audit; not a protocol comparison.

| Structure | Condition | Turns | Progress | Complete | Successful moves | Failed moves | Oracle-followed |
|---|---|---:|---:|---|---:|---:|---:|
| structure_001 (medium) | Natural-language team | 20 | 0.423 | False | 12 | 8 | 17 |
| structure_008 (simple) | Natural-language team | 20 | 0.622 | False | 16 | 4 | 16 |
| structure_003 (complex) | Natural-language team | 20 | 0.373 | False | 9 | 11 | 8 |
| structure_001 (medium) | Zero-message LLM Builder | 20 | 0.735 | False | 20 | 0 | 17 |
| structure_008 (simple) | Zero-message LLM Builder | 15 | 0.985 | True | 15 | 0 | 15 |
| structure_003 (complex) | Zero-message LLM Builder | 20 | 0.245 | False | 6 | 14 | 6 |
| structure_001 (medium) | Zero-message random oracle | 20 | 0.735 | False | 14 | 6 | 14 |
| structure_008 (simple) | Zero-message random oracle | 15 | 0.985 | True | 15 | 0 | 15 |
| structure_003 (complex) | Zero-message random oracle | 20 | 0.245 | False | 6 | 14 | 6 |

## Main observation

Both zero-message conditions completed the simple structure in 15 turns (upstream completion threshold: 0.95); the natural-language team did not complete any of the three structures within 20 turns. The random-policy trace chose a listed oracle action on all 15 turns and invoked no model. This is a counterexample to strict communication necessity for this structure under the oracle-assisted setup, not evidence that natural language is generally worse.

The oracle candidate list is target-grounded and the Builder never receives the Directors' private views directly. Therefore the result tests this benchmark configuration with its privileged action oracle. It does not answer whether ordinary LLM agents can collaborate without communication when the oracle is removed.

## Inference and implementation audit

The natural-language condition used 240 completed server calls; the zero-message LLM Builder used 60; the corrected random-oracle condition used 0. The corrected control's empty-candidate action is a clarification and does not call the model.
The first random-policy implementation made 14 unintended model calls after the candidate list became empty. Those data are excluded from the primary result and described in [the implementation audit](../experiments/craft_v0_24/IMPLEMENTATION_AUDIT.md). The corrected full three-structure control is the reported result.

This is one seed, three structures, and one local model. It establishes benchmark eligibility concerns only; it does not establish protocol superiority, generalization, or an efficiency frontier.
