# CRAFT public trace audit v0.2

This is a logging and information-path audit, not a rerun of CRAFT or an evaluation of a Tacit protocol.

## Data and method

- Source: [Abhijnan/craft-benchmark-lean](https://huggingface.co/datasets/Abhijnan/craft-benchmark-lean), dataset revision `2aecfe0cbb7f1c6743454df5a65e65d573287712`, Apache-2.0.
- File: `data/train-00000-of-00001.parquet`, 188,315,472 bytes, SHA-256 `1651c21993547f423185d5285b48f7dbb37476b14fe3f9a607639f8d3a42223f`.
- The file contains 5,946 turn rows from 300 model × structure × run games, across 15 Director models.
- For each turn, `conversation_snapshot` records the append-only conversation history. The pinned runner trims histories longer than 50 entries to the last 40 after writing each snapshot. The audit accounts for that trim, then counts new `D1:`/`D2:`/`D3:` posts and compares them with the non-empty `D1_message`/`D2_message`/`D3_message` fields used to construct the Builder transcript from the `director_responses` dictionary.
- Sixteen snapshots could not be aligned to their prior context; these are excluded. The remaining 5,930 turns align, and each contains exactly three new Director posts.

## Findings

| Measure | Result |
|---|---:|
| Aligned turns with 3 Director posts in shared history | 5,930/5,930 |
| Aligned turns where posts exceed per-role Builder messages | 4,652/5,930 (78.5%) |
| Director posts collapsed before Builder transcript | 5,349/17,790 (30.1%) |
| Aligned turns with 1 / 2 / 3 Builder messages | 697 / 3,955 / 1,278 |

All 15 models show this behavior. By model, 283–330 aligned turns contain at least one collapsed post, and 321–385 posts are collapsed. Full counts are in the [machine-readable results](CRAFT_TRACE_DATASET_AUDIT_V0_2.json); the script is [audit_craft_trace_dataset.py](audit_craft_trace_dataset.py).

This confirms an information-path asymmetry in the public trace set: a repeated-ID response remains in the shared history seen by later Directors, but the Builder's current-turn transcript retains only the last dictionary value for that ID. It changes both which content reaches the receiver and how generated-message cost maps to delivered-message cost. The data do not identify whether this was present in the exact revision used for every paper result, so this is a property of the inspected released runner and traces, not a claim that the paper's conclusions are invalid.

## Consequence for Tacit

Do not use this released trace set as a clean natural-language communication baseline. A faithful local reproduction must first state and pin a director scheduling policy, preserve all selected messages or deliberately define the deduplication rule, and give the Builder and future Directors exactly the transcripts specified by that policy. Generated posts and delivered messages must be counted separately. Then reproduce a natural-language run before changing the communication representation.
