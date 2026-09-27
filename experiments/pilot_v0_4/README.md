# DuoSum held-out calibration v0.4

This version carries the interaction calibration to four held-out episodes, one at each input width. It adds a common, format-neutral instruction requiring a bare integer in `submit_result`, but the held-out traces show that the instruction does not reliably control the answer form. The cases, model, engine, prompt, and weight hashes are pinned in [`policies.json`](policies.json).

Conditions share the same interaction scaffold and integer-answer instruction: no message-format hint, concise NL, compact key-value, JSON, base-2 digits, and no communication. This is a small protocol diagnostic, not a clean estimate of representation effects: strict success was 0/8 in five arms and 1/8 in binary, with representation adherence failures described below. No claim of general superiority follows from four episodes.

Trace audit added after the first analysis found that the model sent decimal strings in every `binary` episode and single-quoted Python-dict strings in every `json_schema` episode. Those two arms therefore did not instantiate their assigned formats; their outcomes and payload sizes are not evidence about binary or valid JSON communication. Compact-KV messages exactly matched their assigned format and value in 9/10 messages; concise-NL did so in 9/13. Some repeats carried the other agent's value, showing that syntax adherence alone is not semantic fidelity. The analyzer reports message-level adherence alongside task outcomes.

## Run locally

Use the same local Transformers server described in [`pilot_v0_3/README.md`](../pilot_v0_3/README.md). Install the upstream engine dependency into the ignored project cache and set the local environment:

```powershell
uv pip install --target .cache/python-packages srsly
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_MODEL_PATH = '.cache/models/Qwen3-1.7B'
```

Start the model endpoint, then run in a second terminal:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/pilot_v0_4/run_duosum_heldout.py
```

Analyze aggregate results with:

```powershell
python experiments/pilot_v0_4/analyze_duosum_heldout.py
```

Raw traces remain under ignored `.cache/pilot_v0_4/`. The report keeps exact-integer tool success separate from post-hoc semantic exactness and counts both total model tokens and message payload bytes.
