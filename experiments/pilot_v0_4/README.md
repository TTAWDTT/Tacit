# DuoSum held-out calibration v0.4

This version carries the interaction calibration to four held-out seeds, one at each input width. It fixes the v0.3 answer-type failure by adding a common, format-neutral contract that requires the `submit_result` answer to be a bare integer. The held-out cases, model, engine, prompt, and weight hashes are pinned in [`policies.json`](policies.json).

Conditions share the same interaction scaffold and final-answer contract: no message-format hint, concise NL, compact key-value, JSON, base-2 digits, and no communication. This is a small test of message representation effects while reducing known control-protocol errors. No claim of general superiority follows from four episodes.

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
