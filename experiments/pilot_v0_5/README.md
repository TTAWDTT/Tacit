# DuoSum held-out comparison v0.5

This protocol uses the remaining two held-out replicates at each input width (eight task episodes total) and compares seven arms: unformatted scaffold, concise natural language, compact key-value, JSON with an explicit example, binary with a decimal-to-binary example, AutoForm-style self-selection, and no communication. It adds explicit format examples because v0.4 found that the local model ignored the binary instruction and emitted invalid JSON. The protocol was frozen in [`policies.json`](policies.json) before held-out calls.

The primary diagnostic separates upstream strict tool success, a deterministic semantic answer score, fixed-format syntax validity, and whether each message decodes to its sender's actual private value. AutoForm has no fixed syntax; its prompt follows the released AutoForm experiment wording, adapted to message communication. Message bytes and all model input/output tokens remain separate, and this run does not impose matched budgets.

The semantic grader accepts an integer, a decimal-integer string, or a full `a+b=c` expression only if its arithmetic is correct and the stated result matches the task gold answer. Raw submissions remain unchanged. See [`benchmarks/duosum_v0_1/grading.py`](../../benchmarks/duosum_v0_1/grading.py).

## Run locally

Start the local model endpoint from [`research/local_chat_server.py`](../../research/local_chat_server.py), then:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/pilot_v0_5/run_duosum_heldout.py
```

After all calls complete:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/pilot_v0_5/analyze_duosum_heldout.py
```

Raw traces are stored only in ignored `.cache/pilot_v0_5/`. Eight episodes on one 1.7B model remain exploratory and cannot establish general format superiority or a scaling law.
