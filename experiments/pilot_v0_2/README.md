# Interaction calibration v0.2

This run asks whether a minimal, shared collaboration scaffold repairs the receiver-side tool-following failure observed in v0.1. It is a calibration experiment on one valid two-agent Global Max instance, not a language comparison or confirmatory study.

## Frozen conditions

- `unconstrained`: no additional interaction or message-format prompt.
- `scaffold_only`: add the same concise interaction policy to both agents.
- `concise_nl_scaffold`: keep that interaction policy and add one-sentence plain-English message guidance.
- `no_communication`: disable the send action and add no collaboration scaffold.

The Global Max instance passes the local-answer sufficiency check: agent 0's local maximum is 827, agent 1's is 780, and the global maximum is 827. The non-communicating pair cannot both derive the global answer by applying the operation to their own shards.

All arms use the same upstream Silo-Bench P2P engine, Qwen3-1.7B model revision and shard checksums, greedy decoding, 256-token default response cap, and four-round limit. The scaffold text is recorded verbatim in [`policies.json`](policies.json), and its repeated prompt tokens count toward total input tokens. The runner also counts send, receive, submit, and self-send tool actions from raw simulator logs.

## Run locally

Use an environment with PyTorch/CUDA, Transformers, FastAPI, Uvicorn, OpenAI Python SDK, Pydantic, HTTPX, and Tenacity installed. The upstream engine additionally needs `srsly`; this command installs its small dependency under the ignored project cache:

```powershell
uv pip install --target .cache/python-packages srsly
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_MODEL_PATH = '.cache/models/Qwen3-1.7B'
```

Start the local endpoint in one terminal:

```powershell
python -m uvicorn research.local_chat_server:app --host 127.0.0.1 --port 8000 --workers 1
```

Run calibration in another terminal:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/pilot_v0_2/run_silobench_calibration.py
```

Raw logs remain in the ignored `.cache/pilot_v0_2/` directory. Only aggregate measurements and analysis belong in the public research record.

## Interpretation gate

The interaction scaffold is useful only if it reduces self-sends/repeated polling and improves complete task success. A comparison between the scaffold arms and the unconstrained arm measures an instruction intervention with its input-token cost. The concise-format effect is the difference between the two scaffold arms. A single deterministic task/model run can diagnose mechanics; it cannot establish robust gains.
