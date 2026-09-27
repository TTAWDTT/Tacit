# DuoSum format pilot v0.3

This exploratory pilot compares the default message style, concise NL, a compact key-value form, and JSON while holding task, model, interaction scaffold, engine, and greedy decoding fixed. It also runs a no-communication control. Four calibration episodes span 4, 8, 12, and 16 input bits. Held-out episodes are reserved for later replication.

The benchmark task is private exact sum: both agents must output `x + y`, with one positive private integer per agent. Each local input is strictly less than the gold sum. The task manifest, exact files, model revision, model-shard hashes, interaction scaffold, and Silo-Bench simulator commit are pinned in [`policies.json`](policies.json). The common scaffold is counted in total model prompt tokens. Format instructions are added only to message-enabled arms.

## Local run

Use Python with PyTorch/CUDA, Transformers, FastAPI, Uvicorn, OpenAI Python SDK, Pydantic, HTTPX, Tenacity, and the upstream engine's `srsly` dependency. Install the small missing dependency inside the ignored project cache:

```powershell
uv pip install --target .cache/python-packages srsly
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_MODEL_PATH = '.cache/models/Qwen3-1.7B'
```

Start the local model server:

```powershell
python -m uvicorn research.local_chat_server:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal, run the pilot:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/pilot_v0_3/run_duosum_pilot.py
```

Raw traces are written only under the ignored `.cache/pilot_v0_3/`. The runner checks all pinned hashes before calling the model, verifies every local input is insufficient for the sum, rotates condition order, and counts send/receive/submit/self-send tool actions.

This pilot does not equalize communication budgets and has only one instance at each input width. It is a feasibility and diagnostic run. It cannot establish a general representation advantage or scaling law.
