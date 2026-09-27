# Exploratory local pilot v0.1

This pilot compares an unconstrained control, a deliberately concise natural-language prompt, an AutoForm-style format-selection prompt, and a fixed JSON schema while holding the Silo-Bench P2P environment, Qwen3-1.7B receiver, task files, and decoding policy fixed. It is deliberately only three two-agent Level-I tasks, plus a no-communication control. It is a feasibility/error-analysis pilot, not a confirmatory study and cannot support superiority claims.

## Pinned inputs

- Task suite: [Silo-Bench](https://github.com/jwyjohn/acl26-silo-bench), upstream commit `e74127782ed1c42fff474249961f022c063d76f2` (Unlicense).
- Model: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (Apache-2.0).
- Exact model-shard SHA-256 values and policy text: [`policies.json`](policies.json).
- Expected task files: `I-01_n2.json`, `I-03_n2.json`, `I-04_n2.json` from the pinned benchmark checkout.

## Local run (PowerShell)

Use an environment with PyTorch/CUDA, Transformers, FastAPI, Uvicorn, OpenAI Python SDK, Pydantic, `srsly`, `httpx`, and `tenacity` installed. No cloud API key is used.

If the model is not already present:

```powershell
git clone --no-checkout https://github.com/jwyjohn/acl26-silo-bench.git .cache/upstream-silo-bench
git -C .cache/upstream-silo-bench checkout e74127782ed1c42fff474249961f022c063d76f2
hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir .cache/models/Qwen3-1.7B
```

Run the local model endpoint in one terminal:

```powershell
$env:TLU_MODEL_PATH = '.cache/models/Qwen3-1.7B'
uvicorn research.local_chat_server:app --host 127.0.0.1 --port 8000 --workers 1
```

Run the pilot in another terminal:

```powershell
python experiments/pilot_v0_1/run_silobench_pilot.py
```

Raw logs and detailed case folders are written to the Git-ignored `.cache/pilot_v0_1/`. Only anonymized aggregate results and methodological changes should be copied into this directory for publication; never commit model weights, credentials, or third-party working checkouts.

The runner makes one fixed, non-task warmup request before timing scenarios and rotates condition order across tasks to avoid always measuring one arm first. Decoding is greedy. These controls reduce simple order effects but do not make three tasks statistically representative.

## Reported measures

The runner records Silo-Bench success and partial correctness; model-reported input and output token totals; UTF-8 bytes in message payloads; exact simulator message-file bytes; messages, rounds, and end-to-end wall time. Tokens include repeated input context and generation, while message bytes isolate the payload and simulator JSON file. They are different accounting boundaries and should stay separate.
