# DuoSum cross-model replication v0.6

This is a paired replication of the frozen v0.5 held-out comparison on the same eight DuoSum episodes and the same seven communication conditions. Its purpose is to test whether v0.5's format-adherence and task-success observations persist with a larger local model. It does not isolate model scale: v0.5 used Qwen3-1.7B with Transformers, while this run uses Qwen3-4B Q4_K_M with llama.cpp.

The conditions, task files, message instructions, evaluator, and deterministic semantic diagnostic are unchanged from v0.5. The v0.6 inference environment is pinned in [`policies.json`](policies.json). The protocol was frozen before any v0.6 model calls. The server and runner use localhost only; raw traces remain under ignored `.cache/pilot_v0_6/`.

## Runtime

Download the pinned Q4_K_M GGUF and Windows CUDA 12.4 llama.cpp b11202 release assets into `.cache/models/` and `.cache/runtimes/`, verifying the SHA-256 digests in `policies.json`. Start the local server (consult `llama-server.exe --help` for the pinned binary):

```powershell
& .cache/runtimes/llama.cpp/llama-server.exe `
  --model .cache/models/Qwen3-4B-Q4_K_M.gguf `
  --alias Qwen3-4B-Q4_K_M `
  --host 127.0.0.1 --port 8000 --n-gpu-layers 99 --ctx-size 8192 `
  --temp 0 --chat-template-kwargs '{"enable_thinking":false}'
```

Then run the frozen episodes:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_GGUF_PATH = '.cache/models/Qwen3-4B-Q4_K_M.gguf'
python experiments/pilot_v0_6/run_duosum_heldout.py
python experiments/pilot_v0_6/analyze_duosum_heldout.py
```

The runner makes one warm-up call before the 56 episode-condition runs. All model requests target `127.0.0.1`. Greedy decoding and a 256-token completion cap are used to match v0.5 as closely as this backend allows. Disable thinking through the chat template. Report token counts as backend-reported values and avoid comparing them as though tokenizer/backend accounting were necessarily identical.

Eight paired episodes on one quantized model remain exploratory. Any differences are confounded by parameter count, quantization, runtime, and model-specific chat templating; they cannot establish a pure scaling effect or general protocol superiority.
