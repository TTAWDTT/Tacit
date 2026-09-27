# Qwen3-8B local scale diagnostic v0.10

This is a model-capability diagnostic following v0.8 and v0.9. The same two hybrid roles and same 12 task seeds are used, with the local model changed from Qwen3-4B to Qwen3-8B. The comparison is preregistered at [`preregistration.json`](preregistration.json) and does not rank communication formats.

The model asset is pinned to Hugging Face repository revision `7c41481f57cb95916b40956ab2f0b139b296d974`, file `Qwen3-8B-Q4_K_M.gguf`, 5,027,783,488 bytes, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`. The official [Qwen3-8B GGUF repository](https://huggingface.co/Qwen/Qwen3-8B-GGUF) documents Q4_K_M and llama.cpp usage.

## Run

Download the pinned asset into this project's ignored cache (the Python downloader uses the pinned repository revision):

```powershell
$env:PYTHONPATH = '.cache/python-packages'
@'
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id='Qwen/Qwen3-8B-GGUF',
    filename='Qwen3-8B-Q4_K_M.gguf',
    revision='7c41481f57cb95916b40956ab2f0b139b296d974',
    local_dir='.cache/models',
))
'@ | python -
```

Start the already pinned llama.cpp server with this alias and model path:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath '.cache/runtimes/llama.cpp/llama-server.exe' `
  -ArgumentList @('--model','.cache/models/Qwen3-8B-Q4_K_M.gguf', `
    '--alias','Qwen3-8B-Q4_K_M','--host','127.0.0.1','--port','8000', `
    '--n-gpu-layers','99','--ctx-size','8192','--temp','0', `
    '--n-predict','256','--reasoning','off')
```

Confirm `http://127.0.0.1:8000/health` returns 200, then run:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_GGUF_PATH = '.cache/models/Qwen3-8B-Q4_K_M.gguf'
python experiments/pilot_v0_10/run_scale_hybrid.py
```

The runner checks the pinned GGUF size and checksum, task manifest, and upstream engine commit before model inference. If the full GPU offload does not load on the local 8GB card, stop before inference, record and commit the fallback settings, then continue with a documented deviation. Raw traces stay in ignored `.cache/pilot_v0_10/`; the analyzer runs after completion and can later rebuild the public report without model calls.
