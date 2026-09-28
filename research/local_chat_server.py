"""Minimal OpenAI-compatible Transformers server for local experiments.

This is an experimental bridge, not a production inference server. It runs one
request at a time and reports model-token usage measured by the loaded tokenizer.
"""

from __future__ import annotations

import os
import time
import uuid
import json
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "CUDA_CACHE_PATH", str(Path(__file__).resolve().parents[1] / ".cache" / "cuda")
)

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = os.environ.get("TLU_MODEL_PATH", ".cache/models/Qwen3-1.7B")
MODEL_NAME = os.environ.get("TLU_MODEL_NAME", "Qwen3-1.7B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TORCH_THREADS = max(1, int(os.environ.get("TLU_TORCH_THREADS", "2")))
torch.set_num_threads(TORCH_THREADS)
torch.set_num_interop_threads(1)
USAGE_LOG = os.environ.get("TLU_USAGE_LOG")
USAGE_LOG_LOCK = threading.Lock()

app = FastAPI(title="The Language That LLMs Use: local experiment endpoint")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE == "cpu":
    model.to(DEVICE)
model.eval()


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()
    request_started = time.perf_counter()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    except Exception as exc:  # pragma: no cover - surfaced as API error for callers
        raise HTTPException(status_code=400, detail=f"chat template failed: {exc}") from exc

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    max_new_tokens = max(1, min(int(body.get("max_tokens", 256)), 2048))
    generation_started = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
    except Exception as exc:  # pragma: no cover - surfaced as API error for callers
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
    completion_ids = generated[0, input_ids.shape[1] :]
    generation_seconds = time.perf_counter() - generation_started
    content = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    completion_tokens = int(completion_ids.numel())
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
                if completion_ids.numel() < max_new_tokens
                else "length",
            }
        ],
        "usage": {
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": completion_tokens,
            "total_tokens": int(input_ids.shape[1]) + completion_tokens,
        },
    }
    if USAGE_LOG:
        usage_record = {
            "request_id": response["id"],
            "model": MODEL_NAME,
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": completion_tokens,
            "generation_seconds": round(generation_seconds, 6),
            "request_seconds": round(time.perf_counter() - request_started, 6),
            "finish_reason": response["choices"][0]["finish_reason"],
        }
        log_path = Path(USAGE_LOG)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG_LOCK, log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(usage_record, separators=(",", ":")) + "\n")
    return JSONResponse(response)
