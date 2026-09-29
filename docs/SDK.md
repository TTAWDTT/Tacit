# Tacit runtime SDK (experimental)

Tacit includes a small, protocol-neutral Python runtime for passing one LLM-generated message to another LLM. It is designed to make protocol comparisons runnable and auditable while research continues. It does not designate a winning language or imply a measured performance advantage.

## Minimal exchange

Implement a protocol as a stable identifier plus sender and receiver instructions, then provide any two clients that implement `complete(messages) -> ChatCompletion`:

```python
from tacit import OpenAICompatibleClient, exchange_once

class Protocol:
    protocol_id = "evidence-summary-v0"
    sender_instruction = "Separate observations from inferences; include uncertainty."
    receiver_instruction = "Evaluate the message as evidence and state remaining uncertainty."

sender = OpenAICompatibleClient("http://localhost:8000/v1", "sender-model")
receiver = OpenAICompatibleClient("http://localhost:8001/v1", "receiver-model")
result = exchange_once(
    sender, receiver, protocol=Protocol(),
    sender_context="Facts visible to sender", receiver_context="Facts visible to receiver",
    receiver_task="Combine evidence and answer the task",
)
print(result.message, result.receiver.text)
```

From a checkout, install the package in editable mode with `python -m pip install -e .`. No runtime dependencies are installed. On PowerShell, configure the example endpoints before running it:

```powershell
$env:TLU_SENDER_URL = "http://localhost:8000/v1"
$env:TLU_RECEIVER_URL = "http://localhost:8001/v1"
python examples/two_agent_exchange.py
```

The adapter uses only Python's standard library and is compatible with OpenAI-style `/chat/completions` endpoints, including the repository's optional local server. Set `TLU_SENDER_URL`, `TLU_RECEIVER_URL`, model names, and API keys to configure the runnable example in [`examples/two_agent_exchange.py`](../examples/two_agent_exchange.py). The adapter does not load a model until a request is made; this SDK itself does not start a server or download weights.

## Accounting boundary

`ExchangeResult` retains the sender's exact returned string and measures its UTF-8 payload bytes. It does not trim, parse, compress, or repair the message. Provider-reported prompt/completion token counts and service time are preserved when available; missing values remain missing. `transmission_record(...)` emits one `tlu.costs.v3` transmission entry. The caller must supply the actual transport boundary and measured framing bytes. The receiver prompt includes private context and the verbatim message; its total input tokens are model inference cost, not message-only channel tokens. Supply `recipient_tokenizer` and `recipient_payload_tokens` together only when the delivered payload was separately counted with that tokenizer.

For experiments that need a concrete agent boundary, [`LocalTCPMessageChannel`](../tacit/channel.py) sends a length-prefixed JSON envelope over an ephemeral `127.0.0.1` TCP socket. It waits for the receiver callback and charges the one-byte delivery acknowledgment. `Transmission.cost_record()` partitions the serialized JSON payload field from the envelope, length prefix, and acknowledgment; `logical_text_utf8_bytes` is kept separately in metadata. Its declared boundary is **application-layer loopback bytes**: TCP/IP and link-layer headers are excluded, so do not report it as a physical-network measurement. The channel binds only to loopback and is intended for local experiments and tests.

For codecs that already produce bytes, [`LocalTCPFrameChannel`](../tacit/channel.py) transmits a bounded UTF-8 JSON metadata header followed by the exact raw payload bytes. Its `tlu.frame.v1` wire format has a four-byte metadata-length prefix and one-byte callback acknowledgment. `FrameTransmission.cost_record()` reports raw payload bytes separately from metadata, prefix, and acknowledgment bytes, and records `media_type`, `encoding`, and caller-supplied JSON metadata such as tensor dtype and shape. This is an opaque transport: it does not encode model outputs, decode a received representation into model input, or make arbitrary bytes consumable by a text-only LLM. Those codec and model-adapter costs must be implemented and measured by the experiment using it.

The first release supports one sender and one receiver call, text model endpoints, and measured loopback text or opaque-byte transport. Multi-round scheduling, retries, tool calls, built-in latent codecs/model adapters, automated tokenizer-specific payload counts, and production serving are outside this SDK version. This is an experimental research interface; compare protocols only with matched prompts, task splits, model populations, and complete costs as described in [`COST_ACCOUNTING.md`](COST_ACCOUNTING.md).
