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

For a frozen, shareable two-party protocol, use `ProtocolCard` instead of defining a local class. The JSON schema is `tlu.shared_protocol_card.v1`; parsing rejects extra or duplicate fields and enforces the same 128-character ID and 32 KiB per-instruction limits used by the v0.4 experiment:

```python
from pathlib import Path
from hashlib import sha256
from tacit import OpenAICompatibleClient, ProtocolCard, exchange_once

card_bytes = Path("protocol-card.json").read_bytes()
card = ProtocolCard.from_json(card_bytes)
card_source_sha256 = sha256(card_bytes).hexdigest()  # hash the exact shared artifact
result = exchange_once(
    OpenAICompatibleClient("http://localhost:8000/v1", "sender-model"),
    OpenAICompatibleClient("http://localhost:8001/v1", "receiver-model"),
    protocol=card,
    sender_context="Facts visible to sender",
    receiver_context="Facts visible to receiver",
    receiver_task="Combine evidence and answer the task",
)
```

`ProtocolCard.to_json_bytes()` emits deterministic compact UTF-8 JSON for creating a card. Keep and hash the exact bytes distributed to participants when recording provenance; reserializing a card can change that artifact hash. A frozen instruction card makes a protocol portable and identifiable. It does not establish shared semantics, compositionality, robustness, or a performance advantage; those remain empirical questions. Any protocol induction, tuning, or onboarding cost must be measured separately.

The runnable [`protocol_card_exchange.py`](../examples/protocol_card_exchange.py) uses the accompanying [`protocol_card.json`](../examples/protocol_card.json), reports its exact source hash, and records message bytes and provider token counts. The card is an illustrative natural-language baseline. Install the package in editable mode, start two compatible endpoints, set the same endpoint variables as the minimal exchange above, and run:

```powershell
python examples/protocol_card_exchange.py
```

Set `TLU_PROTOCOL_CARD` to use another frozen card. The example contacts configured endpoints only when executed; importing the module does not make requests or start a server.

From a checkout, install the package in editable mode with `python -m pip install -e .`. No runtime dependencies are installed. On PowerShell, configure the example endpoints before running it:

```powershell
$env:TLU_SENDER_URL = "http://localhost:8000/v1"
$env:TLU_RECEIVER_URL = "http://localhost:8001/v1"
python examples/two_agent_exchange.py
```

The adapter uses only Python's standard library and is compatible with OpenAI-style `/chat/completions` endpoints, including the repository's optional local server. Set `TLU_SENDER_URL`, `TLU_RECEIVER_URL`, model names, and API keys to configure the runnable example in [`examples/two_agent_exchange.py`](../examples/two_agent_exchange.py). The client omits the API key from its object representation, rejects HTTP redirects by default, and validates endpoint, timeout, and token-limit configuration before making a request. Set `follow_redirects=True` only when the configured endpoint's redirect targets are trusted. The adapter does not load a model until a request is made; this SDK itself does not start a server or download weights.

## Fixed-schedule multi-agent exchange

For tasks that need clarification or staged evidence, `exchange_dialogue(...)` supports one or more agents when a sealed `final_answer_agent` is specified, and two or more agents for any routed message schedule. An empty schedule is the true no-message control: it makes only the sealed answer call and emits no transmission records. Each non-empty schedule item is `(sender, recipient)`; the legacy sequence of agent names remains available for exactly two agents and routes each turn to the other participant. Each model call receives its own private context and only the transcript of messages it sent or received. Messages routed between other agents do not leak into its prompt. Every scheduled message crosses `LocalTCPMessageChannel`; `DialogueResult` exposes exact per-turn sender/recipient records, model-call usage, and summed application-layer bytes. Set `wire_budget_bytes` to enforce a hard aggregate cap over the serialized JSON payloads, envelopes, length prefixes, and acknowledgments. A completion that would exceed the remaining budget is recorded as a model call but is not delivered; `stop_reason` becomes `wire_budget_exhausted`. This preserves inference costs for rejected messages. The cap excludes TCP/IP headers and does not cap generation tokens or model compute, which remain separately reported. Set `final_answer_agent` to make one post-dialogue model call whose `final_submission` is sealed for an external scorer; it is included in model-call records but is never transmitted or charged as channel bytes. The final call still occurs if the message budget ends the schedule early, so the task can be scored from the messages actually delivered. The schedule and `max_turns` are explicit, so the SDK never treats message text as an implicit stop signal. This is a fixed-policy unicast primitive; it does not provide dynamic routing, retries, or broadcast.

[`examples/multi_turn_exchange.py`](../examples/multi_turn_exchange.py) shows a three-turn `A → B → A` run under a 4,096-byte application-wire cap. [`examples/three_agent_exchange.py`](../examples/three_agent_exchange.py) shows an explicit five-turn unicast schedule among three agents followed by a sealed final submission from C. Both contact only configured endpoints and do not start a server or load a model. Use a passing resource preflight and endpoint-specific safeguards before executing local inference.

[`examples/multiparty_private_sum.py`](../examples/multiparty_private_sum.py) gives each sender one private value and asks a separate receiver to sum only the delivered messages. It includes a no-message baseline, a full-information receiver control, strict final-answer scoring, and separate per-sender syntax, value-fidelity, and delivery outcomes. This separation catches cases where sender errors cancel and leave the final sum accidentally correct. It also records transport/model-call costs, with a 4,096-byte wire cap and a 12-call planned-request cap. It is a runnable SDK illustration rather than a frozen benchmark; no model result or protocol advantage is implied. First create a passing report for the sender and receiver ports with `./experiments/emergent_ood_v0_3/resource_preflight.ps1 -Ports 8000,8001`, then invoke `python examples/multiparty_private_sum.py --resource-preflight .cache/emergent_ood_v0_3/resource_preflight.json --values 1 2 3` within five minutes. Configure `TLU_SUM_SENDER_URL`, `TLU_SUM_RECEIVER_URL`, and model/API-key variables as needed. The example only permits loopback HTTP endpoints, never starts a server, and never loads weights.

For reusable role instructions in a multi-agent schedule, `DialogueProtocolCard` implements the `tlu.dialogue_protocol_card.v1` schema with a frozen mapping from agent names to instructions. The existing three-agent example loads [`dialogue_protocol_card.json`](../examples/dialogue_protocol_card.json) and prints its exact source hash; set `TLU_DIALOGUE_PROTOCOL_CARD` to point at another card. The card defines role instructions only. The caller still supplies the agent roster, routing schedule, turn cap, stop rule, and wire budget. Its example instructions are plain-language scaffolding, not a measured protocol candidate.

## Accounting boundary

`ExchangeResult` retains the sender's exact returned `message.content` string and measures its UTF-8 payload bytes. It does not trim, parse, compress, or repair the message. Provider-reported prompt/completion token counts and service time are preserved when available; missing values remain missing. For endpoints that return a separate `reasoning_content` field, `ChatCompletion.reasoning_content_present` records only whether that field contained text; its contents are never copied into the completion or transmitted message. Completion-token usage remains the provider's complete reported count, so reasoning generation is still reflected in inference cost. `transmission_record(...)` emits one `tlu.costs.v3` transmission entry. The caller must supply the actual transport boundary and measured framing bytes. The receiver prompt includes private context and the verbatim message; its total input tokens are model inference cost, not message-only channel tokens. Supply `recipient_tokenizer` and `recipient_payload_tokens` together only when the delivered payload was separately counted with that tokenizer.

For experiments that need a concrete agent boundary, [`LocalTCPMessageChannel`](../tacit/channel.py) sends a length-prefixed JSON envelope over an ephemeral `127.0.0.1` TCP socket. It waits for the receiver callback and charges the one-byte delivery acknowledgment. `Transmission.cost_record()` partitions the serialized JSON payload field from the envelope, length prefix, and acknowledgment; `logical_text_utf8_bytes` is kept separately in metadata. Its declared boundary is **application-layer loopback bytes**: TCP/IP and link-layer headers are excluded, so do not report it as a physical-network measurement. The channel binds only to loopback and is intended for local experiments and tests.

For codecs that already produce bytes, [`LocalTCPFrameChannel`](../tacit/channel.py) transmits a bounded UTF-8 JSON metadata header followed by the exact raw payload bytes. Its `tlu.frame.v1` wire format has a four-byte metadata-length prefix and one-byte callback acknowledgment. `FrameTransmission.cost_record()` reports raw payload bytes separately from metadata, prefix, and acknowledgment bytes, and records `media_type`, `encoding`, and caller-supplied JSON metadata such as tensor dtype and shape. This is an opaque transport: it does not encode model outputs, decode a received representation into model input, or make arbitrary bytes consumable by a text-only LLM. Those codec and model-adapter costs must be implemented and measured by the experiment using it.

This release supports one-turn exchanges between two endpoints and fixed-schedule unicast dialogues among two or more agents, plus one-agent sealed-answer controls with no transmissions. It includes text endpoints, sealed final submissions, and measured loopback text or opaque-byte transport. The dialogue byte cap constrains delivered application-wire bytes; it does not make model inference cost equal across protocols. Dynamic scheduling, broadcast, retries, tool calls, built-in latent codecs/model adapters, automated tokenizer-specific payload counts, and production serving are outside this SDK version. This is an experimental research interface; compare protocols only with matched prompts, task splits, model populations, and complete costs as described in [`COST_ACCOUNTING.md`](COST_ACCOUNTING.md).
