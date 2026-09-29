"""Protocol-neutral two-model text exchange and OpenAI-compatible adapter.

The runtime records the exact generated message that is delivered. It does not
silently normalize, compress, or repair a protocol message.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from tacit.channel import LocalTCPMessageChannel, Transmission


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ChatCompletion:
    """One completed chat request with provider-reported usage when available."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    service_seconds: float | None = None
    request_id: str | None = None
    finish_reason: str | None = None


class ChatModel(Protocol):
    """Minimal interface implemented by a model client."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> ChatCompletion:
        """Return one assistant response to the supplied chat messages."""


class TextProtocol(Protocol):
    """Instructions that define how sender and receiver use a wire message."""

    @property
    def protocol_id(self) -> str: ...

    @property
    def sender_instruction(self) -> str: ...

    @property
    def receiver_instruction(self) -> str: ...


class DialogueProtocol(Protocol):
    """Fixed instructions for a multi-turn two-agent dialogue protocol."""

    @property
    def protocol_id(self) -> str: ...

    @property
    def agent_instructions(self) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class ExchangeResult:
    """Auditable result of one sender-to-receiver exchange."""

    protocol_id: str
    message: str
    payload_bytes: int
    sender: ChatCompletion
    receiver: ChatCompletion

    def transmission_record(
        self,
        *,
        round_number: int,
        sender_name: str,
        recipient_name: str,
        transport_boundary: str,
        framing_bytes: int = 0,
        recipient_tokenizer: str | None = None,
        recipient_payload_tokens: int | None = None,
    ) -> JsonObject:
        """Return a tlu.costs.v3 transmission entry for a text payload.

        ``framing_bytes`` must be measured at the chosen transport boundary;
        this helper cannot infer HTTP or application framing.
        """
        if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
            raise ValueError("round_number must be a positive integer")
        if isinstance(framing_bytes, bool) or not isinstance(framing_bytes, int) or framing_bytes < 0:
            raise ValueError("framing_bytes must be a non-negative integer")
        if transport_boundary not in {"network", "inter_process"}:
            raise ValueError("transport_boundary must be 'network' or 'inter_process'")
        row: JsonObject = {
            "round": round_number,
            "sender": sender_name,
            "recipients": [recipient_name],
            "payload_bytes": self.payload_bytes,
            "framing_bytes": framing_bytes,
            "encoding": "utf-8",
            "media_type": "text/plain; charset=utf-8",
            "transport_boundary": transport_boundary,
            "payload_metadata": {"protocol_id": self.protocol_id},
        }
        if (recipient_tokenizer is None) != (recipient_payload_tokens is None):
            raise ValueError(
                "recipient_tokenizer and recipient_payload_tokens must be provided together"
            )
        if recipient_payload_tokens is not None:
            if (
                isinstance(recipient_payload_tokens, bool)
                or not isinstance(recipient_payload_tokens, int)
                or recipient_payload_tokens < 0
            ):
                raise ValueError("recipient_payload_tokens must be a non-negative integer")
            row["recipient_tokens"] = {
                recipient_name: {
                    "tokenizer": recipient_tokenizer,
                    "tokens": recipient_payload_tokens,
                }
            }
        return row


@dataclass(frozen=True)
class DialogueTurn:
    """One scheduled model call and the exact message delivered to its peer."""

    speaker: str
    recipient: str
    completion: ChatCompletion
    transmission: Transmission


@dataclass(frozen=True)
class DialogueResult:
    """Complete fixed-schedule dialogue with call and wire-cost accessors."""

    protocol_id: str
    turns: tuple[DialogueTurn, ...]

    @property
    def model_calls(self) -> int:
        return len(self.turns)

    @property
    def wire_bytes(self) -> int:
        return sum(turn.transmission.total_application_bytes for turn in self.turns)

    def transmission_records(self) -> list[JsonObject]:
        """Return one exact tlu.costs.v3 transmission record per scheduled turn."""
        return [turn.transmission.cost_record() for turn in self.turns]

    def model_call_records(
        self,
        *,
        tokenizers: Mapping[str, str] | None = None,
    ) -> list[JsonObject]:
        """Return provider usage by agent; unknown tokenizer IDs stay explicit."""
        tokenizer_map = {} if tokenizers is None else tokenizers
        speakers = {turn.speaker for turn in self.turns}
        if not isinstance(tokenizer_map, Mapping) or any(
            name not in speakers or not isinstance(tokenizer, str) or not tokenizer.strip()
            for name, tokenizer in tokenizer_map.items()
        ):
            raise ValueError("tokenizers must map participating agent names to non-empty identifiers")
        return [
            {
                "agent": turn.speaker,
                "stage": "dialogue_turn",
                "model": turn.completion.model,
                "tokenizer": tokenizer_map.get(turn.speaker, "not_reported"),
                "input_tokens": turn.completion.input_tokens,
                "output_tokens": turn.completion.output_tokens,
                "service_seconds": turn.completion.service_seconds,
                "retry": False,
                "truncated": turn.completion.finish_reason == "length",
            }
            for turn in self.turns
        ]


@dataclass(frozen=True)
class OpenAICompatibleClient:
    """Dependency-free client for local or remote OpenAI-compatible endpoints.

    The endpoint is called only when ``complete`` is invoked. API keys are
    supplied explicitly and are never logged by this client.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 120.0
    max_tokens: int = 512
    follow_redirects: bool = True

    def complete(self, messages: Sequence[Mapping[str, str]]) -> ChatCompletion:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(
            {"model": self.model, "messages": list(messages), "max_tokens": self.max_tokens},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            if self.follow_redirects:
                response_context = urlopen(request, timeout=self.timeout_seconds)
            else:
                response_context = build_opener(_RejectRedirectHandler).open(
                    request, timeout=self.timeout_seconds
                )
            with response_context as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"chat endpoint returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"could not reach chat endpoint: {exc.reason}") from exc
        elapsed = time.perf_counter() - started

        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"]
            model_name = payload.get("model") or self.model
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("chat endpoint returned an invalid completion response") from exc
        if not isinstance(text, str):
            raise RuntimeError("chat endpoint completion content must be text")
        usage = payload.get("usage") or {}
        return ChatCompletion(
            text=text,
            model=str(model_name),
            input_tokens=_optional_nonnegative_int(usage.get("prompt_tokens")),
            output_tokens=_optional_nonnegative_int(usage.get("completion_tokens")),
            service_seconds=_optional_nonnegative_number(
                usage.get("generation_seconds", usage.get("service_seconds", elapsed))
            ),
            request_id=payload.get("id"),
            finish_reason=choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
        )


class _RejectRedirectHandler(HTTPRedirectHandler):
    """Reject redirects to prevent loopback-only calls forwarding prompts off-host."""

    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def exchange_once(
    sender_model: ChatModel,
    receiver_model: ChatModel,
    *,
    protocol: TextProtocol,
    sender_context: str,
    receiver_context: str,
    receiver_task: str,
) -> ExchangeResult:
    """Ask one model to send a message and another to use that exact message."""
    sender_result = sender_model.complete(
        [
            {"role": "system", "content": protocol.sender_instruction},
            {"role": "user", "content": sender_context},
        ]
    )
    wire_message = sender_result.text
    receiver_result = receiver_model.complete(
        [
            {"role": "system", "content": protocol.receiver_instruction},
            {
                "role": "user",
                "content": (
                    f"Private context: {receiver_context}\n\n"
                    f"Message from the other agent (verbatim):\n{wire_message}\n\n"
                    f"Task: {receiver_task}"
                ),
            },
        ]
    )
    return ExchangeResult(
        protocol_id=protocol.protocol_id,
        message=wire_message,
        payload_bytes=len(wire_message.encode("utf-8")),
        sender=sender_result,
        receiver=receiver_result,
    )


def exchange_dialogue(
    agents: Mapping[str, ChatModel],
    *,
    protocol: DialogueProtocol,
    private_contexts: Mapping[str, str],
    schedule: Sequence[str],
    task: str,
    max_turns: int,
    channel_timeout_seconds: float = 30.0,
) -> DialogueResult:
    """Run a fixed-schedule, two-agent exchange over measured loopback TCP.

    At each turn, an agent sees only its own private context and the public
    transcript delivered so far. The caller controls the exact schedule and
    turn cap; this function does not infer a stopping signal from message text.
    """
    names = set(agents)
    if len(names) != 2 or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("agents must contain exactly two non-empty string names")
    if set(private_contexts) != names:
        raise ValueError("private_contexts must contain exactly one entry per agent")
    if any(not isinstance(context, str) for context in private_contexts.values()):
        raise ValueError("each private context must be a string")
    instructions = protocol.agent_instructions
    if not isinstance(protocol.protocol_id, str) or not protocol.protocol_id.strip():
        raise ValueError("protocol_id must be a non-empty string")
    if not isinstance(instructions, Mapping) or set(instructions) != names:
        raise ValueError("agent_instructions must contain exactly one instruction per agent")
    if any(not isinstance(text, str) or not text.strip() for text in instructions.values()):
        raise ValueError("each agent instruction must be a non-empty string")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence) or not schedule:
        raise ValueError("schedule must be a non-empty sequence of agent names")
    if len(schedule) > max_turns:
        raise ValueError("schedule exceeds max_turns")
    if any(not isinstance(name, str) or name not in names for name in schedule):
        raise ValueError("schedule contains an unknown agent")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    transcript: list[dict[str, str]] = []
    turns: list[DialogueTurn] = []
    received: list[dict[str, Any]] = []
    with LocalTCPMessageChannel(received.append, timeout_seconds=channel_timeout_seconds) as channel:
        for index, speaker in enumerate(schedule, start=1):
            recipient = next(name for name in names if name != speaker)
            messages = [
                {"role": "system", "content": instructions[speaker]},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task,
                            "private_context": private_contexts[speaker],
                            "public_transcript": transcript,
                            "instruction": "Send exactly the next message for this turn.",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ]
            completion = agents[speaker].complete(messages)
            received_before = len(received)
            transmission = channel.send(
                completion.text,
                protocol_id=protocol.protocol_id,
                round_number=index,
                sender=speaker,
                recipient=recipient,
            )
            delivered = received[received_before:]
            if len(delivered) != 1 or delivered[0].get("payload") != completion.text:
                raise RuntimeError("loopback channel did not deliver the exact sender message")
            transcript.append({"sender": speaker, "message": delivered[0]["payload"]})
            turns.append(DialogueTurn(speaker, recipient, completion, transmission))

    return DialogueResult(protocol.protocol_id, tuple(turns))


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_nonnegative_number(value: Any) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return float(value)
