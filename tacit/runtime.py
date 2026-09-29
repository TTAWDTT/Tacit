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
from urllib.request import Request, urlopen


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
            with urlopen(request, timeout=self.timeout_seconds) as response:
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
        )


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
