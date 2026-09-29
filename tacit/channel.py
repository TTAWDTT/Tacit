"""Measured loopback TCP channel for serialized inter-agent text messages.

This is an application-layer research transport: a length-prefixed UTF-8 JSON
envelope sent over a real loopback TCP socket. It counts application bytes,
including the four-byte length prefix, and excludes TCP/IP/link-layer headers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import socketserver
import struct
import threading
from typing import Any, Callable


MAX_ENVELOPE_BYTES = 8 * 1024 * 1024
MAX_FRAME_BYTES = 64 * 1024 * 1024
MESSAGE_SCHEMA = "tlu.message.v1"
FRAME_SCHEMA = "tlu.frame.v1"


@dataclass(frozen=True)
class Transmission:
    """One delivered message and its measured application-layer costs."""

    message: str
    protocol_id: str
    round_number: int
    sender: str
    recipient: str
    logical_payload_bytes: int
    serialized_payload_bytes: int
    framing_bytes: int

    @property
    def total_application_bytes(self) -> int:
        return self.serialized_payload_bytes + self.framing_bytes

    def cost_record(
        self,
        *,
        recipient_tokenizer: str = "not_measured",
        recipient_payload_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return a tlu.costs.v3 transmission with an exact byte partition.

        Tokenizer counts are explicitly null unless the caller supplies a
        recipient-tokenizer measurement of the delivered payload alone.
        """
        if not isinstance(recipient_tokenizer, str) or not recipient_tokenizer.strip():
            raise ValueError("recipient_tokenizer must be a non-empty string")
        if recipient_payload_tokens is not None and (
            isinstance(recipient_payload_tokens, bool)
            or not isinstance(recipient_payload_tokens, int)
            or recipient_payload_tokens < 0
        ):
            raise ValueError("recipient_payload_tokens must be a non-negative integer or None")
        return {
            "round": self.round_number,
            "sender": self.sender,
            "recipients": [self.recipient],
            "payload_bytes": self.serialized_payload_bytes,
            "framing_bytes": self.framing_bytes,
            "encoding": "utf-8",
            "media_type": "application/json",
            "transport_boundary": "network",
            "payload_metadata": {
                "message_schema": MESSAGE_SCHEMA,
                "protocol_id": self.protocol_id,
                "logical_text_utf8_bytes": self.logical_payload_bytes,
                "application_layer_scope": "length-prefixed loopback TCP; TCP/IP headers excluded",
            },
            "recipient_tokens": {
                self.recipient: {
                    "tokenizer": recipient_tokenizer,
                    "tokens": recipient_payload_tokens,
                }
            },
        }


@dataclass(frozen=True)
class FrameTransmission:
    """One delivered opaque payload and its exact application-layer costs."""

    payload: bytes
    protocol_id: str
    round_number: int
    sender: str
    recipient: str
    media_type: str
    encoding: str
    framing_bytes: int
    payload_metadata: dict[str, Any]

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)

    @property
    def total_application_bytes(self) -> int:
        return self.payload_bytes + self.framing_bytes

    def cost_record(
        self,
        *,
        recipient_tokenizer: str = "not_measured",
        recipient_payload_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(recipient_tokenizer, str) or not recipient_tokenizer.strip():
            raise ValueError("recipient_tokenizer must be a non-empty string")
        if recipient_payload_tokens is not None and (
            isinstance(recipient_payload_tokens, bool)
            or not isinstance(recipient_payload_tokens, int)
            or recipient_payload_tokens < 0
        ):
            raise ValueError("recipient_payload_tokens must be a non-negative integer or None")
        record = {
            "round": self.round_number,
            "sender": self.sender,
            "recipients": [self.recipient],
            "payload_bytes": self.payload_bytes,
            "framing_bytes": self.framing_bytes,
            "encoding": self.encoding,
            "media_type": self.media_type,
            "transport_boundary": "network",
            "payload_metadata": {
                **self.payload_metadata,
                "frame_schema": FRAME_SCHEMA,
                "protocol_id": self.protocol_id,
                "application_layer_scope": "length-prefixed loopback TCP; TCP/IP headers excluded",
            },
            "recipient_tokens": {
                self.recipient: {
                    "tokenizer": recipient_tokenizer,
                    "tokens": recipient_payload_tokens,
                }
            },
        }
        return record


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class LocalTCPMessageChannel:
    """Deliver JSON-framed messages through an ephemeral loopback TCP socket.

    The receiver callback gets a decoded envelope. This channel is intended
    for local experiments and tests, not for remote or production deployment.
    """

    def __init__(
        self,
        on_message: Callable[[dict[str, Any]], None],
        *,
        max_envelope_bytes: int = MAX_ENVELOPE_BYTES,
        timeout_seconds: float = 120.0,
    ) -> None:
        if max_envelope_bytes < 1:
            raise ValueError("max_envelope_bytes must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._on_message = on_message
        self._max_envelope_bytes = max_envelope_bytes
        self._timeout_seconds = timeout_seconds
        self._server: _ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("channel is not running")
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def __enter__(self) -> "LocalTCPMessageChannel":
        if self._server is not None:
            raise RuntimeError("channel is already running")
        channel = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                self.request.settimeout(channel._timeout_seconds)
                header = _recv_exact(self.request, 4)
                (size,) = struct.unpack("!I", header)
                if size < 1 or size > channel._max_envelope_bytes:
                    raise ValueError("message envelope size is outside the configured limit")
                body = _recv_exact(self.request, size)
                envelope = _decode_envelope(body)
                try:
                    channel._on_message(envelope)
                except Exception:
                    self.request.sendall(b"\x00")
                    return
                self.request.sendall(b"\x01")

        self._server = _ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self._server.timeout = 0.25
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="tacit-loopback-message-channel",
            daemon=True,
        )
        self._thread.start()
        return self

    def measure(
        self,
        message: str,
        *,
        protocol_id: str,
        round_number: int,
        sender: str,
        recipient: str,
    ) -> Transmission:
        """Measure exact application bytes without delivering the message."""
        _validate_message_fields(message, protocol_id, round_number, sender, recipient)
        body, transmission = _prepare_message(
            message, protocol_id, round_number, sender, recipient
        )
        if len(body) > self._max_envelope_bytes:
            raise ValueError("message envelope exceeds max_envelope_bytes")
        return transmission

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_seconds)
        self._server = None
        self._thread = None

    def send(
        self,
        message: str,
        *,
        protocol_id: str,
        round_number: int,
        sender: str,
        recipient: str,
    ) -> Transmission:
        """Serialize, transmit, and await callback completion for one message."""
        if self._server is None:
            raise RuntimeError("channel is not running")
        _validate_message_fields(message, protocol_id, round_number, sender, recipient)
        body, transmission = _prepare_message(
            message, protocol_id, round_number, sender, recipient
        )
        if len(body) > self._max_envelope_bytes:
            raise ValueError("message envelope exceeds max_envelope_bytes")
        prefix = struct.pack("!I", len(body))
        host, port = self.address
        with socket.create_connection((host, port), timeout=self._timeout_seconds) as client:
            client.settimeout(self._timeout_seconds)
            client.sendall(prefix + body)
            client.shutdown(socket.SHUT_WR)
            # The one-byte acknowledgment is included in framing cost. It
            # confirms callback completion without exposing callback results.
            status = client.recv(1)
            if status != b"\x01":
                raise RuntimeError("receiver callback failed or did not acknowledge delivery")

        return transmission


class LocalTCPFrameChannel:
    """Send opaque bytes with a JSON metadata header and raw binary body.

    Wire format: 4-byte big-endian metadata length, UTF-8 JSON metadata,
    exactly ``payload_length`` raw bytes, then a one-byte callback ACK. The
    metadata length and payload are bounded before allocation. This transports
    representations; it does not make arbitrary bytes consumable by a text LLM.
    """

    def __init__(
        self,
        on_frame: Callable[[dict[str, Any], bytes], None],
        *,
        max_frame_bytes: int = MAX_FRAME_BYTES,
        timeout_seconds: float = 120.0,
    ) -> None:
        if max_frame_bytes < 1:
            raise ValueError("max_frame_bytes must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._on_frame = on_frame
        self._max_frame_bytes = max_frame_bytes
        self._timeout_seconds = timeout_seconds
        self._server: _ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("channel is not running")
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def __enter__(self) -> "LocalTCPFrameChannel":
        if self._server is not None:
            raise RuntimeError("channel is already running")
        channel = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                self.request.settimeout(channel._timeout_seconds)
                (metadata_size,) = struct.unpack("!I", _recv_exact(self.request, 4))
                if metadata_size < 1 or metadata_size > channel._max_frame_bytes:
                    raise ValueError("frame metadata size is outside the configured limit")
                metadata = _decode_frame_metadata(_recv_exact(self.request, metadata_size))
                payload_size = metadata["payload_length"]
                if metadata_size + payload_size > channel._max_frame_bytes:
                    raise ValueError("frame size is outside the configured limit")
                payload = _recv_exact(self.request, payload_size)
                try:
                    channel._on_frame(metadata, payload)
                except Exception:
                    self.request.sendall(b"\x00")
                    return
                self.request.sendall(b"\x01")

        self._server = _ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="tacit-loopback-frame-channel",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_seconds)
        self._server = None
        self._thread = None

    def send(
        self,
        payload: bytes,
        *,
        protocol_id: str,
        round_number: int,
        sender: str,
        recipient: str,
        media_type: str = "application/octet-stream",
        encoding: str = "identity",
        payload_metadata: dict[str, Any] | None = None,
    ) -> FrameTransmission:
        if self._server is None:
            raise RuntimeError("channel is not running")
        _validate_frame_fields(protocol_id, round_number, sender, recipient, media_type, encoding)
        if not isinstance(payload, bytes):
            raise ValueError("payload must be bytes")
        if payload_metadata is not None and not isinstance(payload_metadata, dict):
            raise ValueError("payload_metadata must be an object")
        details = dict(payload_metadata or {})
        reserved = {"schema", "protocol_id", "round", "sender", "recipient", "payload_length"}
        if reserved.intersection(details):
            raise ValueError("payload_metadata contains reserved frame fields")
        metadata = {
            "schema": FRAME_SCHEMA,
            "protocol_id": protocol_id,
            "round": round_number,
            "sender": sender,
            "recipient": recipient,
            "media_type": media_type,
            "encoding": encoding,
            "payload_length": len(payload),
            "payload_metadata": details,
        }
        try:
            header = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("payload_metadata must contain JSON values") from exc
        if len(header) + len(payload) > self._max_frame_bytes:
            raise ValueError("frame exceeds max_frame_bytes")
        prefix = struct.pack("!I", len(header))
        host, port = self.address
        with socket.create_connection((host, port), timeout=self._timeout_seconds) as client:
            client.settimeout(self._timeout_seconds)
            client.sendall(prefix + header + payload)
            client.shutdown(socket.SHUT_WR)
            if client.recv(1) != b"\x01":
                raise RuntimeError("receiver callback failed or did not acknowledge delivery")
        return FrameTransmission(
            payload=payload,
            protocol_id=protocol_id,
            round_number=round_number,
            sender=sender,
            recipient=recipient,
            media_type=media_type,
            encoding=encoding,
            framing_bytes=len(prefix) + len(header) + 1,
            payload_metadata=details,
        )


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("connection closed before a complete message frame arrived")
        chunks.extend(chunk)
    return bytes(chunks)


def _prepare_message(
    message: str,
    protocol_id: str,
    round_number: int,
    sender: str,
    recipient: str,
) -> tuple[bytes, Transmission]:
    """Serialize one text envelope and derive its exact application-byte cost."""
    envelope = {
        "schema": MESSAGE_SCHEMA,
        "protocol_id": protocol_id,
        "round": round_number,
        "sender": sender,
        "recipient": recipient,
        "payload": message,
    }
    body = json.dumps(
        envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload_field = json.dumps(
        message, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    serialized_payload_bytes = len(payload_field)
    framing_bytes = 4 + len(body) + 1 - serialized_payload_bytes
    if framing_bytes < 0:
        raise RuntimeError("serialized envelope byte partition is invalid")
    transmission = Transmission(
        message=message,
        protocol_id=protocol_id,
        round_number=round_number,
        sender=sender,
        recipient=recipient,
        logical_payload_bytes=len(message.encode("utf-8")),
        serialized_payload_bytes=serialized_payload_bytes,
        framing_bytes=framing_bytes,
    )
    return body, transmission


def _decode_envelope(body: bytes) -> dict[str, Any]:
    try:
        envelope = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("message envelope must be valid UTF-8 JSON") from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != MESSAGE_SCHEMA:
        raise ValueError("unsupported message envelope")
    _validate_message_fields(
        envelope.get("payload"),
        envelope.get("protocol_id"),
        envelope.get("round"),
        envelope.get("sender"),
        envelope.get("recipient"),
    )
    return envelope


def _validate_message_fields(
    message: Any,
    protocol_id: Any,
    round_number: Any,
    sender: Any,
    recipient: Any,
) -> None:
    if not isinstance(message, str):
        raise ValueError("message must be text")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("protocol_id must be a non-empty string")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise ValueError("round_number must be a positive integer")
    if not isinstance(sender, str) or not sender.strip():
        raise ValueError("sender must be a non-empty string")
    if not isinstance(recipient, str) or not recipient.strip():
        raise ValueError("recipient must be a non-empty string")


def _decode_frame_metadata(body: bytes) -> dict[str, Any]:
    try:
        metadata = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frame metadata must be valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != FRAME_SCHEMA:
        raise ValueError("unsupported binary frame")
    _validate_frame_fields(
        metadata.get("protocol_id"), metadata.get("round"), metadata.get("sender"),
        metadata.get("recipient"), metadata.get("media_type"), metadata.get("encoding"),
    )
    size = metadata.get("payload_length")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("payload_length must be a non-negative integer")
    if not isinstance(metadata.get("payload_metadata"), dict):
        raise ValueError("payload_metadata must be an object")
    return metadata


def _validate_frame_fields(
    protocol_id: Any, round_number: Any, sender: Any, recipient: Any,
    media_type: Any, encoding: Any,
) -> None:
    _validate_message_fields("", protocol_id, round_number, sender, recipient)
    if not isinstance(media_type, str) or not media_type.strip():
        raise ValueError("media_type must be a non-empty string")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("encoding must be a non-empty string")
