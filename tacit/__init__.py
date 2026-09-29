"""Small, protocol-neutral runtime for LLM-to-LLM message experiments."""

from .runtime import (
    ChatCompletion,
    ChatModel,
    DialogueProtocol,
    DialogueResult,
    DialogueTurn,
    ExchangeResult,
    OpenAICompatibleClient,
    TextProtocol,
    exchange_dialogue,
    exchange_once,
)
from .channel import FrameTransmission, LocalTCPFrameChannel, LocalTCPMessageChannel, Transmission
from .protocol import PROTOCOL_CARD_SCHEMA, ProtocolCard

__all__ = [
    "ChatCompletion",
    "ChatModel",
    "DialogueProtocol",
    "DialogueResult",
    "DialogueTurn",
    "ExchangeResult",
    "OpenAICompatibleClient",
    "TextProtocol",
    "exchange_dialogue",
    "exchange_once",
    "LocalTCPMessageChannel",
    "LocalTCPFrameChannel",
    "Transmission",
    "FrameTransmission",
    "PROTOCOL_CARD_SCHEMA",
    "ProtocolCard",
]
