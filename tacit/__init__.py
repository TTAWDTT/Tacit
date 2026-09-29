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
from .protocol import (
    DIALOGUE_PROTOCOL_CARD_SCHEMA,
    PROTOCOL_CARD_SCHEMA,
    DialogueProtocolCard,
    ProtocolCard,
)

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
    "DIALOGUE_PROTOCOL_CARD_SCHEMA",
    "DialogueProtocolCard",
]
