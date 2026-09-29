"""Small, protocol-neutral runtime for LLM-to-LLM message experiments."""

from .runtime import (
    ChatCompletion,
    ChatModel,
    ExchangeResult,
    OpenAICompatibleClient,
    TextProtocol,
    exchange_once,
)
from .channel import FrameTransmission, LocalTCPFrameChannel, LocalTCPMessageChannel, Transmission

__all__ = [
    "ChatCompletion",
    "ChatModel",
    "ExchangeResult",
    "OpenAICompatibleClient",
    "TextProtocol",
    "exchange_once",
    "LocalTCPMessageChannel",
    "LocalTCPFrameChannel",
    "Transmission",
    "FrameTransmission",
]
