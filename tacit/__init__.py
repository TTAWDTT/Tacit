"""Small, protocol-neutral runtime for LLM-to-LLM message experiments."""

from .runtime import (
    ChatCompletion,
    ChatModel,
    ExchangeResult,
    OpenAICompatibleClient,
    TextProtocol,
    exchange_once,
)

__all__ = [
    "ChatCompletion",
    "ChatModel",
    "ExchangeResult",
    "OpenAICompatibleClient",
    "TextProtocol",
    "exchange_once",
]
