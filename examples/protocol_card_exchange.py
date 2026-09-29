"""Run a shared JSON ProtocolCard against two OpenAI-compatible endpoints.

This example uses an ordinary natural-language card as an SDK smoke path. It
does not claim that this card is a novel or more efficient language.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from tacit import OpenAICompatibleClient, ProtocolCard, exchange_once


def main() -> None:
    card_path = Path(os.environ.get(
        "TLU_PROTOCOL_CARD", str(Path(__file__).with_name("protocol_card.json"))
    ))
    card_bytes = card_path.read_bytes()
    card = ProtocolCard.from_json(card_bytes)
    result = exchange_once(
        OpenAICompatibleClient(
            base_url=os.environ.get("TLU_SENDER_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get("TLU_SENDER_MODEL", "local-model-a"),
            api_key=os.environ.get("TLU_SENDER_API_KEY"),
            timeout_seconds=30,
            max_tokens=160,
            follow_redirects=False,
        ),
        OpenAICompatibleClient(
            base_url=os.environ.get("TLU_RECEIVER_URL", "http://127.0.0.1:8001/v1"),
            model=os.environ.get("TLU_RECEIVER_MODEL", "local-model-b"),
            api_key=os.environ.get("TLU_RECEIVER_API_KEY"),
            timeout_seconds=30,
            max_tokens=160,
            follow_redirects=False,
        ),
        protocol=card,
        sender_context="Observed A before B in three trials; the sample is small.",
        receiver_context="You observed B before A in two trials.",
        receiver_task="Give a joint conclusion about the likely ordering and confidence.",
    )
    print(json.dumps({
        "protocol_id": result.protocol_id,
        "protocol_card_sha256": hashlib.sha256(card_bytes).hexdigest(),
        "message": result.message,
        "payload_bytes": result.payload_bytes,
        "sender_model": result.sender.model,
        "sender_input_tokens": result.sender.input_tokens,
        "sender_output_tokens": result.sender.output_tokens,
        "receiver_model": result.receiver.model,
        "receiver_input_tokens": result.receiver.input_tokens,
        "receiver_output_tokens": result.receiver.output_tokens,
        "receiver_response": result.receiver.text,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
