"""Run a three-turn, two-agent local chat exchange.

This example uses a fixed schedule; it is not a live resource monitor and does
not start a server. Configure two loopback OpenAI-compatible endpoints first.
"""
from __future__ import annotations

import json
import os

from tacit import OpenAICompatibleClient, exchange_dialogue


class EvidenceDialogue:
    protocol_id = "evidence-dialogue-fixed-three-turn-v1"
    agent_instructions = {
        "A": (
            "Share evidence you observed, label uncertainty, and use the messages visible to you. "
            "Do not claim to know the other agent's private context."
        ),
        "B": (
            "Share evidence you observed, label uncertainty, and use the messages visible to you. "
            "Do not claim to know the other agent's private context."
        ),
    }


def main() -> None:
    agent_a = OpenAICompatibleClient(
        os.environ.get("TLU_AGENT_A_URL", "http://127.0.0.1:8000/v1"),
        os.environ.get("TLU_AGENT_A_MODEL", "local-model-a"),
        api_key=os.environ.get("TLU_AGENT_A_API_KEY"),
        timeout_seconds=30,
        max_tokens=128,
        follow_redirects=False,
    )
    agent_b = OpenAICompatibleClient(
        os.environ.get("TLU_AGENT_B_URL", "http://127.0.0.1:8001/v1"),
        os.environ.get("TLU_AGENT_B_MODEL", "local-model-b"),
        api_key=os.environ.get("TLU_AGENT_B_API_KEY"),
        timeout_seconds=30,
        max_tokens=128,
        follow_redirects=False,
    )
    result = exchange_dialogue(
        {"A": agent_a, "B": agent_b},
        protocol=EvidenceDialogue(),
        private_contexts={
            "A": "Observation A: the sensor activated twice, but both readings may be noisy.",
            "B": "Observation B: the log records one activation after the maintenance event.",
        },
        schedule=("A", "B", "A"),
        task="Assess whether the maintenance event likely changed the sensor behavior.",
        max_turns=3,
        wire_budget_bytes=4096,
    )
    print(json.dumps({
        "protocol_id": result.protocol_id,
        "model_calls": result.model_calls,
        "wire_bytes": result.wire_bytes,
        "wire_budget_bytes": result.wire_budget_bytes,
        "stop_reason": result.stop_reason,
        "model_call_records": result.model_call_records(),
        "transmission_records": result.transmission_records(),
        "messages": [
            {"speaker": turn.speaker, "message": turn.completion.text,
             "delivered": turn.transmission is not None}
            for turn in result.turns
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
