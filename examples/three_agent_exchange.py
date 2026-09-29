"""Run a fixed-schedule unicast exchange among three configured endpoints.

This example does not start servers or load models. Configure three
OpenAI-compatible endpoints and apply the project's resource gates first.
"""
from __future__ import annotations

import json
import os

from tacit import OpenAICompatibleClient, exchange_dialogue

DEFAULT_PORTS = {"A": 8000, "B": 8001, "C": 8002}


class EvidenceProtocol:
    protocol_id = "evidence-unicast-three-agent-v1"
    agent_instructions = {
        name: (
            "Share only evidence from your private context or messages routed to you. "
            "Label uncertainty and do not claim to know unreceived messages."
        )
        for name in ("A", "B", "C")
    }


def _client(name: str) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=os.environ.get(
            f"TLU_AGENT_{name}_URL", f"http://127.0.0.1:{DEFAULT_PORTS[name]}/v1"
        ),
        model=os.environ.get(f"TLU_AGENT_{name}_MODEL", f"local-model-{name.lower()}"),
        api_key=os.environ.get(f"TLU_AGENT_{name}_API_KEY"),
        timeout_seconds=30,
        max_tokens=128,
        follow_redirects=False,
    )


def main() -> None:
    result = exchange_dialogue(
        {name: _client(name) for name in ("A", "B", "C")},
        protocol=EvidenceProtocol(),
        private_contexts={
            "A": "Trial log: event X preceded event Y in 3 of 4 observations.",
            "B": "Independent sensor: event Y preceded event X once; sensor is noisy.",
            "C": "Maintenance record: sensor calibration changed between trials 2 and 3.",
        },
        schedule=(("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("C", "B")),
        task="Assess the likely event ordering and how the calibration change affects confidence.",
        max_turns=5,
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
            {
                "sender": turn.speaker,
                "recipient": turn.recipient,
                "message": turn.completion.text,
                "delivered": turn.transmission is not None,
            }
            for turn in result.turns
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
