from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from research.multiparty_sum_transport_frontier import measure_frontier


def test_exact_frontier_codebooks_map_to_transport_costs_without_sockets() -> None:
    with patch("socket.create_connection", side_effect=AssertionError("measure must not open sockets")):
        report = measure_frontier((2, 3, 4))
    for agent_count_text, budget_rows in report["frontiers"].items():
        agent_count = int(agent_count_text)
        assert len(budget_rows) == 2 * agent_count + 1
        for row in budget_rows:
            assert row["success_verified_by_text_codec"] == row["success_numerator"]
            assert row["success_verified_by_frame_codec"] == row["success_numerator"]
            for schedule in row["transport"].values():
                for variant in schedule.values():
                    assert variant["model_calls"] == 0
                    assert variant["sender_messages_including_zero_width_empty_codewords"] == agent_count
                    assert variant["protocol_id_utf8_bytes"] == 16
                    assert variant["frame_application_bytes"] == (
                        variant["frame_payload_bytes"] + variant["frame_framing_bytes"]
                    )
                    assert variant["text_application_bytes"] == (
                        variant["text_payload_serialized_bytes"] + variant["text_framing_bytes"]
                    )


def test_zero_bit_oracle_still_pays_transport_framing() -> None:
    row = measure_frontier((2,))["frontiers"]["2"][0]
    assert row["actual_fixed_width_payload_bits"] == 0
    for variant in row["transport"]["empty_message_for_every_sender"].values():
        assert variant["text_payload_utf8_bytes"] == 0
        assert variant["frame_payload_bytes"] == 0
        assert variant["text_framing_bytes"] > 0
        assert variant["frame_framing_bytes"] > 0
    for variant in row["transport"]["omit_zero_width_senders"].values():
        assert variant["sender_messages_transmitted"] == 0
        assert variant["text_application_bytes"] == 0
        assert variant["frame_application_bytes"] == 0


def test_committed_transport_ledger_reproduces_exactly() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = json.loads((root / "research/data/MULTIPARTY_SUM_TRANSPORT_FRONTIER_V0_1.json").read_text(encoding="utf-8"))
    assert artifact == measure_frontier((2, 3, 4))
