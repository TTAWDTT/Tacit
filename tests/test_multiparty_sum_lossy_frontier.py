from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pytest

from research.multiparty_sum_lossy_frontier import exact_frontier, frontier_report
from research.multiparty_sum_scaling import scaling_row


def _check_codebook(frontier_row: dict, agents: int) -> int:
    partitions = frontier_row["optimal_sender_partitions"]
    decoder = {
        tuple(row["message_ids"]): row["decoded_sum"]
        for row in frontier_row["optimal_receiver_decoder"]
    }
    correct = 0
    for inputs in product(range(4), repeat=agents):
        message_ids = tuple(
            next(index for index, block in enumerate(partition) if value in block)
            for value, partition in zip(inputs, partitions)
        )
        correct += decoder[message_ids] == sum(inputs)
    return correct


def test_exact_zero_and_full_payload_endpoints_match_closed_form() -> None:
    for agents in range(1, 5):
        rows = exact_frontier(agents)
        assert rows[0]["optimal_success_numerator"] == scaling_row(agents, 4)["no_message_exact_success_numerator"]
        assert rows[-1]["optimal_exact_sum_success"] == 1.0
        assert rows[-1]["protocol_minimum_bits"] == 2 * agents


def test_known_small_frontiers_and_emitted_codebooks_are_exact() -> None:
    expected_m2 = [0.25, 0.25, 0.5, 0.5, 1.0]
    rows = exact_frontier(2)
    assert [row["optimal_exact_sum_success"] for row in rows] == expected_m2
    for row in rows:
        assert row["optimal_success_numerator"] == _check_codebook(row, 2)


def test_frontier_is_monotonic_and_search_limit_is_explicit() -> None:
    for agents in range(1, 5):
        rates = [row["optimal_exact_sum_success"] for row in exact_frontier(agents)]
        assert rates == sorted(rates)
    with pytest.raises(ValueError, match="from 1 through 4"):
        exact_frontier(5)


def test_committed_json_artifact_matches_the_calculator() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = json.loads((root / "research/data/MULTIPARTY_SUM_LOSSY_FRONTIER_V0_1.json").read_text(encoding="utf-8"))
    assert artifact == frontier_report((2, 3, 4))
