"""Exhaustive exact-success frontier for fixed-width private-sum protocols.

The oracle ranges over every deterministic simultaneous encoder partition of
the four-value source alphabet for each sender. It reports ideal payload bits;
it excludes framing, prompts, model compute, and stochastic decoding errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import Any


DOMAIN = (0, 1, 2, 3)
VERSION = "0.1.0"


def _partitions(items: tuple[int, ...] = DOMAIN) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return canonical set partitions of ``items``."""
    result: list[tuple[tuple[int, ...], ...]] = [()]
    for item in items:
        expanded: list[tuple[tuple[int, ...], ...]] = []
        for partition in result:
            expanded.append((*partition, (item,)))
            for block_index in range(len(partition)):
                blocks = list(partition)
                blocks[block_index] = (*blocks[block_index], item)
                expanded.append(tuple(blocks))
        result = expanded
    return tuple(sorted(result))


PARTITIONS = _partitions()


def _partition_bits(partition: tuple[tuple[int, ...], ...]) -> int:
    return math.ceil(math.log2(len(partition))) if len(partition) > 1 else 0


def _optimal_decoder(
    partitions: tuple[tuple[tuple[int, ...], ...], ...],
    *, include_decoder: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    """Return exact correct-vector count and a deterministic MAP decoder table."""
    successful_vectors = 0
    decoder = []
    for message_ids in product(*(range(len(partition)) for partition in partitions)):
        observed_blocks = [partitions[i][message_id] for i, message_id in enumerate(message_ids)]
        coefficients = [1]
        for block in observed_blocks:
            updated = [0] * (len(coefficients) + max(block))
            for partial_sum, count in enumerate(coefficients):
                if count:
                    for value in block:
                        updated[partial_sum + value] += count
            coefficients = updated
        maximum = max(coefficients)
        decoded_sum = coefficients.index(maximum)
        successful_vectors += maximum
        if include_decoder:
            decoder.append({"message_ids": list(message_ids), "decoded_sum": decoded_sum})
    return successful_vectors, decoder


@lru_cache(maxsize=None)
def exact_frontier(agent_count: int) -> list[dict[str, Any]]:
    """Enumerate the exact deterministic Bayes-success frontier for 1..4 senders.

    Each sender has one fixed-width message, and per-sender widths may differ.
    The budget is the sum of payload bits. All mappings from the four inputs to
    at most four message symbols are considered, modulo irrelevant label swaps.
    """
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or not 1 <= agent_count <= 4:
        raise ValueError("agent_count must be an integer from 1 through 4 (exhaustive search limit)")
    best_at_cost: dict[int, tuple[int, tuple[tuple[tuple[int, ...], ...], ...]]] = {}
    for protocol in combinations_with_replacement(PARTITIONS, agent_count):
        cost = sum(_partition_bits(partition) for partition in protocol)
        success_count, _decoder = _optimal_decoder(protocol)
        previous = best_at_cost.get(cost)
        if previous is None or success_count > previous[0]:
            best_at_cost[cost] = (success_count, protocol)

    total_vectors = len(DOMAIN) ** agent_count
    frontier: list[dict[str, Any]] = []
    for budget in range(2 * agent_count + 1):
        feasible = [(success_count, cost, protocol)
                    for cost, (success_count, protocol) in best_at_cost.items() if cost <= budget]
        incumbent_count = max(success_count for success_count, _cost, _protocol in feasible)
        _count, minimum_cost, incumbent_protocol = min(
            (success_count, cost, protocol)
            for success_count, cost, protocol in feasible
            if success_count == incumbent_count
        )
        decoder_count, decoder = _optimal_decoder(incumbent_protocol, include_decoder=True)
        assert decoder_count == incumbent_count
        frontier.append({
            "payload_budget_bits": budget,
            "optimal_success_numerator": incumbent_count,
            "input_vector_count": total_vectors,
            "optimal_exact_sum_success": incumbent_count / total_vectors,
            "optimal_sender_partition_sizes": [len(partition) for partition in incumbent_protocol],
            "optimal_sender_partitions": [[list(block) for block in partition] for partition in incumbent_protocol],
            "optimal_receiver_decoder": decoder,
            "protocol_minimum_bits": minimum_cost,
        })
    return frontier


def frontier_report(agent_counts: tuple[int, ...]) -> dict[str, Any]:
    """Return the versioned, JSON-serializable result artifact."""
    return {
        "schema_version": f"tlu.multiparty-sum-lossy-frontier.v{VERSION}",
        "source_domain": list(DOMAIN),
        "source_prior": "independent_uniform_per_sender",
        "calculator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_class": "deterministic simultaneous one-message encoders; each sender uses a fixed-width binary codeword; arbitrary sender-specific partitions and exact-sum MAP decoders",
        "payload_only": True,
        "enumeration": "all set partitions of the source alphabet, up to sender permutation; message-label permutations are equivalent",
        "frontiers": {str(agent_count): exact_frontier(agent_count) for agent_count in agent_counts},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()
    try:
        report = frontier_report(tuple(args.agents))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
