"""Exact model-free scaling reference for simultaneous private-input summation.

The result is a coding-complexity oracle, not an LLM performance predictor. It
assumes independent uniform inputs, one simultaneous message per sender, a
fixed shared protocol, a noiseless binary channel, and exact integer-sum output.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable


def _validate_domain_size(domain_size: int) -> int:
    if isinstance(domain_size, bool) or not isinstance(domain_size, int):
        raise ValueError("domain_size must be an integer")
    if domain_size < 2 or domain_size & (domain_size - 1):
        raise ValueError("domain_size must be a power of two >= 2")
    return domain_size.bit_length() - 1


def no_message_exact_success(agent_count: int, domain_size: int) -> tuple[int, int]:
    """Return (numerator, denominator) for optimal exact-sum guessing.

    Each private input is uniform over range(domain_size). With no messages,
    the receiver's best fixed answer is a mode of the sum distribution.
    """
    _validate_domain_size(domain_size)
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or agent_count < 1:
        raise ValueError("agent_count must be a positive integer")
    counts = [1]
    for _ in range(agent_count):
        next_counts = [0] * (len(counts) + domain_size - 1)
        for partial_sum, count in enumerate(counts):
            for value in range(domain_size):
                next_counts[partial_sum + value] += count
        counts = next_counts
    return max(counts), domain_size**agent_count


def scaling_row(agent_count: int, domain_size: int) -> dict[str, int | float]:
    """Compute exact binary communication floor and no-message Bayes score."""
    input_width = _validate_domain_size(domain_size)
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or agent_count < 1:
        raise ValueError("agent_count must be a positive integer")
    numerator, denominator = no_message_exact_success(agent_count, domain_size)
    total_bits = agent_count * input_width
    return {
        "agents": agent_count,
        "domain_size": domain_size,
        "bits_per_private_input": input_width,
        "exact_zero_error_payload_lower_bound_bits": total_bits,
        "attainable_fixed_width_payload_bits": total_bits,
        "simultaneous_messages": agent_count,
        "llm_calls_if_each_sender_and_referee_is_called_once": agent_count + 1,
        "no_message_exact_success_numerator": numerator,
        "no_message_exact_success_denominator": denominator,
        "no_message_exact_success": numerator / denominator,
    }


def scaling_table(agent_counts: Iterable[int], domain_size: int) -> list[dict[str, int | float]]:
    """Compute a deterministic table; order and duplicate counts are retained."""
    return [scaling_row(agent_count, domain_size) for agent_count in agent_counts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-size", type=int, default=4)
    parser.add_argument("--agents", type=int, nargs="+", default=[2, 3, 4, 8, 16])
    args = parser.parse_args()
    print(json.dumps(scaling_table(args.agents, args.domain_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
