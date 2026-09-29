"""Exact model-free communication frontier for Private Match v0.2.

The bound assumes a uniformly sampled k-record receiver table, a uniform
target within that table, a shared encoder codebook, and one noiseless message
from a sender that does not see the receiver table. Codebook setup and physical
wire framing are deliberately outside this payload-alphabet reference.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import comb


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _choose(n: int, k: int) -> int:
    """Binomial coefficient with the standard zero convention for n < k."""
    return comb(n, k) if n >= k else 0


def optimal_success_probability(
    *, space_size: int, candidate_count: int, message_count: int
) -> Fraction:
    """Return the exact optimal expected match accuracy with <= B messages.

    The optimal deterministic encoder partitions the N possible records into
    balanced message classes. On receiving a message and the candidate table,
    the Bayes decoder chooses among candidates in that class. Randomized
    encoders/decoders cannot improve this finite Bayesian optimum.
    """
    _integer("space_size", space_size, 2)
    _integer("candidate_count", candidate_count, 2)
    _integer("message_count", message_count, 1)
    if candidate_count > space_size:
        raise ValueError("candidate_count cannot exceed space_size")

    bins = min(message_count, space_size)
    small, larger_bin_count = divmod(space_size, bins)
    denominator = _choose(space_size, candidate_count)
    collision_table_sum = (
        (bins - larger_bin_count) * _choose(space_size - small, candidate_count)
        + larger_bin_count * _choose(space_size - small - 1, candidate_count)
    )
    return Fraction(
        bins * denominator - collision_table_sum,
        candidate_count * denominator,
    )


def bit_budget_frontier(*, space_size: int, candidate_count: int, max_bits: int) -> list[dict[str, int | str | float]]:
    """Evaluate the optimum for every fixed-width payload budget 0..max_bits."""
    _integer("max_bits", max_bits, 0)
    output = []
    for bits in range(max_bits + 1):
        messages = min(1 << bits, space_size)
        exact = optimal_success_probability(
            space_size=space_size,
            candidate_count=candidate_count,
            message_count=messages,
        )
        output.append({
            "payload_bits": bits,
            "available_messages": messages,
            "success_fraction": f"{exact.numerator}/{exact.denominator}",
            "success_probability": float(exact),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument("--vocabulary-size", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-bits", type=int, default=12)
    args = parser.parse_args()
    if args.features < 1 or args.vocabulary_size < 2:
        parser.error("features must be >=1 and vocabulary-size must be >=2")
    space_size = args.vocabulary_size ** args.features
    try:
        result = bit_budget_frontier(
            space_size=space_size,
            candidate_count=args.candidates,
            max_bits=args.max_bits,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "space_size": space_size,
        "candidate_count": args.candidates,
        "shared_codebook_setup_cost_included": False,
        "wire_framing_cost_included": False,
        "frontier": result,
    }, indent=2))


if __name__ == "__main__":
    main()
