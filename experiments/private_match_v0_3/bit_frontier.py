"""Exact ideal bit-budget frontier for triadic Private Match v0.3.

This is a model-free reference for two simultaneous, independent, fixed-width
messages over a complete q-by-q candidate table. It excludes message framing,
prompts, setup, inference, and model errors.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json


def _validate_q(q: int) -> None:
    if isinstance(q, bool) or not isinstance(q, int) or q < 2 or q & (q - 1):
        raise ValueError("q must be an integer power of two >= 2")


def optimal_success_probability(*, q: int, total_payload_bits: int) -> Fraction:
    """Maximum Bayes exact-match accuracy under a total fixed-width bit cap.

    The fixed schedule identifies which source sent each payload. Each source
    gets an integer number of payload bits, and unused budget is allowed.
    """
    _validate_q(q)
    if (isinstance(total_payload_bits, bool) or not isinstance(total_payload_bits, int)
            or total_payload_bits < 0):
        raise ValueError("total_payload_bits must be a non-negative integer")
    useful_bits = min(total_payload_bits, 2 * (q.bit_length() - 1))
    return Fraction(1 << useful_bits, q * q)


def optimal_allocation(*, q: int, total_payload_bits: int) -> dict[str, int]:
    """Return a balanced optimal allocation, capped at log2(q) per source."""
    _validate_q(q)
    if (isinstance(total_payload_bits, bool) or not isinstance(total_payload_bits, int)
            or total_payload_bits < 0):
        raise ValueError("total_payload_bits must be a non-negative integer")
    width = q.bit_length() - 1
    useful_bits = min(total_payload_bits, 2 * width)
    x_bits = min(width, (useful_bits + 1) // 2)
    y_bits = useful_bits - x_bits
    return {"sender_x_bits": x_bits, "sender_y_bits": y_bits}


def frontier(*, q: int, max_total_payload_bits: int) -> list[dict[str, int | str | float]]:
    _validate_q(q)
    if (isinstance(max_total_payload_bits, bool)
            or not isinstance(max_total_payload_bits, int)
            or max_total_payload_bits < 0):
        raise ValueError("max_total_payload_bits must be a non-negative integer")
    rows = []
    for budget in range(max_total_payload_bits + 1):
        exact = optimal_success_probability(q=q, total_payload_bits=budget)
        allocation = optimal_allocation(q=q, total_payload_bits=budget)
        rows.append({
            "total_payload_bit_budget": budget,
            **allocation,
            "payload_bits_used": sum(allocation.values()),
            "success_fraction": f"{exact.numerator}/{exact.denominator}",
            "success_probability": float(exact),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--max-bits", type=int, default=8)
    args = parser.parse_args()
    try:
        rows = frontier(q=args.q, max_total_payload_bits=args.max_bits)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "task": "complete_cartesian_q_by_q_uniform_hidden_target",
        "q": args.q,
        "candidate_count": args.q * args.q,
        "assumptions": [
            "two simultaneous messages, one from each coordinate source",
            "fixed schedule reveals the sender slot",
            "noiseless fixed-width binary payloads",
            "receiver observes the complete candidate table",
            "uniform target over all q squared rows",
        ],
        "excluded_costs": ["framing", "prompts", "codebook setup", "inference", "model errors"],
        "frontier": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
