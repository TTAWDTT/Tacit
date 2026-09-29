from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Any


SCHEMA = "tlu.candidate-set-random-code-frontier.v1"


def random_code_success(*, candidate_size: int, message_symbols: int) -> Fraction:
    """Expected Bayes exact-selection accuracy under an ideal random codebook.

    Each distinct target meaning receives an independent uniform symbol from
    an alphabet of ``message_symbols`` values. The sender sees only the target;
    the receiver sees a uniformly targeted candidate set and picks one member
    from the target's code class. The codebook is shared before the episode.
    """
    if isinstance(candidate_size, bool) or not isinstance(candidate_size, int) or candidate_size < 1:
        raise ValueError("candidate_size must be a positive integer")
    if isinstance(message_symbols, bool) or not isinstance(message_symbols, int) or message_symbols < 1:
        raise ValueError("message_symbols must be a positive integer")
    m = message_symbols
    k = candidate_size
    return Fraction(m, k) * (1 - Fraction(m - 1, m) ** k)


def build_frontier(*, candidate_size: int, max_bits: int) -> dict[str, Any]:
    if isinstance(max_bits, bool) or not isinstance(max_bits, int) or not 0 <= max_bits <= 16:
        raise ValueError("max_bits must be an integer in 0..16")
    rows = []
    for bits in range(max_bits + 1):
        symbols = 1 << bits
        accuracy = random_code_success(candidate_size=candidate_size, message_symbols=symbols)
        rows.append({
            "payload_bits": bits,
            "message_symbols": symbols,
            "expected_exact_selection": float(accuracy),
            "exact_fraction": f"{accuracy.numerator}/{accuracy.denominator}",
        })
    return {
        "schema": SCHEMA,
        "baseline": "ideal_random_codebook_with_receiver_candidate_side_information",
        "is_language_or_llm_result": False,
        "candidate_size": candidate_size,
        "max_bits": max_bits,
        "assumptions": [
            "the hidden target is uniform among distinct rows in the candidate set",
            "the sender observes only the target; the receiver observes the whole candidate set",
            "the shared codebook is a target-independent uniformly random function into the message alphabet",
            "the receiver uses the Bayes-optimal candidate within the received-symbol class",
            "payload cost is a fixed-width ideal bit count; framing, codebook/key setup, inference, and compute are excluded",
        ],
        "frontier": rows,
        "limits": [
            "this is a coding-theory reference, not a proposed language or measured LLM protocol",
            "a pseudorandom hash implementation only approximates the ideal random-function assumption",
            "operational comparison must charge codebook/seed distribution and full application/inference cost",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-size", type=int, required=True)
    parser.add_argument("--max-bits", type=int, default=4)
    args = parser.parse_args()
    try:
        result = build_frontier(candidate_size=args.candidate_size, max_bits=args.max_bits)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
