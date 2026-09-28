#!/usr/bin/env python3
"""Compute the exact classical one-bit random-access success curve."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import comb, pi, sqrt
from pathlib import Path


def exact_success(n: int) -> Fraction:
    if n < 1:
        raise ValueError("n must be positive")
    # Optimal deterministic n-to-1 classical RAC under uniform source/query.
    return Fraction(1, 2) + Fraction(comb(n - 1, (n - 1) // 2), 1 << n)


def selected_scales(max_n: int) -> list[int]:
    if max_n < 1:
        raise ValueError("max_n must be positive")
    values = set(range(1, min(max_n, 128) + 1))
    n = 1
    while n <= max_n:
        values.add(n)
        n *= 2
    values.add(max_n)
    return sorted(values)


def make_report(max_n: int) -> dict[str, object]:
    rows = []
    for n in selected_scales(max_n):
        p = exact_success(n)
        advantage = p - Fraction(1, 2)
        asymptotic_advantage = 1 / sqrt(2 * pi * n)
        rows.append(
            {
                "n_source_bits": n,
                "message_budget_bits": 1,
                "success_fraction": f"{p.numerator}/{p.denominator}",
                "success_probability": float(p),
                "advantage_over_chance_fraction": f"{advantage.numerator}/{advantage.denominator}",
                "advantage_over_chance": float(advantage),
                "asymptotic_advantage_1_over_sqrt_2pi_n": asymptotic_advantage,
                "exact_to_asymptotic_advantage_ratio": float(advantage) / asymptotic_advantage,
            }
        )
    return {
        "schema": "tlu.private_query.one_bit_scaling.v1",
        "prior_result": "Ambainis et al. 2009, classical n-to-1 random access code",
        "assumptions": {
            "source": "uniform over {0,1}^n",
            "receiver_query": "uniform over coordinates and hidden from sender at encode time",
            "sender_channel": "one classical bit",
            "decoder": "query-dependent deterministic output",
            "score": "average exact coordinate-retrieval probability",
        },
        "exact_formula": "1/2 + binom(n-1, floor((n-1)/2)) / 2^n",
        "asymptotic_formula": "1/2 + 1/sqrt(2*pi*n) + O(n^(-3/2))",
        "model_calls": 0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = make_report(args.max_n)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
