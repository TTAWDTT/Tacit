#!/usr/bin/env python3
"""Exact one-bit private-query code enumeration; no models or dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_encoder(n: int, truth_table: int) -> tuple[int, tuple[int, ...]]:
    """Return correct (source,index) pairs and optimal decoder bits.

    Bit x of truth_table is the message for source integer x, where source bit
    i is read from the i-th least-significant bit. Decoder entry 2*i+m gives
    its output for query i and message m. Ties decode to zero deterministically.
    """
    ones_by_message = [[0] * n for _ in range(2)]
    totals_by_message = [0, 0]
    for source in range(1 << n):
        message = (truth_table >> source) & 1
        totals_by_message[message] += 1
        for i in range(n):
            ones_by_message[message][i] += (source >> i) & 1

    decoder = []
    correct_pairs = 0
    for i in range(n):
        for message in range(2):
            ones = ones_by_message[message][i]
            zeros = totals_by_message[message] - ones
            guess = int(ones > zeros)
            decoder.append(guess)
            correct_pairs += max(ones, zeros)
    return correct_pairs, tuple(decoder)


def enumerate_optimum(n: int) -> dict[str, object]:
    if n < 1 or n > 4:
        raise ValueError("exact truth-table enumeration supports 1 <= n <= 4")

    source_count = 1 << n
    encoder_count = 1 << source_count
    best_correct = -1
    best_codes: list[dict[str, object]] = []
    for truth_table in range(encoder_count):
        correct, decoder = evaluate_encoder(n, truth_table)
        candidate = {
            "truth_table_hex": hex(truth_table),
            "decoder_bits_by_query_then_message": list(decoder),
        }
        if correct > best_correct:
            best_correct = correct
            best_codes = [candidate]
        elif correct == best_correct and len(best_codes) < 8:
            best_codes.append(candidate)

    total_pairs = n * source_count
    numerator, denominator = best_correct, total_pairs
    from math import gcd

    divisor = gcd(numerator, denominator)
    return {
        "n_source_bits": n,
        "query_distribution": "uniform over coordinates",
        "source_distribution": "uniform over bit vectors",
        "sender_message_budget_bits": 1,
        "encoder_count_enumerated": encoder_count,
        "correct_source_query_pairs": best_correct,
        "total_source_query_pairs": total_pairs,
        "success_fraction": f"{numerator // divisor}/{denominator // divisor}",
        "success_probability": best_correct / total_pairs,
        "fourier_upper_bound": 0.5 + 0.5 / (n**0.5),
        "examples_of_optimal_encoders_first_8": best_codes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_n <= 4:
        parser.error("--max-n must be between 1 and 4")

    report = {
        "schema": "tlu.private_query.enumeration.v1",
        "method": "exhaustive deterministic encoders; Bayes-optimal decoders",
        "model_calls": 0,
        "controls": {
            "no_message_success_probability": 0.5,
            "query_given_to_sender_one_bit_success_probability": 1.0,
            "full_source_n_bit_success_probability": 1.0,
            "query_disclosure_to_sender_included_in_one_bit_channel_budget": False,
        },
        "results": [enumerate_optimum(n) for n in range(1, args.max_n + 1)],
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
