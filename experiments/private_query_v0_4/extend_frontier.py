#!/usr/bin/env python3
"""Exact n=5 private-query codebook frontiers for budgets up to two bits."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import combinations
from math import comb
from pathlib import Path


def weighted_distance_table(n: int, weights: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    count = 1 << n
    return tuple(
        tuple(sum(weights[i] for i in range(n) if ((x ^ y) >> i) & 1) for y in range(count))
        for x in range(count)
    )


def exact_frontier_cell(
    n: int,
    budget_bits: int,
    query_weights: tuple[int, ...],
    transfer_weights: tuple[int, ...],
) -> dict[str, object]:
    """Enumerate every decoder codebook and optimize its deterministic sender."""
    if n != 5 or not 0 <= budget_bits <= 2:
        raise ValueError("v0.4 supports n=5 and message budgets 0, 1, or 2 bits")
    if len(query_weights) != n or len(transfer_weights) != n:
        raise ValueError("each query prior must contain exactly n coordinate weights")
    if any(weight <= 0 for weight in query_weights + transfer_weights):
        raise ValueError("query weights must be positive integers")

    source_count = 1 << n
    alphabet = 1 << budget_bits
    train_distances = weighted_distance_table(n, query_weights)
    transfer_distances = weighted_distance_table(n, transfer_weights)
    best_distortion: int | None = None
    best_codebook_count = 0
    best_example: tuple[int, ...] | None = None
    transfer_scores: list[Fraction] = []

    for codebook in combinations(range(source_count), alphabet):
        train_total = 0
        transfer_total = 0
        for source in range(source_count):
            # Deterministic tie-breaking matches v0.2: lowest codeword wins.
            chosen = min(codebook, key=lambda codeword: (train_distances[source][codeword], codeword))
            train_total += train_distances[source][chosen]
            transfer_total += transfer_distances[source][chosen]
        if best_distortion is None or train_total < best_distortion:
            best_distortion = train_total
            best_codebook_count = 1
            best_example = codebook
            transfer_scores = [
                Fraction(
                    source_count * sum(transfer_weights) - transfer_total,
                    source_count * sum(transfer_weights),
                )
            ]
        elif train_total == best_distortion:
            best_codebook_count += 1
            transfer_scores.append(
                Fraction(
                    source_count * sum(transfer_weights) - transfer_total,
                    source_count * sum(transfer_weights),
                )
            )

    assert best_distortion is not None and best_example is not None
    denominator = source_count * sum(query_weights)
    accuracy = Fraction(denominator - best_distortion, denominator)
    transfer_mean = sum(transfer_scores, Fraction()) / len(transfer_scores)
    return {
        "n_source_bits": n,
        "message_budget_bits": budget_bits,
        "message_alphabet_size": alphabet,
        "query_weights": list(query_weights),
        "transfer_query_weights": list(transfer_weights),
        "codebooks_enumerated": comb(source_count, alphabet),
        "optimal_codebook_count": best_codebook_count,
        "example_optimal_reconstruction_codebook": list(best_example),
        "success_fraction": f"{accuracy.numerator}/{accuracy.denominator}",
        "success_probability": float(accuracy),
        "frozen_optimal_training_protocol_transfer": {
            "success_min_fraction": f"{min(transfer_scores).numerator}/{min(transfer_scores).denominator}",
            "success_mean_fraction": f"{transfer_mean.numerator}/{transfer_mean.denominator}",
            "success_max_fraction": f"{max(transfer_scores).numerator}/{max(transfer_scores).denominator}",
            "target_prior_oracle_success_fraction": None,
        },
    }


def generate_report() -> dict[str, object]:
    rows = []
    for label, query_weights, transfer_label, transfer_weights in (
        ("uniform", (1,) * 5, "skewed_first_coordinate_7_to_1", (7, 1, 1, 1, 1)),
        ("skewed_first_coordinate_7_to_1", (7, 1, 1, 1, 1), "uniform", (1,) * 5),
    ):
        for budget in range(3):
            row = exact_frontier_cell(5, budget, query_weights, transfer_weights)
            row["query_case"] = label
            row["transfer_query_case"] = transfer_label
            rows.append(row)
    by_prior = {(row["query_case"], row["message_budget_bits"]): row for row in rows}
    for row in rows:
        transfer = row["frozen_optimal_training_protocol_transfer"]
        target = by_prior[(row["transfer_query_case"], row["message_budget_bits"])]
        target_fraction = Fraction(target["success_fraction"])
        transfer["target_prior_oracle_success_fraction"] = target["success_fraction"]
        for name, regret_key in (("min", "max"), ("mean", "mean"), ("max", "min")):
            score = Fraction(transfer[f"success_{name}_fraction"])
            regret = target_fraction - score
            transfer[f"target_prior_regret_{regret_key}_fraction"] = f"{regret.numerator}/{regret.denominator}"
    return {
        "schema": "tlu.private_query.frontier.n5.v0_4",
        "method": "exact exhaustive decoder-codebook enumeration with nearest weighted-Hamming sender",
        "model_calls": 0,
        "assumptions": {
            "uniform_source_vectors": True,
            "receiver_query_private_to_sender": True,
            "integer_query_weights_normalized_to_a_probability_distribution": True,
            "codebook_discovery_storage_and_setup_cost_included": False,
            "frozen_sender_tie_break": "lowest codeword",
        },
        "candidate_codebook_counts_by_budget": {str(b): comb(32, 1 << b) for b in range(3)},
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.dumps(generate_report(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
