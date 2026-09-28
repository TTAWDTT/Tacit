#!/usr/bin/env python3
"""Exact finite rate-distortion frontiers for private-coordinate queries."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import combinations
from math import comb
from pathlib import Path


def encode_by_nearest_codeword(
    n: int,
    codebook: tuple[int, ...],
    query_weights: tuple[int, ...],
) -> tuple[tuple[int, ...], int]:
    assignments = []
    total = 0
    for source in range(1 << n):
        distances = [
            sum(query_weights[i] for i in range(n) if ((source >> i) ^ (codeword >> i)) & 1)
            for codeword in codebook
        ]
        selected = min(range(len(codebook)), key=lambda slot: (distances[slot], codebook[slot]))
        assignments.append(selected)
        total += distances[selected]
    return tuple(assignments), total


def accuracy_fraction(n: int, distortion: int, query_weights: tuple[int, ...]) -> Fraction:
    denominator = (1 << n) * sum(query_weights)
    return Fraction(denominator - distortion, denominator)


def frozen_protocol_distortion(
    n: int,
    codebook: tuple[int, ...],
    assignments: tuple[int, ...],
    query_weights: tuple[int, ...],
) -> int:
    return sum(
        sum(
            query_weights[i]
            for i in range(n)
            if ((source >> i) ^ (codebook[assignments[source]] >> i)) & 1
        )
        for source in range(1 << n)
    )


def best_codebook(
    n: int,
    budget_bits: int,
    query_weights: tuple[int, ...],
    transfer_weights: tuple[int, ...],
) -> dict[str, object]:
    """Find the optimal fixed-length codebook under weighted Hamming loss.

    Each codeword is the n-bit reconstruction a receiver would produce for a
    message. For each source vector, the optimal sender selects its nearest
    codeword under the query weights. Thus enumerating q codewords exactly
    optimizes all deterministic q-message encoders and decoders, where q=2^b.
    """
    if not 1 <= n <= 4:
        raise ValueError("exact codebook enumeration supports 1 <= n <= 4")
    if len(query_weights) != n or any(w <= 0 for w in query_weights):
        raise ValueError("query_weights must contain n positive integers")
    if not 0 <= budget_bits <= n:
        raise ValueError("budget_bits must be between zero and n")

    source_count = 1 << n
    codebook_size = 1 << budget_bits
    best_total = None
    best_codebooks = 0
    first_best: tuple[int, ...] | None = None
    optimal_transfer_scores: list[Fraction] = []
    codebooks_checked = 0
    for codebook in combinations(range(source_count), codebook_size):
        codebooks_checked += 1
        assignments, total_distortion = encode_by_nearest_codeword(n, codebook, query_weights)
        transfer_distortion = frozen_protocol_distortion(n, codebook, assignments, transfer_weights)
        transfer_accuracy = accuracy_fraction(n, transfer_distortion, transfer_weights)
        if best_total is None or total_distortion < best_total:
            best_total = total_distortion
            best_codebooks = 1
            first_best = codebook
            optimal_transfer_scores = [transfer_accuracy]
        elif total_distortion == best_total:
            best_codebooks += 1
            optimal_transfer_scores.append(transfer_accuracy)

    assert best_total is not None and first_best is not None
    total_weighted_pairs = source_count * sum(query_weights)
    accuracy = Fraction(total_weighted_pairs - best_total, total_weighted_pairs)
    transfer_min = min(optimal_transfer_scores)
    transfer_max = max(optimal_transfer_scores)
    transfer_mean = sum(optimal_transfer_scores, Fraction()) / len(optimal_transfer_scores)
    return {
        "n_source_bits": n,
        "message_budget_bits": budget_bits,
        "message_alphabet_size": codebook_size,
        "query_weights": list(query_weights),
        "query_distribution": "integer weights normalized to sum 1",
        "source_distribution": "uniform over bit vectors",
        "codebooks_enumerated": codebooks_checked,
        "optimal_codebook_count": best_codebooks,
        "example_optimal_reconstruction_codebook": list(first_best),
        "minimum_weighted_hamming_distortion_sum": best_total,
        "weighted_hamming_denominator": total_weighted_pairs,
        "success_fraction": f"{accuracy.numerator}/{accuracy.denominator}",
        "success_probability": float(accuracy),
        "optimal_training_protocols_frozen_and_tested_on_other_query_prior": {
            "query_weights": list(transfer_weights),
            "encoder_rule": "nearest codeword under training prior; ties choose lowest codeword",
            "success_min_fraction": f"{transfer_min.numerator}/{transfer_min.denominator}",
            "success_min_probability": float(transfer_min),
            "success_mean_fraction": f"{transfer_mean.numerator}/{transfer_mean.denominator}",
            "success_mean_probability": float(transfer_mean),
            "success_max_fraction": f"{transfer_max.numerator}/{transfer_max.denominator}",
            "success_max_probability": float(transfer_max),
        },
    }


def generate_report() -> dict[str, object]:
    rows = []
    for n in (2, 3, 4):
        query_cases = (
            ("uniform", (1,) * n, "skewed_first_coordinate_7_to_1", (7,) + (1,) * (n - 1)),
            ("skewed_first_coordinate_7_to_1", (7,) + (1,) * (n - 1), "uniform", (1,) * n),
        )
        for query_name, weights, _transfer_name, transfer_weights in query_cases:
            for budget in range(n + 1):
                result = best_codebook(n, budget, weights, transfer_weights)
                result["query_case"] = query_name
                rows.append(result)
    lookup = {
        (row["n_source_bits"], row["query_case"], row["message_budget_bits"]): row
        for row in rows
    }
    for row in rows:
        other_case = (
            "skewed_first_coordinate_7_to_1"
            if row["query_case"] == "uniform"
            else "uniform"
        )
        target_oracle = lookup[(row["n_source_bits"], other_case, row["message_budget_bits"])]
        target_optimum = Fraction(target_oracle["success_fraction"])
        transfer = row["optimal_training_protocols_frozen_and_tested_on_other_query_prior"]
        transfer["target_prior_oracle_success_fraction"] = target_oracle["success_fraction"]
        transfer["target_prior_oracle_success_probability"] = float(target_optimum)
        for score_stat, regret_stat in (("min", "max"), ("mean", "mean"), ("max", "min")):
            transfer_score = Fraction(transfer[f"success_{score_stat}_fraction"])
            regret = target_optimum - transfer_score
            transfer[f"target_prior_regret_{regret_stat}_fraction"] = f"{regret.numerator}/{regret.denominator}"
            transfer[f"target_prior_regret_{regret_stat}_probability"] = float(regret)
    return {
        "schema": "tlu.private_query.frontier.v1",
        "method": "exhaustive reconstruction-codebook enumeration; nearest-codeword sender",
        "model_calls": 0,
        "assumptions": {
            "codebook_shared_without_cost": True,
            "query_known_to_sender": False,
            "setup_and_prompt_cost_included": False,
            "query_index_disclosure_is_not_free": True,
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    serialized = json.dumps(generate_report(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
