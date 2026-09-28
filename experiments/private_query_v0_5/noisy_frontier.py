#!/usr/bin/env python3
"""Exhaustive exact finite RAC frontier through a binary symmetric channel."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import product
from pathlib import Path


N = 4
PRIORS = {
    "uniform": (1, 1, 1, 1),
    "skewed_first_coordinate_7_to_1": (7, 1, 1, 1),
}
NOISE_LEVELS = {
    "0/1": (0, 1),
    "1/8": (1, 8),
    "1/4": (1, 4),
    "1/2": (1, 2),
}


def weighted_distances(weights: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    count = 1 << N
    return tuple(
        tuple(sum(weights[i] for i in range(N) if ((source ^ reconstruction) >> i) & 1) for reconstruction in range(count))
        for source in range(count)
    )


def channel_numerators(budget: int, noise_numerator: int, noise_denominator: int) -> tuple[tuple[int, ...], ...]:
    alphabet = 1 << budget
    return tuple(
        tuple(
            noise_numerator ** ((sent ^ received).bit_count())
            * (noise_denominator - noise_numerator) ** (budget - (sent ^ received).bit_count())
            for received in range(alphabet)
        )
        for sent in range(alphabet)
    )


def exact_cell(
    budget: int,
    noise_label: str,
    training_prior_name: str,
    transfer_prior_name: str,
) -> dict[str, object]:
    if budget not in (0, 1, 2):
        raise ValueError("v0.5 supports message budgets 0, 1, and 2 bits")
    if noise_label not in NOISE_LEVELS:
        raise ValueError("noise level is not in the frozen preregistration")
    if training_prior_name not in PRIORS or transfer_prior_name not in PRIORS:
        raise ValueError("query prior is not in the frozen preregistration")

    noise_numerator, noise_denominator = NOISE_LEVELS[noise_label]
    probability_denominator = noise_denominator**budget
    transition = channel_numerators(budget, noise_numerator, noise_denominator)
    alphabet = 1 << budget
    reconstruction_count = 1 << N
    source_count = reconstruction_count
    training_weights = PRIORS[training_prior_name]
    transfer_weights = PRIORS[transfer_prior_name]
    training_dist = weighted_distances(training_weights)
    transfer_dist = weighted_distances(transfer_weights)

    best_training_distortion: int | None = None
    optimal_decoder_count = 0
    example_decoder: tuple[int, ...] | None = None
    transfer_distortion_sum = 0
    transfer_distortion_min: int | None = None
    transfer_distortion_max: int | None = None

    for decoder in product(range(reconstruction_count), repeat=alphabet):
        train_total = 0
        transfer_total = 0
        for source in range(source_count):
            sent_costs = tuple(
                sum(transition[sent][received] * training_dist[source][decoder[received]] for received in range(alphabet))
                for sent in range(alphabet)
            )
            sent_message = min(range(alphabet), key=lambda message: (sent_costs[message], message))
            train_total += sent_costs[sent_message]
            transfer_total += sum(
                transition[sent_message][received] * transfer_dist[source][decoder[received]]
                for received in range(alphabet)
            )

        if best_training_distortion is None or train_total < best_training_distortion:
            best_training_distortion = train_total
            optimal_decoder_count = 1
            example_decoder = decoder
            transfer_distortion_sum = transfer_total
            transfer_distortion_min = transfer_total
            transfer_distortion_max = transfer_total
        elif train_total == best_training_distortion:
            optimal_decoder_count += 1
            transfer_distortion_sum += transfer_total
            transfer_distortion_min = min(transfer_distortion_min, transfer_total)
            transfer_distortion_max = max(transfer_distortion_max, transfer_total)

    assert best_training_distortion is not None and example_decoder is not None
    assert transfer_distortion_min is not None and transfer_distortion_max is not None
    train_denominator = source_count * sum(training_weights) * probability_denominator
    transfer_denominator = source_count * sum(transfer_weights) * probability_denominator
    training_accuracy = Fraction(train_denominator - best_training_distortion, train_denominator)
    transfer_count = optimal_decoder_count
    transfer_mean_distortion = Fraction(transfer_distortion_sum, transfer_count)
    transfer_mean_accuracy = Fraction(transfer_denominator - transfer_mean_distortion, transfer_denominator)
    transfer_min_accuracy = Fraction(transfer_denominator - transfer_distortion_max, transfer_denominator)
    transfer_max_accuracy = Fraction(transfer_denominator - transfer_distortion_min, transfer_denominator)

    return {
        "n_source_bits": N,
        "message_budget_bits": budget,
        "receiver_query_prior": training_prior_name,
        "query_weights": list(training_weights),
        "frozen_transfer_query_prior": transfer_prior_name,
        "transfer_query_weights": list(transfer_weights),
        "binary_symmetric_channel_flip_probability": noise_label,
        "channel_transition_probability_denominator": probability_denominator,
        "decoder_maps_enumerated": reconstruction_count**alphabet,
        "optimal_decoder_map_count": optimal_decoder_count,
        "example_optimal_decoder_map_by_received_word": list(example_decoder),
        "training_prior_optimal_success_fraction": f"{training_accuracy.numerator}/{training_accuracy.denominator}",
        "training_prior_optimal_success_probability": float(training_accuracy),
        "frozen_optimal_protocol_transfer": {
            "success_min_fraction": f"{transfer_min_accuracy.numerator}/{transfer_min_accuracy.denominator}",
            "success_mean_fraction": f"{transfer_mean_accuracy.numerator}/{transfer_mean_accuracy.denominator}",
            "success_max_fraction": f"{transfer_max_accuracy.numerator}/{transfer_max_accuracy.denominator}",
        },
    }


def generate_report() -> dict[str, object]:
    rows = []
    for noise in NOISE_LEVELS:
        for budget in range(3):
            for training_prior, transfer_prior in (
                ("uniform", "skewed_first_coordinate_7_to_1"),
                ("skewed_first_coordinate_7_to_1", "uniform"),
            ):
                rows.append(exact_cell(budget, noise, training_prior, transfer_prior))

    noiseless = [row for row in rows if row["binary_symmetric_channel_flip_probability"] == "0/1"]
    prior_results = {}
    for row in noiseless:
        prior_results[(row["receiver_query_prior"], row["message_budget_bits"])] = Fraction(
            row["training_prior_optimal_success_fraction"]
        )
    if [prior_results[("uniform", b)] for b in range(3)] != [Fraction(1, 2), Fraction(11, 16), Fraction(13, 16)]:
        raise AssertionError("noiseless uniform frontier does not match the pinned n=4 reference")
    if [prior_results[("skewed_first_coordinate_7_to_1", b)] for b in range(3)] != [Fraction(1, 2), Fraction(17, 20), Fraction(37, 40)]:
        raise AssertionError("noiseless skewed frontier does not match the pinned n=4 reference")

    for prior_name in PRIORS:
        half_noise = [
            Fraction(row["training_prior_optimal_success_fraction"])
            for row in rows
            if row["receiver_query_prior"] == prior_name
            and row["binary_symmetric_channel_flip_probability"] == "1/2"
        ]
        if half_noise != [Fraction(1, 2)] * 3:
            raise AssertionError("a fully noisy BSC must reduce every budget to chance accuracy")
        for noise in NOISE_LEVELS:
            scores = [
                Fraction(row["training_prior_optimal_success_fraction"])
                for row in rows
                if row["receiver_query_prior"] == prior_name
                and row["binary_symmetric_channel_flip_probability"] == noise
            ]
            if scores != sorted(scores):
                raise AssertionError("optimal success must be nondecreasing with message budget")

    return {
        "schema": "tlu.private_query.noisy_frontier.v0_5",
        "study": "private_query_noisy_channel_frontier_v0_5",
        "method": "exhaustive decoder-map enumeration; exact Bayes-optimal deterministic sender for each source vector",
        "model_calls": 0,
        "source_distribution": "uniform over all 4-bit vectors",
        "receiver_query_is_private_to_sender": True,
        "decoder_maps_allow_duplicate_reconstructions": True,
        "shared_code_discovery_storage_and_setup_cost_included": False,
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
