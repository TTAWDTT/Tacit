"""Exact stratified Private Match controls for non-uniform prior transfer.

This model-free runner builds real role-separated task episodes for every
coordinate pair with the exact multiplicity implied by rational independent
evaluation priors. It compares a frozen training-prior binary codebook with a
codebook adapted to the evaluation prior at equal payload width.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from math import lcm
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from experiments.private_match_v0_3.bit_frontier import optimal_nonuniform_codebook
from experiments.private_match_v0_3.generate_tasks import (
    _HMACRandom,
    _validate_q,
    _validate_task_key,
    generate_episode_for_target,
    oracle_answer,
    score_answer,
    task_key_id,
)


def _prior(values: Sequence[int | Fraction], *, name: str) -> tuple[Fraction, ...]:
    if not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a non-empty probability sequence")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in values):
        raise ValueError(f"{name} entries must be exact integers or fractions")
    result = tuple(Fraction(value) for value in values)
    if any(value < 0 for value in result) or sum(result, Fraction(0)) != 1:
        raise ValueError(f"{name} must contain non-negative probabilities summing exactly to 1")
    return result


def _stratified_targets(
    probabilities_x: tuple[Fraction, ...], probabilities_y: tuple[Fraction, ...], *,
    max_episodes: int,
) -> list[tuple[int, int]]:
    denominator_x = lcm(*(probability.denominator for probability in probabilities_x))
    denominator_y = lcm(*(probability.denominator for probability in probabilities_y))
    episode_count = denominator_x * denominator_y
    if episode_count > max_episodes:
        raise ValueError(
            f"exact rational cohort requires {episode_count} episodes, over cap {max_episodes}"
        )
    counts_x = [int(probability * denominator_x) for probability in probabilities_x]
    counts_y = [int(probability * denominator_y) for probability in probabilities_y]
    return [
        (x_index, y_index)
        for x_index, x_count in enumerate(counts_x)
        for y_index, y_count in enumerate(counts_y)
        for _ in range(x_count * y_count)
    ]


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def evaluate_prior_shift(
    *, task_key: bytes,
    probabilities_x_train: Sequence[int | Fraction],
    probabilities_y_train: Sequence[int | Fraction],
    probabilities_x_eval: Sequence[int | Fraction],
    probabilities_y_eval: Sequence[int | Fraction],
    payload_bits_x: int,
    payload_bits_y: int,
    seed: int = 900_000,
    max_episodes: int = 100_000,
) -> dict[str, Any]:
    """Compare frozen and prior-adapted codebooks on an exact target cohort.

    Rational probabilities are converted to their smallest exact integer
    multiplicities. The Cartesian product gives an exact finite cohort for
    independent coordinates; a keyed permutation hides ordering effects.
    Candidate-table order and IDs stay independently keyed per episode.
    """
    _validate_task_key(task_key)
    priors = {
        "x_train": _prior(probabilities_x_train, name="probabilities_x_train"),
        "y_train": _prior(probabilities_y_train, name="probabilities_y_train"),
        "x_eval": _prior(probabilities_x_eval, name="probabilities_x_eval"),
        "y_eval": _prior(probabilities_y_eval, name="probabilities_y_eval"),
    }
    q = len(priors["x_train"])
    _validate_q(q)
    if any(len(prior) != q for prior in priors.values()):
        raise ValueError("all train and evaluation priors must have the same q-value support")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(max_episodes, bool) or not isinstance(max_episodes, int) or max_episodes < 1:
        raise ValueError("max_episodes must be a positive integer")

    frozen_x = optimal_nonuniform_codebook(
        probabilities=priors["x_train"], payload_bits=payload_bits_x,
    )
    frozen_y = optimal_nonuniform_codebook(
        probabilities=priors["y_train"], payload_bits=payload_bits_y,
    )
    adapted_x = optimal_nonuniform_codebook(
        probabilities=priors["x_eval"], payload_bits=payload_bits_x,
    )
    adapted_y = optimal_nonuniform_codebook(
        probabilities=priors["y_eval"], payload_bits=payload_bits_y,
    )
    targets = _stratified_targets(
        priors["x_eval"], priors["y_eval"], max_episodes=max_episodes,
    )
    _HMACRandom(task_key, domain="prior-shift-target-order", seed=seed).shuffle(targets)

    correct = {"frozen_training_codebook": 0, "evaluation_adapted_codebook": 0}
    for index, (x_index, y_index) in enumerate(targets):
        episode_seed = seed + index
        episode = generate_episode_for_target(
            episode_id=f"pmt3ps-{episode_seed:012d}", seed=episode_seed, q=q,
            task_key=task_key, x_index=x_index, y_index=y_index,
        )
        sender_x, sender_y, receiver, gold = episode
        for name, code_x, code_y in (
            ("frozen_training_codebook", frozen_x, frozen_y),
            ("evaluation_adapted_codebook", adapted_x, adapted_y),
        ):
            decoded_x = code_x.decode(code_x.encode(x_index))
            decoded_y = code_y.decode(code_y.encode(y_index))
            answer = oracle_answer(receiver, f"x{decoded_x:04d}", f"y{decoded_y:04d}")
            correct[name] += score_answer(receiver, gold, answer)

    episode_count = len(targets)
    expected = {
        "frozen_training_codebook": (
            frozen_x.success_probability(priors["x_eval"])
            * frozen_y.success_probability(priors["y_eval"])
        ),
        "evaluation_adapted_codebook": (
            adapted_x.success_probability(priors["x_eval"])
            * adapted_y.success_probability(priors["y_eval"])
        ),
    }
    observed = {
        name: Fraction(successes, episode_count)
        for name, successes in correct.items()
    }
    if observed != expected:
        raise AssertionError("stratified episode outcomes disagree with exact codebook probabilities")

    return {
        "schema_version": "tlu.private-match-prior-shift.v1",
        "q": q,
        "episode_count": episode_count,
        "seed_start": seed,
        "task_key_id": task_key_id(task_key),
        "payload_bits": {"sender_x": payload_bits_x, "sender_y": payload_bits_y},
        "target_sampling": "exact rational independent-product stratification; keyed order permutation",
        "priors": {name: [_fraction_text(value) for value in prior]
                   for name, prior in priors.items()},
        "conditions": {
            name: {
                "successes": correct[name],
                "episodes": episode_count,
                "success_fraction": _fraction_text(observed[name]),
                "theoretical_success_fraction": _fraction_text(expected[name]),
            }
            for name in correct
        },
        "limits": [
            "model-free deterministic codebook oracle; no LLM inference",
            "exact rational cohort, not a random sample or population confidence interval",
            "training priors and codebook setup are assumed shared and free",
            "evaluation coordinates are independent and candidate tables are complete Cartesian products",
            "payload bits exclude framing, instructions, serialization envelope, and inference cost",
        ],
    }


def _parse_prior(value: str) -> tuple[Fraction, ...]:
    try:
        return tuple(Fraction(item.strip()) for item in value.split(","))
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("priors must be comma-separated exact fractions") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-key-file", type=Path, required=True)
    parser.add_argument("--train-x", type=_parse_prior, required=True)
    parser.add_argument("--train-y", type=_parse_prior, required=True)
    parser.add_argument("--eval-x", type=_parse_prior, required=True)
    parser.add_argument("--eval-y", type=_parse_prior, required=True)
    parser.add_argument("--bits-x", type=int, required=True)
    parser.add_argument("--bits-y", type=int, required=True)
    parser.add_argument("--seed", type=int, default=900_000)
    parser.add_argument("--max-episodes", type=int, default=100_000)
    args = parser.parse_args()
    try:
        task_key = args.task_key_file.read_bytes()
        result = evaluate_prior_shift(
            task_key=task_key,
            probabilities_x_train=args.train_x,
            probabilities_y_train=args.train_y,
            probabilities_x_eval=args.eval_x,
            probabilities_y_eval=args.eval_y,
            payload_bits_x=args.bits_x,
            payload_bits_y=args.bits_y,
            seed=args.seed,
            max_episodes=args.max_episodes,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
