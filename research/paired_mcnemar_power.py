"""Sensitivity analysis for paired binary protocol-success comparisons.

The exact conditional McNemar test is evaluated on discordant pairs. Power is
unconditional over the random discordant-pair count and conditional on the
number of discordances. SciPy computes binomial tail probabilities; integer
combinatorics determine the exact-test rejection boundary.
"""
from __future__ import annotations

import json
from functools import lru_cache
from fractions import Fraction
from math import comb

import numpy as np
from scipy.stats import binom


TARGET_POWER = 0.80
FAMILY_ALPHA = Fraction(1, 20)
COMPARISON_COUNT = 6
MAX_EPISODES = 1000
SCENARIOS = (
    {"risk_difference": 0.05, "discordance": 0.15},
    {"risk_difference": 0.10, "discordance": 0.20},
    {"risk_difference": 0.15, "discordance": 0.25},
    {"risk_difference": 0.20, "discordance": 0.30},
)


@lru_cache(maxsize=None)
def _critical_boundary(discordant_pairs: int, alpha: Fraction) -> int:
    """Largest lower-tail x rejected by the two-sided exact binomial test."""
    boundary = int(binom.ppf(float(alpha) / 2, discordant_pairs, 0.5))
    cumulative = sum(comb(discordant_pairs, x) for x in range(boundary + 1))
    while boundary >= 0 and Fraction(2 * cumulative, 1 << discordant_pairs) > alpha:
        cumulative -= comb(discordant_pairs, boundary)
        boundary -= 1
    while boundary + 1 < discordant_pairs / 2:
        next_count = cumulative + comb(discordant_pairs, boundary + 1)
        if Fraction(2 * next_count, 1 << discordant_pairs) > alpha:
            break
        cumulative = next_count
        boundary += 1
    return boundary


def _conditional_rejection_probabilities(
    *, theta: float, alpha: Fraction, max_discordant_pairs: int,
) -> np.ndarray:
    """Exact-test rejection chance conditional on each discordant count."""
    probabilities = np.zeros(max_discordant_pairs + 1)
    for discordant in range(1, max_discordant_pairs + 1):
        boundary = _critical_boundary(discordant, alpha)
        if boundary >= 0:
            probabilities[discordant] = (
                binom.cdf(boundary, discordant, theta)
                + binom.sf(discordant - boundary - 1, discordant, theta)
            )
    return probabilities


def exact_conditional_mcnemar_power(
    *, episodes: int, gain_only: float, loss_only: float,
    alpha: Fraction, conditional_rejections: np.ndarray | None = None,
) -> float:
    """Power for paired outcomes with specified discordant-cell probabilities."""
    discordance = gain_only + loss_only
    if not 0 < discordance <= 1 or not 0 <= loss_only < gain_only:
        raise ValueError("require 0 <= loss_only < gain_only and positive total discordance <= 1")
    theta = gain_only / discordance
    rejections = (
        conditional_rejections
        if conditional_rejections is not None
        else _conditional_rejection_probabilities(
            theta=theta, alpha=alpha, max_discordant_pairs=episodes,
        )
    )
    counts = np.arange(episodes + 1)
    return float(np.dot(
        binom.pmf(counts, episodes, discordance),
        rejections[:episodes + 1],
    ))


def minimum_episodes(
    *, risk_difference: float, discordance: float,
    alpha: Fraction, target_power: float = TARGET_POWER,
    max_episodes: int = MAX_EPISODES,
) -> dict[str, int | float]:
    """Smallest N meeting target power under a two-sided exact test."""
    if risk_difference <= 0 or risk_difference > discordance or discordance > 1:
        raise ValueError("require 0 < risk_difference <= discordance <= 1")
    gain_only = (discordance + risk_difference) / 2
    loss_only = (discordance - risk_difference) / 2
    theta = gain_only / discordance
    rejections = _conditional_rejection_probabilities(
        theta=theta, alpha=alpha, max_discordant_pairs=max_episodes,
    )
    for episodes in range(1, max_episodes + 1):
        power = exact_conditional_mcnemar_power(
            episodes=episodes,
            gain_only=gain_only,
            loss_only=loss_only,
            alpha=alpha,
            conditional_rejections=rejections,
        )
        if power >= target_power:
            previous_power = exact_conditional_mcnemar_power(
                episodes=episodes - 1,
                gain_only=gain_only,
                loss_only=loss_only,
                alpha=alpha,
                conditional_rejections=rejections,
            ) if episodes > 1 else 0.0
            return {
                "minimum_paired_episodes": episodes,
                "power_at_minimum": power,
                "power_one_episode_less": previous_power,
            }
    raise ValueError(f"target power not reached by {max_episodes} episodes")


def main() -> None:
    family_alpha = FAMILY_ALPHA / COMPARISON_COUNT
    rows = []
    for scenario in SCENARIOS:
        risk_difference = scenario["risk_difference"]
        discordance = scenario["discordance"]
        gain_only = (discordance + risk_difference) / 2
        loss_only = (discordance - risk_difference) / 2
        row: dict[str, object] = {
            **scenario,
            "gain_only_probability": gain_only,
            "loss_only_probability": loss_only,
            "single_contrast_alpha": float(FAMILY_ALPHA),
            "six_comparison_bonferroni_alpha": float(family_alpha),
            "single_contrast": minimum_episodes(
                risk_difference=risk_difference, discordance=discordance,
                alpha=FAMILY_ALPHA,
            ),
            "six_comparison_family": minimum_episodes(
                risk_difference=risk_difference, discordance=discordance,
                alpha=family_alpha,
            ),
        }
        rows.append(row)
    print(json.dumps({
        "test": "two-sided exact conditional McNemar",
        "target_power": TARGET_POWER,
        "familywise_alpha": float(FAMILY_ALPHA),
        "bonferroni_comparison_count": COMPARISON_COUNT,
        "four_episode_best_possible_p_value": 2 / (2 ** 4),
        "interpretation": "Sensitivity analysis only; discordance assumptions are not estimated from model data.",
        "scenarios": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
