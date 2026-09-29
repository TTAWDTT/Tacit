"""Distribution-free split-level interval and assumption-driven power sensitivity.

The experimental unit is one independently generated composition split. A
paired protocol effect is the difference in split-level success rates and is
therefore bounded by [-1, 1]. The Hoeffding interval below is finite-sample
valid for independent split effects. Power is estimated only under explicitly
provided discrete distributions; it is not estimated from model outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SCENARIOS: dict[str, list[dict[str, float]]] = {
    "stable_effect_0.10": [{"difference": 0.10, "probability": 1.0}],
    "moderate_heterogeneity_mean_0.10": [
        {"difference": 0.20, "probability": 0.75},
        {"difference": -0.20, "probability": 0.25},
    ],
    "rare_large_gain_mean_0.10": [
        {"difference": 0.50, "probability": 0.20},
        {"difference": 0.00, "probability": 0.80},
    ],
    "zero_mean_symmetric_null": [
        {"difference": -0.50, "probability": 0.50},
        {"difference": 0.50, "probability": 0.50},
    ],
}
DEFAULT_CLUSTERS = (25, 50, 100, 250, 500, 1000)
DEFAULT_ALPHA = 0.05
DEFAULT_REPLICATES = 2000


def hoeffding_radius(*, clusters: int, alpha: float) -> float:
    """Two-sided Hoeffding radius for an iid mean of values in [-1, 1]."""
    if clusters < 1:
        raise ValueError("clusters must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return math.sqrt(2 * math.log(2 / alpha) / clusters)


def validate_scenario(name: str, support: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not name or not support:
        raise ValueError("each scenario needs a non-empty name and support")
    values: list[float] = []
    probabilities: list[float] = []
    for atom in support:
        value = float(atom["difference"])
        probability = float(atom["probability"])
        if not -1 <= value <= 1:
            raise ValueError(f"scenario {name!r} has difference outside [-1, 1]")
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(f"scenario {name!r} has an invalid probability")
        values.append(value)
        probabilities.append(probability)
    weights = np.asarray(probabilities, dtype=float)
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"scenario {name!r} probabilities must sum to 1")
    return np.asarray(values, dtype=float), weights


def scenario_power(
    *, support: list[dict[str, float]], clusters: int, alpha: float,
    replicates: int, rng: np.random.Generator, chunk_size: int = 256,
) -> dict[str, float | int]:
    """Estimate rejection probability under a discrete split-effect scenario."""
    if replicates < 1 or chunk_size < 1:
        raise ValueError("replicates and chunk_size must be positive")
    values, probabilities = validate_scenario("scenario", support)
    radius = hoeffding_radius(clusters=clusters, alpha=alpha)
    rejected = 0
    completed = 0
    while completed < replicates:
        batch = min(chunk_size, replicates - completed)
        draws = rng.choice(values, size=(batch, clusters), p=probabilities)
        means = draws.mean(axis=1)
        rejected += int(np.count_nonzero((means > radius) | (means < -radius)))
        completed += batch
    estimate = rejected / replicates
    return {
        "clusters": clusters,
        "hoeffding_radius": radius,
        "rejections": rejected,
        "replicates": replicates,
        "estimated_power_or_type1": estimate,
        "monte_carlo_se": math.sqrt(estimate * (1 - estimate) / replicates),
    }


def analyze(
    *, scenarios: dict[str, list[dict[str, float]]], cluster_counts: tuple[int, ...],
    alpha: float = DEFAULT_ALPHA, replicates: int = DEFAULT_REPLICATES,
    seed: int = 0,
) -> dict[str, Any]:
    if not cluster_counts or any(n < 1 for n in cluster_counts):
        raise ValueError("cluster_counts must contain positive integers")
    if not scenarios:
        raise ValueError("at least one scenario is required")
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for name, support in scenarios.items():
        values, probabilities = validate_scenario(name, support)
        mean = float(np.dot(values, probabilities))
        variance = float(np.dot((values - mean) ** 2, probabilities))
        output[name] = {
            "assumed_mean_difference": mean,
            "assumed_split_effect_sd": math.sqrt(variance),
            "support": support,
            "operating_characteristics": [
                scenario_power(
                    support=support, clusters=n, alpha=alpha,
                    replicates=replicates, rng=rng,
                )
                for n in cluster_counts
            ],
        }
    return {
        "schema": "tlu.split-mean-power-sensitivity.v1",
        "estimand": "mean paired difference in split-level exact-selection rates",
        "independent_unit": "independently generated composition split",
        "outcome_bound": [-1, 1],
        "interval": "two-sided Hoeffding interval for iid bounded split effects",
        "alpha": alpha,
        "cluster_counts": list(cluster_counts),
        "monte_carlo_replicates": replicates,
        "seed": seed,
        "interpretation": (
            "Sensitivity analysis only. Discrete split-effect distributions are assumptions, "
            "not estimates from model data. Hoeffding coverage requires independent split draws; "
            "the bound may be very conservative. estimated_power_or_type1 is Monte Carlo power "
            "for nonzero-mean scenarios and type-I rejection probability for a zero-mean scenario."
        ),
        "scenarios": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, help="JSON object mapping names to discrete supports")
    parser.add_argument("--clusters", type=int, nargs="+", default=DEFAULT_CLUSTERS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8")) if args.scenarios else DEFAULT_SCENARIOS
    report = analyze(
        scenarios=scenarios, cluster_counts=tuple(args.clusters), alpha=args.alpha,
        replicates=args.replicates, seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
