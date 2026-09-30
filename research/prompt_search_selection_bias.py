"""Exact iid-Bernoulli sensitivity calculation for selecting the best prompt.

This is a deliberately simple reference calculation, not a model of the
clustered, adaptive v0.4 search. It computes E[max(X_1, ..., X_M)] / n - p,
where each X_i is an independent Binomial(n, p) score.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from collections.abc import Iterable


def expected_selection_optimism(n: int, p: float, candidates: int) -> float:
    """Return E[max candidate accuracy] - p under iid Binomial(n, p) scores."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1 inclusive")
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates <= 0:
        raise ValueError("candidates must be a positive integer")

    probabilities = [
        math.comb(n, successes) * p**successes * (1.0 - p) ** (n - successes)
        for successes in range(n + 1)
    ]
    cdf: list[float] = []
    cumulative = 0.0
    for probability in probabilities:
        cumulative += probability
        cdf.append(cumulative)

    expected_max_count = sum(
        1.0 - cdf[k - 1] ** candidates for k in range(1, n + 1)
    )
    return expected_max_count / n - p


def paired_cluster_selection_sensitivity(
    clusters: int,
    cluster_size: int,
    p: float,
    candidates: int,
    intracluster_correlation: float,
    replicates: int = 10_000,
    seed: int = 37,
) -> tuple[float, float]:
    """Monte Carlo optimism for a paired beta-binomial random-intercept null.

    Each task cluster has a latent difficulty q_j shared by all candidates.
    Conditional on q_j, candidate outcomes are independent Bernoulli(q_j).
    The returned pair is mean(max candidate accuracy - p) and its Monte Carlo
    standard error. This sensitivity model is not a fitted model of v0.4.
    """
    for name, value in (("clusters", clusters), ("cluster_size", cluster_size),
                        ("candidates", candidates), ("replicates", replicates)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1 inclusive")
    if (
        not math.isfinite(intracluster_correlation)
        or not 0.0 <= intracluster_correlation <= 1.0
    ):
        raise ValueError("intracluster_correlation must be between 0 and 1 inclusive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    # At endpoint base rates, all outcomes are deterministic under this null.
    if p in (0.0, 1.0):
        return 0.0, 0.0

    rng = random.Random(seed)
    icc = intracluster_correlation
    concentration = (1.0 - icc) / icc if 0.0 < icc < 1.0 else None
    observations: list[float] = []
    total = clusters * cluster_size
    for _ in range(replicates):
        scores = [0] * candidates
        for _ in range(clusters):
            if icc == 0.0:
                difficulty = p
            elif icc == 1.0:
                difficulty = float(rng.random() < p)
            else:
                assert concentration is not None
                difficulty = rng.betavariate(
                    p * concentration, (1.0 - p) * concentration
                )
            for _ in range(cluster_size):
                for candidate in range(candidates):
                    scores[candidate] += rng.random() < difficulty
        observations.append(max(scores) / total - p)

    estimate = statistics.mean(observations)
    mcse = statistics.stdev(observations) / math.sqrt(replicates)
    return estimate, mcse


def render_table(
    sample_sizes: Iterable[int], base_rates: Iterable[float], candidate_counts: Iterable[int]
) -> str:
    lines = ["n,p,candidates,expected_optimism"]
    for n in sample_sizes:
        for p in base_rates:
            for candidates in candidate_counts:
                value = expected_selection_optimism(n, p, candidates)
                lines.append(f"{n},{p:.2f},{candidates},{value:.6f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 12, 64])
    parser.add_argument("--p", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--candidates", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--paired-cluster-sensitivity",
        action="store_true",
        help="also simulate the paired beta-binomial random-intercept null at p=0.5",
    )
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--cluster-size", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    print(render_table(args.n, args.p, args.candidates))
    if args.paired_cluster_sensitivity:
        print("\npaired beta-binomial random-intercept sensitivity (p=0.50)")
        print("icc,candidates,expected_optimism,mcse,replicates,seed")
        for icc in (0.0, 0.10, 0.25, 0.50):
            for candidates in args.candidates:
                estimate, mcse = paired_cluster_selection_sensitivity(
                    args.clusters,
                    args.cluster_size,
                    0.5,
                    candidates,
                    icc,
                    args.replicates,
                    args.seed,
                )
                print(
                    f"{icc:.2f},{candidates},{estimate:.6f},{mcse:.6f},"
                    f"{args.replicates},{args.seed}"
                )


if __name__ == "__main__":
    main()
