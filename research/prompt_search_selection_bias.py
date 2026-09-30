"""Exact iid-Bernoulli sensitivity calculation for selecting the best prompt.

This is a deliberately simple reference calculation, not a model of the
clustered, adaptive v0.4 search. It computes E[max(X_1, ..., X_M)] / n - p,
where each X_i is an independent Binomial(n, p) score.
"""

from __future__ import annotations

import argparse
import math
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
    args = parser.parse_args()
    print(render_table(args.n, args.p, args.candidates))


if __name__ == "__main__":
    main()
