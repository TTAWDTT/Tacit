"""Reproduce a fixed-compressor counterexample to a per-query KV bound.

This is a deterministic, model-free mathematical audit. It makes no model
calls and uses the two-key, one-dimensional attention construction documented
in research/CONDENSEFLOW_AUDIT_V0_1.md.
"""
from __future__ import annotations

import argparse
import json
from math import exp, sqrt, tanh


def fixed_compressor_counterexample(query_magnitude: float = 5.0) -> dict[str, float | int | str]:
    if query_magnitude <= 0:
        raise ValueError("query_magnitude must be positive")

    # Keys and values are [+1, -1]; queries are [+q, -q]; d_head = 1.
    # Each query's best one-position attention mass is sigmoid(2q).
    rho = 1.0 / (1.0 + exp(-2.0 * query_magnitude))
    theorem_rhs = 2.0 * (1.0 - rho) * sqrt(2.0)  # V_max=1, n_q=2

    # Any one-row stochastic A=[p, 1-p] compresses K and V to one pair.
    # A one-position cache returns the same scalar c for both queries.
    # The best shared c is zero, yielding this exact minimum Frobenius error.
    minimum_fixed_matrix_error = sqrt(2.0) * tanh(query_magnitude)

    return {
        "construction": "keys=[1,-1], values=[1,-1], queries=[q,-q], d_head=1, K=1",
        "query_magnitude": query_magnitude,
        "n_queries": 2,
        "attention_concentration_rho": rho,
        "theorem_bound": theorem_rhs,
        "minimum_error_over_one_fixed_row_stochastic_A": minimum_fixed_matrix_error,
        "violation_ratio": minimum_fixed_matrix_error / theorem_rhs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-magnitude", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(fixed_compressor_counterexample(args.query_magnitude), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
