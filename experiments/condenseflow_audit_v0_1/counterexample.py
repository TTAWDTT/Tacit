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
    shared_rho = 1.0 - rho
    shared_support_bound = 2.0 * (1.0 - shared_rho) * sqrt(2.0)

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
        "best_common_support_mass": shared_rho,
        "query_shared_support_bound": shared_support_bound,
        "minimum_error_over_one_fixed_row_stochastic_A": minimum_fixed_matrix_error,
        "violation_ratio": minimum_fixed_matrix_error / theorem_rhs,
    }


def support_overlap_frontier(
    context_size: int = 16,
    compression_slots: tuple[int, ...] = (1, 2, 4, 8),
    distinct_salient_positions: tuple[int, ...] = (1, 2, 4, 8, 16),
    diffuse_mass: float = 0.05,
) -> dict[str, object]:
    """Compute exact attention-mass bounds in a sparse-salience toy family.

    Each query places 1-diffuse_mass on its unique salient position and spreads
    diffuse_mass uniformly over the context. This measures shared-support
    coverage only; it is not an LLM result or a model of learned LTC behavior.
    """
    if context_size < 1:
        raise ValueError("context_size must be positive")
    if not 0.0 <= diffuse_mass < 1.0:
        raise ValueError("diffuse_mass must be in [0, 1)")
    if not compression_slots or any(slot < 1 or slot > context_size for slot in compression_slots):
        raise ValueError("compression_slots must lie in [1, context_size]")
    if not distinct_salient_positions or any(count < 1 or count > context_size for count in distinct_salient_positions):
        raise ValueError("distinct_salient_positions must lie in [1, context_size]")

    rows = []
    for salient_count in distinct_salient_positions:
        for slots in compression_slots:
            individual = 1.0 - diffuse_mass + diffuse_mass * slots / context_size
            shared = (
                individual if slots >= salient_count
                else diffuse_mass * slots / context_size
            )
            rows.append({
                "context_size": context_size,
                "compression_slots": slots,
                "distinct_salient_positions": salient_count,
                "diffuse_mass": diffuse_mass,
                "per_query_top_k_mass": individual,
                "best_common_support_mass": shared,
                "shared_support_error_bound_factor": 2.0 * (1.0 - shared),
            })
    return {
        "status": "analytic toy-family prediction; not model evidence",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-magnitude", type=float, default=5.0)
    parser.add_argument("--show-support-scaling", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {
        "fixed_compressor_counterexample": fixed_compressor_counterexample(args.query_magnitude),
    }
    if args.show_support_scaling:
        report["support_overlap_frontier"] = support_overlap_frontier()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
