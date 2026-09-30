"""Exact scaling predictions for the modular higher-order split family."""
from __future__ import annotations

import argparse
import json
import math
from typing import Any


SCHEMA = "tlu.emergent-ood-split-scaling.v1"


def scaling_report(*, dimensions: int, value_count: int, candidate_count: int = 4) -> dict[str, Any]:
    """Return exact counts for the split and its one-way zero-error oracle."""
    for name, value, minimum in (
        ("dimensions", dimensions, 2),
        ("value_count", value_count, 2),
        ("candidate_count", candidate_count, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    universe = value_count ** dimensions
    if candidate_count > universe:
        raise ValueError("candidate_count cannot exceed the meaning universe")
    held_out = value_count ** (dimensions - 1)
    partial_orders = []
    for order in range(1, dimensions):
        assignments = math.comb(dimensions, order) * value_count ** order
        held_completions = value_count ** (dimensions - order - 1)
        train_completions = (value_count - 1) * held_completions
        partial_orders.append({
            "order": order,
            "partial_assignments": assignments,
            "held_out_completions_per_assignment": held_completions,
            "training_completions_per_assignment": train_completions,
            "training_support_pairs": assignments * train_completions,
        })

    full_universe_bits = (universe - 1).bit_length()
    held_out_support_bits = (held_out - 1).bit_length()
    full_universe_bytes = (full_universe_bits + 7) // 8
    held_out_support_bytes = (held_out_support_bits + 7) // 8

    return {
        "schema": SCHEMA,
        "dimensions": dimensions,
        "value_count_per_dimension": value_count,
        "universe_size": universe,
        "held_out_size": held_out,
        "training_size": universe - held_out,
        "held_out_fraction": held_out / universe,
        "held_out_fraction_exact": {"numerator": 1, "denominator": value_count},
        "proper_partial_orders": partial_orders,
        "balanced_no_message_accuracy": 1 / candidate_count,
        "balanced_no_message_accuracy_exact": {"numerator": 1, "denominator": candidate_count},
        "candidate_count": candidate_count,
        "one_way_zero_error_payload_lower_bound_bits": full_universe_bits,
        "rank_code_achieves_lower_bound_bits": True,
        "held_out_support_fixed_width_zero_error_payload_lower_bound_bits": held_out_support_bits,
        "held_out_support_rank_code_payload_bytes": held_out_support_bytes,
        "full_universe_rank_code_payload_bytes": full_universe_bytes,
        "held_out_rank_code_vs_full_universe_ideal_bit_saving": (
            full_universe_bits - held_out_support_bits
        ),
        "held_out_rank_code_vs_full_universe_serialized_byte_saving": (
            full_universe_bytes - held_out_support_bytes
        ),
        "assumptions": [
            "equal value cardinality V on every one of d categorical axes",
            "hold out a tuple when the sum of its independently permuted axis ranks is 0 modulo V",
            "uniform target among k candidates with candidate order independent of the target",
            "fixed-length worst-case payload, noiseless one-way communication, and sender does not observe the candidate table for the payload lower bound",
            "held-out-support payload bound additionally assumes the target is guaranteed held out and both endpoints share the public split and decoder",
        ],
        "interpretation_limit": "These are exact task-geometry and ideal payload predictions, not LLM performance or language-efficiency evidence; rounded payload bytes exclude framing and shared decoder setup.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--values", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=4)
    args = parser.parse_args()
    try:
        report = scaling_report(
            dimensions=args.dimensions, value_count=args.values, candidate_count=args.candidates,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
