"""Exact expected-length prefix-code oracle for Private Match v0.3.

This is a task-specific functional-compression baseline, not an LLM language.
For each source, a deterministic partition groups coordinate values that the
receiver will treat as equivalent. Huffman coding minimizes expected prefix
length for that partition. Enumerating all integer partitions of q gives the
exact one-shot expected-payload frontier for q <= 32.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import heapq
import json
from typing import Iterator


MAX_ENUMERATED_Q = 32


def _integer_partitions(total: int, parts: int, ceiling: int | None = None) -> Iterator[tuple[int, ...]]:
    """Yield non-increasing positive integer partitions into exactly parts."""
    if parts == 0:
        if total == 0:
            yield ()
        return
    if parts == 1:
        if 1 <= total <= (total if ceiling is None else ceiling):
            yield (total,)
        return
    largest = min(total - parts + 1, total if ceiling is None else ceiling)
    smallest = (total + parts - 1) // parts
    for first in range(largest, smallest - 1, -1):
        for tail in _integer_partitions(total - first, parts - 1, first):
            yield (first, *tail)


def _huffman_codewords(weights: tuple[int, ...]) -> tuple[str, ...]:
    """Return a deterministic Huffman code, using integer weights."""
    if len(weights) == 1:
        return ("",)
    heap: list[tuple[int, int, int | tuple[object, object]]] = []
    serial = 0
    for leaf, weight in enumerate(weights):
        heapq.heappush(heap, (weight, serial, leaf))
        serial += 1
    while len(heap) > 1:
        left_weight, _, left = heapq.heappop(heap)
        right_weight, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (left_weight + right_weight, serial, (left, right)))
        serial += 1

    result = [""] * len(weights)

    def visit(node: int | tuple[object, object], prefix: str) -> None:
        if isinstance(node, int):
            result[node] = prefix
            return
        left, right = node
        visit(left, prefix + "0")  # type: ignore[arg-type]
        visit(right, prefix + "1")  # type: ignore[arg-type]

    visit(heap[0][2], "")
    return tuple(result)


@dataclass(frozen=True)
class PrefixPartitionCode:
    """A prefix-code partition for one uniform coordinate source."""

    q: int
    classes: tuple[tuple[int, ...], ...]
    codewords: tuple[str, ...]

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def expected_bits(self) -> Fraction:
        return Fraction(
            sum(len(values) * len(codeword)
                for values, codeword in zip(self.classes, self.codewords)),
            self.q,
        )

    @property
    def worst_case_bits(self) -> int:
        return max(map(len, self.codewords))

    @property
    def representatives(self) -> tuple[int, ...]:
        return tuple(values[0] for values in self.classes)

    def encode(self, value: int) -> str:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.q:
            raise ValueError(f"coordinate value must be an integer in 0..{self.q - 1}")
        for values, codeword in zip(self.classes, self.codewords):
            if value in values:
                return codeword
        raise AssertionError("codebook does not cover its source alphabet")

    def decode_representative(self, codeword: str) -> int:
        for values, encoded in zip(self.classes, self.codewords):
            if encoded == codeword:
                return values[0]
        raise ValueError("message is not a codeword in this codebook")


def optimal_prefix_partition(q: int, class_count: int) -> PrefixPartitionCode:
    """Find the minimum expected prefix length using exactly class_count classes."""
    _validate_q(q)
    if isinstance(class_count, bool) or not isinstance(class_count, int) or not 1 <= class_count <= q:
        raise ValueError("class_count must be an integer in 1..q")

    best_key: tuple[int, tuple[int, ...]] | None = None
    best: PrefixPartitionCode | None = None
    for sizes in _integer_partitions(q, class_count):
        codes = _huffman_codewords(sizes)
        weighted_length = sum(size * len(code) for size, code in zip(sizes, codes))
        key = weighted_length, sizes
        if best_key is None or key < best_key:
            start = 0
            classes = []
            for size in sizes:
                classes.append(tuple(range(start, start + size)))
                start += size
            best_key = key
            best = PrefixPartitionCode(q, tuple(classes), codes)
    if best is None:
        raise AssertionError("integer partition enumeration returned no candidate")
    return best


def prefix_code_frontier(q: int) -> dict[str, object]:
    """Return deterministic expected-payload/success points for two senders."""
    _validate_q(q)
    codes = [optimal_prefix_partition(q, count) for count in range(1, q + 1)]
    by_cost_and_success: dict[tuple[Fraction, Fraction], list[tuple[int, int]]] = {}
    for left in codes:
        for right in codes:
            expected_bits = left.expected_bits + right.expected_bits
            success = Fraction(left.class_count * right.class_count, q * q)
            by_cost_and_success.setdefault((expected_bits, success), []).append(
                (left.class_count, right.class_count)
            )

    frontier_keys = [
        key for key in by_cost_and_success
        if not any(
            other_cost <= key[0] and other_success >= key[1]
            and (other_cost < key[0] or other_success > key[1])
            for other_cost, other_success in by_cost_and_success
        )
    ]
    frontier_keys.sort(key=lambda item: (item[0], item[1]))
    points = []
    for expected_bits, success in frontier_keys:
        allocations = sorted(by_cost_and_success[(expected_bits, success)])
        points.append({
            "expected_payload_bits": _fraction_text(expected_bits),
            "joint_success": _fraction_text(success),
            "sender_class_allocations": [
                {
                    "sender_x_classes": left,
                    "sender_y_classes": right,
                    "worst_case_payload_bits": (
                        codes[left - 1].worst_case_bits
                        + codes[right - 1].worst_case_bits
                    ),
                }
                for left, right in allocations
            ],
        })

    randomized_vertices = _upper_concave_hull(points)
    randomized_segments = []
    for left, right in zip(randomized_vertices, randomized_vertices[1:]):
        randomized_segments.append({
            "lower_expected_payload_bits": left["expected_payload_bits"],
            "lower_joint_success": left["joint_success"],
            "upper_expected_payload_bits": right["expected_payload_bits"],
            "upper_joint_success": right["joint_success"],
            "higher_cost_protocol_probability": (
                f"(B - {left['expected_payload_bits']}) / "
                f"({right['expected_payload_bits']} - {left['expected_payload_bits']})"
            ),
            "lower_cost_allocations": left["sender_class_allocations"],
            "higher_cost_allocations": right["sender_class_allocations"],
        })

    class_costs = [
        {
            "class_count": code.class_count,
            "expected_bits": _fraction_text(code.expected_bits),
            "worst_case_bits": code.worst_case_bits,
            "class_sizes": [len(values) for values in code.classes],
            "codewords": list(code.codewords),
            "representatives": list(code.representatives),
        }
        for code in codes
    ]
    fixed_width_reference = None
    if q & (q - 1) == 0:
        width = q.bit_length() - 1
        fixed_width_reference = [
            {
                "total_worst_case_bits": budget,
                "joint_success": _fraction_text(Fraction(2 ** min(budget, 2 * width), q * q)),
            }
            for budget in range(2 * width + 1)
        ]
    return {
        "schema_version": "tlu.private-match-prefix-frontier.v1",
        "q": q,
        "task_prior": "independent uniform coordinates",
        "encoder_family": "deterministic one-shot sender partitions",
        "objective": "exact receiver-row success under a sum expected prefix-payload bit budget",
        "class_costs": class_costs,
        "expected_length_pareto_frontier": points,
        "free_shared_randomness_upper_bound": {
            "vertices": randomized_vertices,
            "mixing_segments": randomized_segments,
            "assumption": "a free common coin selects a complete deterministic protocol before each episode; seed agreement, setup, and per-episode tail constraints are excluded",
        },
        "fixed_width_integer_budget_reference": fixed_width_reference,
        "limits": [
            "payload-only prefix lengths; message framing, prompt/codebook exposition, model tokens, and inference cost excluded",
            "expected-bit budgets are not per-episode hard caps; worst-case lengths are reported separately",
            "frontier does not convexify by randomizing across codebooks or episodes",
            "the shared task-specific encoder/decoder is a functional-compression oracle, not an LLM language or a superiority result",
        ],
    }


def randomized_success_at_expected_budget(q: int, expected_budget: Fraction | int | str) -> Fraction:
    """Interpolate the free-shared-randomness upper bound at an expected rate."""
    _validate_q(q)
    budget = Fraction(expected_budget)
    report = prefix_code_frontier(q)
    vertices = report["free_shared_randomness_upper_bound"]["vertices"]
    costs = [Fraction(point["expected_payload_bits"]) for point in vertices]
    if not costs[0] <= budget <= costs[-1]:
        raise ValueError(f"expected_budget must be in {_fraction_text(costs[0])}..{_fraction_text(costs[-1])}")
    for left, right in zip(vertices, vertices[1:]):
        left_cost = Fraction(left["expected_payload_bits"])
        right_cost = Fraction(right["expected_payload_bits"])
        if left_cost <= budget <= right_cost:
            left_success = Fraction(left["joint_success"])
            right_success = Fraction(right["joint_success"])
            return left_success + (budget - left_cost) * (right_success - left_success) / (right_cost - left_cost)
    return Fraction(vertices[-1]["joint_success"])


def _upper_concave_hull(points: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the upper concave hull of cost/success points, preserving endpoints."""
    hull: list[dict[str, object]] = []
    for point in points:
        x = Fraction(point["expected_payload_bits"])
        y = Fraction(point["joint_success"])
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            x0, y0 = Fraction(first["expected_payload_bits"]), Fraction(first["joint_success"])
            x1, y1 = Fraction(second["expected_payload_bits"]), Fraction(second["joint_success"])
            if (y1 - y0) / (x1 - x0) > (y - y1) / (x - x1):
                break
            hull.pop()
        hull.append(point)
    return hull


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _validate_q(q: int) -> None:
    if isinstance(q, bool) or not isinstance(q, int) or not 2 <= q <= MAX_ENUMERATED_Q:
        raise ValueError(f"q must be an integer in 2..{MAX_ENUMERATED_Q}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, default=4, help=f"uniform coordinate alphabet size (2..{MAX_ENUMERATED_Q})")
    parser.add_argument("--output", help="write JSON to this path (default: stdout)")
    args = parser.parse_args()
    try:
        result = prefix_code_frontier(args.q)
    except ValueError as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8", newline="\n")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
