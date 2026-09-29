"""Exact ideal bit-budget frontier for triadic Private Match v0.3.

This is a model-free reference for two simultaneous, independent, fixed-width
messages over a complete q-by-q candidate table. It excludes message framing,
prompts, setup, inference, and model errors.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from collections.abc import Sequence


def _validate_q(q: int) -> None:
    if isinstance(q, bool) or not isinstance(q, int) or q < 2 or q & (q - 1):
        raise ValueError("q must be an integer power of two >= 2")


def optimal_success_probability(*, q: int, total_payload_bits: int) -> Fraction:
    """Maximum Bayes exact-match accuracy under a total fixed-width bit cap.

    The fixed schedule identifies which source sent each payload. Each source
    gets an integer number of payload bits, and unused budget is allowed.
    """
    _validate_q(q)
    if (isinstance(total_payload_bits, bool) or not isinstance(total_payload_bits, int)
            or total_payload_bits < 0):
        raise ValueError("total_payload_bits must be a non-negative integer")
    useful_bits = min(total_payload_bits, 2 * (q.bit_length() - 1))
    return Fraction(1 << useful_bits, q * q)


def _probability_vector(values: Sequence[int | Fraction], *, name: str) -> tuple[Fraction, ...]:
    if not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a non-empty probability sequence")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in values):
        raise ValueError(f"{name} entries must be exact integers or fractions")
    probabilities = tuple(Fraction(value) for value in values)
    if any(value < 0 for value in probabilities) or sum(probabilities, Fraction(0)) != 1:
        raise ValueError(f"{name} must contain non-negative probabilities that sum exactly to 1")
    return probabilities


@dataclass(frozen=True)
class FixedWidthCodebook:
    """A source-index-to-symbol encoder and its frozen MAP decoder."""

    payload_bits: int
    symbols: tuple[int, ...]
    representatives: tuple[int, ...]

    def __post_init__(self) -> None:
        if (isinstance(self.payload_bits, bool) or not isinstance(self.payload_bits, int)
                or self.payload_bits < 0):
            raise ValueError("payload_bits must be a non-negative integer")
        if not isinstance(self.symbols, tuple) or not isinstance(self.representatives, tuple):
            raise ValueError("codebook symbols and representatives must be immutable tuples")
        if not self.symbols or not self.representatives:
            raise ValueError("codebook support and representatives must be non-empty")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.symbols):
            raise ValueError("code symbols must be integers")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in self.representatives):
            raise ValueError("representatives must be integer source indices")
        if max(self.symbols) >= len(self.representatives) or min(self.symbols) < 0:
            raise ValueError("code symbols must refer to declared representatives")
        if set(self.symbols) != set(range(len(self.representatives))):
            raise ValueError("every declared code symbol must be used")
        if any(not 0 <= index < len(self.symbols)
               or self.symbols[index] != symbol
               for symbol, index in enumerate(self.representatives)):
            raise ValueError("each representative must belong to its decoded class")
        if (len(self.representatives) - 1).bit_length() > self.payload_bits:
            raise ValueError("payload_bits cannot represent all code symbols")

    def encode(self, value_index: int) -> str:
        if isinstance(value_index, bool) or not isinstance(value_index, int):
            raise ValueError("value_index must be an integer")
        if not 0 <= value_index < len(self.symbols):
            raise ValueError("value_index is outside the codebook support")
        if self.payload_bits == 0:
            return ""
        return format(self.symbols[value_index], f"0{self.payload_bits}b")

    def decode(self, payload: str) -> int:
        if not isinstance(payload, str) or len(payload) != self.payload_bits:
            raise ValueError("payload must contain exactly payload_bits binary characters")
        if any(bit not in "01" for bit in payload):
            raise ValueError("payload must contain only binary characters")
        symbol = int(payload, 2) if payload else 0
        if symbol >= len(self.representatives):
            raise ValueError("payload names an unused code symbol")
        return self.representatives[symbol]

    def success_probability(self, probabilities: Sequence[int | Fraction]) -> Fraction:
        """Exact one-coordinate success under a supplied evaluation prior.

        The encoder and decoder remain frozen; only the episode distribution
        changes. This measures transfer without letting the decoder adapt.
        """
        prior = _probability_vector(probabilities, name="probabilities")
        if len(prior) != len(self.symbols):
            raise ValueError("evaluation prior support must match the codebook")
        return sum((probability for index, probability in enumerate(prior)
                    if self.representatives[self.symbols[index]] == index), Fraction(0))


def optimal_nonuniform_codebook(
    *, probabilities: Sequence[int | Fraction], payload_bits: int,
) -> FixedWidthCodebook:
    """Build an exact Bayes-optimal fixed-width encoder for one known prior.

    Ties are resolved by source index. The most probable K-1 values receive
    singleton symbols; all remaining values share the final symbol, where
    K=min(support size, 2**payload_bits). The decoder predicts the most
    probable source value in each class. The returned codebook is frozen and
    can be scored on shifted priors with ``success_probability``.
    """
    prior = _probability_vector(probabilities, name="probabilities")
    if isinstance(payload_bits, bool) or not isinstance(payload_bits, int) or payload_bits < 0:
        raise ValueError("payload_bits must be a non-negative integer")
    support_size = len(prior)
    max_useful_bits = (support_size - 1).bit_length()
    symbol_count = support_size if payload_bits >= max_useful_bits else 1 << payload_bits
    order = sorted(range(support_size), key=lambda index: (-prior[index], index))

    symbols = [symbol_count - 1] * support_size
    representatives = []
    for symbol, index in enumerate(order[:symbol_count - 1]):
        symbols[index] = symbol
        representatives.append(index)
    tail = order[symbol_count - 1:]
    representatives.append(tail[0])
    return FixedWidthCodebook(payload_bits, tuple(symbols), tuple(representatives))


def optimal_nonuniform_success_probability(
    *, probabilities_x: Sequence[int | Fraction],
    probabilities_y: Sequence[int | Fraction], total_payload_bits: int,
) -> Fraction:
    """Exact Bayes frontier for independent, non-uniform coordinate priors.

    Each sender's source is independent, but values within a coordinate may
    have arbitrary exact rational probabilities. Candidate tables are complete
    Cartesian products, and fixed-width noiseless sender slots are known.
    """
    px = sorted(_probability_vector(probabilities_x, name="probabilities_x"), reverse=True)
    py = sorted(_probability_vector(probabilities_y, name="probabilities_y"), reverse=True)
    if (isinstance(total_payload_bits, bool) or not isinstance(total_payload_bits, int)
            or total_payload_bits < 0):
        raise ValueError("total_payload_bits must be a non-negative integer")

    def retained_mass(prior: tuple[Fraction, ...], bits: int) -> Fraction:
        return sum(prior[:min(len(prior), 1 << bits)], Fraction(0))

    # More than ceil(log2(support)) bits cannot create additional classes.
    max_x_bits = (len(px) - 1).bit_length()
    max_y_bits = (len(py) - 1).bit_length()
    useful_budget = min(total_payload_bits, max_x_bits + max_y_bits)
    best = Fraction(0)
    for x_bits in range(min(useful_budget, max_x_bits) + 1):
        for y_bits in range(min(useful_budget - x_bits, max_y_bits) + 1):
            candidate = retained_mass(px, x_bits) * retained_mass(py, y_bits)
            best = max(best, candidate)
    return best


def optimal_allocation(*, q: int, total_payload_bits: int) -> dict[str, int]:
    """Return a balanced optimal allocation, capped at log2(q) per source."""
    _validate_q(q)
    if (isinstance(total_payload_bits, bool) or not isinstance(total_payload_bits, int)
            or total_payload_bits < 0):
        raise ValueError("total_payload_bits must be a non-negative integer")
    width = q.bit_length() - 1
    useful_bits = min(total_payload_bits, 2 * width)
    x_bits = min(width, (useful_bits + 1) // 2)
    y_bits = useful_bits - x_bits
    return {"sender_x_bits": x_bits, "sender_y_bits": y_bits}


def frontier(*, q: int, max_total_payload_bits: int) -> list[dict[str, int | str | float]]:
    _validate_q(q)
    if (isinstance(max_total_payload_bits, bool)
            or not isinstance(max_total_payload_bits, int)
            or max_total_payload_bits < 0):
        raise ValueError("max_total_payload_bits must be a non-negative integer")
    rows = []
    for budget in range(max_total_payload_bits + 1):
        exact = optimal_success_probability(q=q, total_payload_bits=budget)
        allocation = optimal_allocation(q=q, total_payload_bits=budget)
        rows.append({
            "total_payload_bit_budget": budget,
            **allocation,
            "payload_bits_used": sum(allocation.values()),
            "success_fraction": f"{exact.numerator}/{exact.denominator}",
            "success_probability": float(exact),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--max-bits", type=int, default=8)
    args = parser.parse_args()
    try:
        rows = frontier(q=args.q, max_total_payload_bits=args.max_bits)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "task": "complete_cartesian_q_by_q_uniform_hidden_target",
        "q": args.q,
        "candidate_count": args.q * args.q,
        "assumptions": [
            "two simultaneous messages, one from each coordinate source",
            "fixed schedule reveals the sender slot",
            "noiseless fixed-width binary payloads",
            "receiver observes the complete candidate table",
            "uniform target over all q squared rows",
        ],
        "excluded_costs": ["framing", "prompts", "codebook setup", "inference", "model errors"],
        "frontier": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
