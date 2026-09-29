"""Exact bit-length references for finite zero-error target codes.

These are source-code length references only. They do not include a real
transport's packet delimiter, envelope, codebook setup, model-token costs, or
LLM decoding compute.
"""
from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class CodeLengthSummary:
    kind: str
    lengths: tuple[int, ...]
    mean_bits: Fraction
    worst_case_bits: int
    self_delimiting: bool
    transport_framing_cost_included: bool


def _validate_symbol_count(symbol_count: int) -> None:
    if isinstance(symbol_count, bool) or not isinstance(symbol_count, int) or symbol_count < 2:
        raise ValueError("symbol_count must be an integer of at least two")


def fixed_width_lengths(symbol_count: int) -> tuple[int, ...]:
    """Return the ideal equal fixed-width length for each target symbol."""
    _validate_symbol_count(symbol_count)
    width = (symbol_count - 1).bit_length()
    return (width,) * symbol_count


def huffman_prefix_lengths(symbol_count: int) -> tuple[int, ...]:
    """Return deterministic optimal Huffman lengths for a uniform source."""
    _validate_symbol_count(symbol_count)
    serial = 0
    heap: list[tuple[int, int, Any]] = []
    for symbol in range(symbol_count):
        heapq.heappush(heap, (1, serial, symbol))
        serial += 1
    while len(heap) > 1:
        left_weight, _, left = heapq.heappop(heap)
        right_weight, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (left_weight + right_weight, serial, (left, right)))
        serial += 1

    lengths = [0] * symbol_count
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            lengths[node] = depth
        else:
            left, right = node
            stack.append((right, depth + 1))
            stack.append((left, depth + 1))
    return tuple(lengths)


def framed_payload_codewords(symbol_count: int) -> tuple[str, ...]:
    """Return shortest distinct non-empty strings, assuming an external frame.

    Prefix-related strings are allowed because the packet boundary reveals the
    end of this one message. The boundary's real serialization cost is omitted.
    """
    _validate_symbol_count(symbol_count)
    words: list[str] = []
    width = 1
    while len(words) < symbol_count:
        for value in range(1 << width):
            words.append(f"{value:0{width}b}")
            if len(words) == symbol_count:
                break
        width += 1
    return tuple(words)


def summarize_uniform_codes(symbol_count: int) -> tuple[CodeLengthSummary, ...]:
    """Compare three ideal code-length models for equiprobable target symbols."""
    fixed = fixed_width_lengths(symbol_count)
    prefix = huffman_prefix_lengths(symbol_count)
    framed_words = framed_payload_codewords(symbol_count)
    framed = tuple(map(len, framed_words))

    def summary(kind: str, lengths: tuple[int, ...], self_delimiting: bool) -> CodeLengthSummary:
        return CodeLengthSummary(
            kind=kind,
            lengths=lengths,
            mean_bits=Fraction(sum(lengths), len(lengths)),
            worst_case_bits=max(lengths),
            self_delimiting=self_delimiting,
            transport_framing_cost_included=False,
        )

    return (
        summary("fixed_width", fixed, True),
        summary("prefix_free_huffman", prefix, True),
        summary("externally_framed_payload_only", framed, False),
    )


def as_dicts(symbol_count: int) -> list[dict[str, Any]]:
    """Serialize references without losing exact rational means."""
    return [
        {
            "symbol_count": symbol_count,
            "kind": row.kind,
            "codeword_lengths_bits": list(row.lengths),
            "mean_payload_bits": f"{row.mean_bits.numerator}/{row.mean_bits.denominator}",
            "worst_case_payload_bits": row.worst_case_bits,
            "self_delimiting": row.self_delimiting,
            "transport_framing_cost_included": row.transport_framing_cost_included,
            "claim_scope": "ideal payload length only; no transport or LLM cost",
        }
        for row in summarize_uniform_codes(symbol_count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=9)
    args = parser.parse_args()
    try:
        result = as_dicts(args.symbols)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
