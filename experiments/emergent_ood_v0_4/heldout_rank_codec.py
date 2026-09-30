"""Exact task-support oracle codec for the modular higher-order OOD split.

This is a fixed-width reference codec, not an LLM language and not a learned
protocol. It is valid only when the target is guaranteed to lie in the public
held-out support of the supplied split. Its ideal payload is measured in bits;
the byte serializer and any shared split metadata must be charged separately.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiments.emergent_ood_v0_4.split import validate_split


CODEC_ID = "tlu.heldout-modular-rank.v1"


class HeldOutRankCodec:
    """Rank a held-out tuple using the first d-1 permuted axis ranks.

    For ``d`` axes with ``V`` values, a modular held-out support contains
    exactly ``V**(d-1)`` tuples. The first d-1 ranks therefore form a bijection
    to that support; the final rank is fixed by the zero-sum modular rule.
    """

    def __init__(self, split: dict[str, Any]) -> None:
        validate_split(split)
        self.attributes = tuple(split["attributes"])
        self.values = tuple(tuple(split["values_by_attribute"][a]) for a in self.attributes)
        self.value_count = len(self.values[0])
        self.dimensions = len(self.attributes)
        self.support_size = self.value_count ** (self.dimensions - 1)
        self.bit_width = (self.support_size - 1).bit_length()
        self.payload_bytes = (self.bit_width + 7) // 8
        self.padding_bits = self.payload_bytes * 8 - self.bit_width
        self.split_sha256 = split["split_sha256"]

        value_to_index = tuple(
            {value: index for index, value in enumerate(axis)} for axis in self.values
        )
        permutations = split["axis_permutations"]
        self._rank_by_value = tuple(
            {value: permutations[i][index] for value, index in value_to_index[i].items()}
            for i in range(self.dimensions)
        )
        self._value_by_rank = tuple(
            {rank: self.values[i][index] for index, rank in enumerate(permutations[i])}
            for i in range(self.dimensions)
        )

    def encode(self, meaning: Mapping[str, str]) -> bytes:
        """Encode one held-out meaning into the canonical fixed-width payload."""
        if not isinstance(meaning, Mapping) or set(meaning) != set(self.attributes):
            raise ValueError("meaning must contain exactly the split's attributes")
        try:
            ranks = tuple(self._rank_by_value[i][meaning[a]] for i, a in enumerate(self.attributes))
        except (KeyError, TypeError) as exc:
            raise ValueError("meaning contains an unknown value") from exc
        if sum(ranks) % self.value_count != 0:
            raise ValueError("meaning is outside the split's held-out target support")

        index = 0
        for rank in ranks[:-1]:
            index = index * self.value_count + rank
        packed = index << self.padding_bits
        return packed.to_bytes(self.payload_bytes, "big")

    def decode(self, payload: bytes) -> dict[str, str]:
        """Decode a canonical payload; reject padding and unused codewords."""
        if not isinstance(payload, bytes) or len(payload) != self.payload_bytes:
            raise ValueError(f"payload must be exactly {self.payload_bytes} bytes")
        packed = int.from_bytes(payload, "big")
        if self.padding_bits and packed & ((1 << self.padding_bits) - 1):
            raise ValueError("payload has non-zero canonical padding bits")
        index = packed >> self.padding_bits
        if index >= self.support_size:
            raise ValueError("payload is an unused fixed-width codeword")

        ranks = [0] * (self.dimensions - 1)
        remainder = index
        for position in range(self.dimensions - 2, -1, -1):
            ranks[position] = remainder % self.value_count
            remainder //= self.value_count
        ranks.append((-sum(ranks)) % self.value_count)
        return {
            attribute: self._value_by_rank[axis][rank]
            for axis, (attribute, rank) in enumerate(zip(self.attributes, ranks))
        }
