"""Generate compositional train/test splits for a 3 x 3 x 3 meaning space."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


AXES = ("shape", "color", "quantity")
VALUES = (0, 1, 2)


def generate_split(seed: int) -> dict[str, Any]:
    """Return an 18/9 split where all unary and pairwise combinations are trained.

    The held-out set is a permuted Latin square: each value on each axis occurs
    three times among held-out meanings, while every pair of values occurs in
    the training set. Only the third-order combinations are novel.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    permutations = []
    for axis in AXES:
        values = sorted(
            VALUES,
            key=lambda value: hashlib.sha256(f"{seed}|{axis}|{value}".encode("ascii")).digest(),
        )
        permutations.append(values)

    train = []
    test = []
    for meaning in itertools.product(VALUES, repeat=len(AXES)):
        permuted = tuple(permutations[axis][value] for axis, value in enumerate(meaning))
        row = dict(zip(AXES, meaning))
        (test if sum(permuted) % 3 == 0 else train).append(row)

    return {
        "schema": "tlu.emergent_ood_split.v1",
        "seed": seed,
        "axes": list(AXES),
        "domain_values": list(VALUES),
        "permutations": {axis: permutations[i] for i, axis in enumerate(AXES)},
        "split_rule": "held_out iff sum(permuted_axis_values) mod 3 == 0",
        "train_meanings": train,
        "held_out_meanings": test,
        "counts": {"universe": 27, "train": len(train), "held_out": len(test)},
        "claims": {
            "all_single_attribute_values_seen_in_train": True,
            "all_pairwise_value_combinations_seen_in_train": True,
            "held_out_receiver_utility_measured": False,
        },
    }


def verify_split(split: dict[str, Any]) -> None:
    """Raise ValueError unless the artifact satisfies its advertised invariants."""
    train = split.get("train_meanings")
    held_out = split.get("held_out_meanings")
    if not isinstance(train, list) or not isinstance(held_out, list):
        raise ValueError("train_meanings and held_out_meanings must be lists")

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        if not isinstance(row, dict) or set(row) != set(AXES):
            raise ValueError("each meaning must contain exactly the declared axes")
        values = tuple(row[axis] for axis in AXES)
        if any(value not in VALUES for value in values):
            raise ValueError("meaning contains a value outside the declared domain")
        return values

    train_keys = [key(row) for row in train]
    held_out_keys = [key(row) for row in held_out]
    universe = set(itertools.product(VALUES, repeat=len(AXES)))
    if len(train_keys) != len(set(train_keys)) or len(held_out_keys) != len(set(held_out_keys)):
        raise ValueError("duplicate meaning")
    if set(train_keys) & set(held_out_keys) or set(train_keys) | set(held_out_keys) != universe:
        raise ValueError("train/test must be disjoint and cover the full meaning space")
    if len(train_keys) != 18 or len(held_out_keys) != 9:
        raise ValueError("expected an 18/9 train/held-out split")

    train_set = set(train_keys)
    for axis in range(3):
        for value in VALUES:
            if not any(row[axis] == value for row in train_set):
                raise ValueError("a unary value is absent from training")
    for first, second in itertools.combinations(range(3), 2):
        for left in VALUES:
            for right in VALUES:
                if not any(row[first] == left and row[second] == right for row in train_set):
                    raise ValueError("a pairwise combination is absent from training")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = generate_split(args.seed)
    verify_split(split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
