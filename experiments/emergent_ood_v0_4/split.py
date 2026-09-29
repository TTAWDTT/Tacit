"""Generate a higher-order compositional split for Emergent OOD studies.

This is a model-free split/schema fixture. It is not a model runner or a
benchmark claim. A meaning is held out when the sum of its independently
permuted per-axis value ranks is zero modulo the value count. Consequently,
every partial assignment that leaves at least one axis unspecified occurs in
both train and held-out meanings, while the complete held-out tuple is absent
from train.
"""
from __future__ import annotations

import argparse
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ATTRIBUTES = ("shape", "color", "quantity", "texture")
DEFAULT_VALUES = (
    ("circle", "square", "triangle", "hexagon"),
    ("red", "blue", "green", "yellow"),
    ("one", "two", "three", "four"),
    ("smooth", "rough", "striped", "dotted"),
)


def _validate_names(
    attributes: Sequence[str], values: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    if not isinstance(attributes, Sequence) or isinstance(attributes, (str, bytes)):
        raise ValueError("attributes must be a sequence of names")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("values must contain one sequence per attribute")
    if any(not isinstance(axis, Sequence) or isinstance(axis, (str, bytes)) for axis in values):
        raise ValueError("each value axis must be a sequence of labels")
    attrs = tuple(attributes)
    axes = tuple(tuple(axis) for axis in values)
    if len(attrs) < 2 or len(attrs) != len(axes):
        raise ValueError("provide at least two attributes and one value axis per attribute")
    if any(not isinstance(item, str) or not item.strip() for item in attrs):
        raise ValueError("attribute names must be non-empty and unique")
    if len(set(attrs)) != len(attrs):
        raise ValueError("attribute names must be non-empty and unique")
    value_count = len(axes[0])
    if value_count < 2 or any(len(axis) != value_count for axis in axes):
        raise ValueError("each attribute must have the same value count, at least two")
    for axis in axes:
        if any(not isinstance(item, str) or not item.strip() for item in axis):
            raise ValueError("values on each attribute axis must be non-empty and unique")
        if len(set(axis)) != value_count:
            raise ValueError("values on each attribute axis must be non-empty and unique")
    return attrs, axes


def _axis_permutation(seed: int, axis_index: int, value_count: int) -> tuple[int, ...]:
    """A stable domain-separated SHA-256 pseudorandom ordering."""
    prefix = f"tlu.emergent-ood-v0.4\0{seed}\0{axis_index}\0".encode("ascii")
    return tuple(
        sorted(
            range(value_count),
            key=lambda value: hashlib.sha256(prefix + str(value).encode("ascii")).digest(),
        )
    )


def _meaning_id(values: Sequence[str]) -> str:
    canonical = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return "m-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_split(
    *, seed: int = 0,
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    values: Sequence[Sequence[str]] = DEFAULT_VALUES,
    ontology_id: str | None = None,
) -> dict[str, Any]:
    """Build and internally validate a balanced modular higher-order split."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if ontology_id is not None and (not isinstance(ontology_id, str) or not ontology_id.strip()):
        raise ValueError("ontology_id must be a non-empty string when provided")
    attrs, axes = _validate_names(attributes, values)
    if ontology_id is None and (attrs != DEFAULT_ATTRIBUTES or axes != DEFAULT_VALUES):
        ontology_payload = json.dumps(
            {"attributes": attrs, "values_by_attribute": dict(zip(attrs, axes))},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        ontology_id = "ontology-" + hashlib.sha256(ontology_payload.encode("utf-8")).hexdigest()[:12]
    value_count = len(axes[0])
    permutations = tuple(
        _axis_permutation(seed, axis_index, value_count)
        for axis_index in range(len(attrs))
    )

    meanings: list[dict[str, Any]] = []
    train: list[str] = []
    held_out: list[str] = []
    for indices in _cartesian_indices(value_count, len(attrs)):
        labels = tuple(permutations[i][index] for i, index in enumerate(indices))
        meaning_values = tuple(axes[i][index] for i, index in enumerate(indices))
        meaning_id = _meaning_id(meaning_values)
        is_held_out = sum(labels) % value_count == 0
        meanings.append(
            {
                "meaning_id": meaning_id,
                "values": list(meaning_values),
                "split": "held_out" if is_held_out else "train",
            }
        )
        (held_out if is_held_out else train).append(meaning_id)

    result: dict[str, Any] = {
        "schema": "tlu.emergent-ood-split.v0.4",
        "generator": "modular-higher-order-v1",
        "seed": seed,
        "attributes": list(attrs),
        "values_by_attribute": {
            attribute: list(axis) for attribute, axis in zip(attrs, axes)
        },
        "axis_permutations": [list(permutation) for permutation in permutations],
        "universe_size": value_count ** len(attrs),
        "train_meaning_ids": sorted(train),
        "held_out_meaning_ids": sorted(held_out),
        "meanings": sorted(meanings, key=lambda row: row["meaning_id"]),
        "coverage_by_order": _coverage_counts(
            attrs=attrs, axes=axes, meanings=meanings, train_ids=set(train),
        ),
        "limits": [
            "model-free split generator only; no sender, receiver, or scorer is implemented",
            "held-out means a complete attribute tuple; individual values and every proper partial assignment remain represented in training",
            "split seed is not a secret and must not be used as a task key for tool-enabled agents",
            "no protocol performance, language emergence, or general benchmark claim is established",
        ],
    }
    if ontology_id is not None:
        result["ontology_id"] = ontology_id
    result["split_sha256"] = _split_digest(result)
    validate_split(result)
    return result


def build_split_from_spec(*, seed: int, spec: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a split from hash-bound manifest metadata or a validated spec."""
    if not isinstance(spec, dict):
        raise ValueError("split spec must be an object")
    allowed = {"seed", "attributes", "values_by_attribute", "ontology_id"}
    required = {"seed", "attributes", "values_by_attribute"}
    if set(spec) - allowed or not required.issubset(spec):
        raise ValueError("split spec has missing or unexpected fields")
    if isinstance(spec["seed"], bool) or not isinstance(spec["seed"], int) or spec["seed"] != seed:
        raise ValueError("split spec seed does not match the requested split seed")
    attributes = spec["attributes"]
    values_by_attribute = spec["values_by_attribute"]
    if not isinstance(attributes, list) or not isinstance(values_by_attribute, dict):
        raise ValueError("split spec attributes/value map are malformed")
    if any(not isinstance(name, str) or not name.strip() for name in attributes):
        raise ValueError("split spec attribute names must be non-empty strings")
    if len(set(attributes)) != len(attributes):
        raise ValueError("split spec attribute names must be unique")
    if set(values_by_attribute) != set(attributes):
        raise ValueError("split spec value axes do not match attributes")
    checked_attributes, axes = _validate_names(
        attributes, [values_by_attribute[name] for name in attributes]
    )
    return build_split(
        seed=seed,
        attributes=checked_attributes,
        values=axes,
        ontology_id=spec.get("ontology_id"),
    )


def split_task_id(split: dict[str, Any]) -> str:
    """Stable task stratum identity, retaining the legacy ID for the default fixture."""
    ontology_id = split.get("ontology_id")
    if ontology_id is None:
        return "four-attribute-higher-order-meaning-matching-v1"
    return f"{len(split['attributes'])}-attribute-higher-order-meaning-matching-v1:{ontology_id}"


def _cartesian_indices(value_count: int, dimensions: int):
    if dimensions == 1:
        for value in range(value_count):
            yield (value,)
        return
    for prefix in _cartesian_indices(value_count, dimensions - 1):
        for value in range(value_count):
            yield (*prefix, value)


def _coverage_counts(
    *, attrs: Sequence[str], axes: Sequence[Sequence[str]],
    meanings: Sequence[dict[str, Any]], train_ids: set[str],
) -> dict[str, int]:
    train = [row for row in meanings if row["meaning_id"] in train_ids]
    counts: dict[str, int] = {}
    for order in range(1, len(attrs)):
        expected = 0
        for index_set in _axis_subsets(len(attrs), order):
            partials = {
                tuple(row["values"][i] for i in index_set)
                for row in train
            }
            axis_expected = sum(1 for _ in _cartesian_values(axes, index_set))
            if len(partials) != axis_expected:
                raise AssertionError(
                    f"training split covers {len(partials)} assignments on axes "
                    f"{index_set}; expected {axis_expected}"
                )
            expected += axis_expected
        counts[str(order)] = expected
    return counts


def _axis_subsets(dimensions: int, order: int):
    from itertools import combinations

    return combinations(range(dimensions), order)


def _cartesian_values(axes: Sequence[Sequence[str]], index_set: Sequence[int]):
    if len(index_set) == 1:
        for value in axes[index_set[0]]:
            yield (value,)
        return
    for prefix in _cartesian_values(axes, index_set[:-1]):
        for value in axes[index_set[-1]]:
            yield (*prefix, value)


def _split_digest(result: dict[str, Any]) -> str:
    digestible = {key: value for key, value in result.items() if key != "split_sha256"}
    canonical = json.dumps(
        digestible, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_split(split: dict[str, Any]) -> None:
    """Verify schema, universe partition, modular rule, and lower-order coverage."""
    if not isinstance(split, dict) or split.get("schema") != "tlu.emergent-ood-split.v0.4":
        raise ValueError("unsupported split schema")
    attrs = split.get("attributes")
    value_map = split.get("values_by_attribute")
    if not isinstance(attrs, list) or not isinstance(value_map, dict):
        raise ValueError("split attributes/value map are missing")
    try:
        checked_attrs, axes = _validate_names(attrs, [value_map[name] for name in attrs])
    except (KeyError, TypeError) as exc:
        raise ValueError("split attribute values are malformed") from exc
    seed = split.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed is invalid")
    if "ontology_id" in split and (
        not isinstance(split["ontology_id"], str) or not split["ontology_id"].strip()
    ):
        raise ValueError("ontology_id must be a non-empty string")
    expected_permutations = [
        list(_axis_permutation(seed, i, len(axes[0]))) for i in range(len(checked_attrs))
    ]
    if split.get("axis_permutations") != expected_permutations:
        raise ValueError("axis permutations do not match seed and generator")

    meanings = split.get("meanings")
    if not isinstance(meanings, list) or len(meanings) != len(axes[0]) ** len(attrs):
        raise ValueError("meaning universe has the wrong size")
    ids = [row.get("meaning_id") for row in meanings if isinstance(row, dict)]
    if len(ids) != len(meanings) or len(set(ids)) != len(ids):
        raise ValueError("meaning IDs must be present and unique")
    rows_by_id = {row["meaning_id"]: row for row in meanings}
    train_ids = split.get("train_meaning_ids")
    held_ids = split.get("held_out_meaning_ids")
    if not isinstance(train_ids, list) or not isinstance(held_ids, list):
        raise ValueError("train/held-out IDs must be lists")
    if (
        any(not isinstance(item, str) for item in (*train_ids, *held_ids))
        or len(train_ids) != len(set(train_ids))
        or len(held_ids) != len(set(held_ids))
    ):
        raise ValueError("train/held-out IDs must be unique strings")
    if set(train_ids) & set(held_ids) or set(train_ids) | set(held_ids) != set(ids):
        raise ValueError("train and held-out IDs must form an exact partition")

    value_to_index = [
        {value: index for index, value in enumerate(axis)} for axis in axes
    ]
    permutation = expected_permutations
    for meaning_id, row in rows_by_id.items():
        values = row.get("values")
        if not isinstance(values, list) or len(values) != len(attrs):
            raise ValueError("each meaning must have one value per attribute")
        try:
            indices = [value_to_index[i][value] for i, value in enumerate(values)]
        except (KeyError, TypeError) as exc:
            raise ValueError("meaning contains an unknown value") from exc
        expected_id = _meaning_id(values)
        if meaning_id != expected_id:
            raise ValueError("meaning ID does not match its attribute values")
        expected_split = (
            "held_out"
            if sum(permutation[i][index] for i, index in enumerate(indices)) % len(axes[0]) == 0
            else "train"
        )
        if row.get("split") != expected_split:
            raise ValueError("meaning split label violates the modular rule")
        if (meaning_id in train_ids) != (expected_split == "train"):
            raise ValueError("train/held-out ID lists disagree with meaning labels")

    recomputed_coverage = _coverage_counts(
        attrs=checked_attrs, axes=axes, meanings=meanings, train_ids=set(train_ids),
    )
    if split.get("coverage_by_order") != recomputed_coverage:
        raise ValueError("coverage summary disagrees with train meanings")
    if split.get("universe_size") != len(meanings):
        raise ValueError("universe_size is incorrect")
    if split.get("split_sha256") != _split_digest(split):
        raise ValueError("split digest mismatch")


def load_ontology_spec(path: Path) -> dict[str, Any]:
    """Load a small, self-contained ontology spec from inside the project."""
    root = Path(__file__).resolve().parents[2]
    target = path if path.is_absolute() else root / path
    target = target.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("ontology spec must resolve inside the project directory") from exc
    try:
        spec = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("ontology spec is missing or invalid JSON") from exc
    required = {"schema", "ontology_id", "attributes", "values_by_attribute"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("ontology spec must contain exactly schema, ontology_id, attributes, and values_by_attribute")
    if spec["schema"] != "tlu.emergent-ood-ontology.v1":
        raise ValueError("unsupported ontology spec schema")
    if not isinstance(spec["attributes"], list) or not isinstance(spec["values_by_attribute"], dict):
        raise ValueError("ontology attributes and values_by_attribute must be a list and object")
    if any(not isinstance(name, str) or not name.strip() for name in spec["attributes"]):
        raise ValueError("ontology attribute names must be non-empty strings")
    if len(set(spec["attributes"])) != len(spec["attributes"]):
        raise ValueError("ontology attribute names must be unique")
    if set(spec["values_by_attribute"]) != set(spec["attributes"]):
        raise ValueError("ontology value axes must match the declared attributes")
    # Validate names before set/dict membership to reject malformed JSON types cleanly.
    _validate_names(
        spec["attributes"], [spec["values_by_attribute"].get(name) for name in spec["attributes"]]
    )
    if not isinstance(spec["ontology_id"], str) or not spec["ontology_id"].strip():
        raise ValueError("ontology_id must be a non-empty string")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ontology", type=Path, help="project-local tlu.emergent-ood-ontology.v1 JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        ontology = load_ontology_spec(args.ontology) if args.ontology else None
        split = build_split(
            seed=args.seed,
            attributes=ontology["attributes"] if ontology else DEFAULT_ATTRIBUTES,
            values=[ontology["values_by_attribute"][name] for name in ontology["attributes"]]
            if ontology else DEFAULT_VALUES,
            ontology_id=ontology["ontology_id"] if ontology else None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    encoded = json.dumps(split, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        project_root = Path(__file__).resolve().parents[2]
        target = args.output if args.output.is_absolute() else project_root / args.output
        target = target.resolve()
        try:
            target.relative_to(project_root)
        except ValueError:
            parser.error("--output must resolve inside the project directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
        print(f"Wrote {target} ({split['split_sha256']})")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
