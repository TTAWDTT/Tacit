"""Create a deterministic negative-control artifact by shuffling message labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from experiments.emergent_ood_v0_4.episodes import _inside_project


CONTROL_SCHEMA = "tlu.usage-example-control.v1"
ARTIFACT_SCHEMA = "tlu.usage_examples.v1"
MAX_SHUFFLE_ATTEMPTS = 10_000


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def shuffle_usage_examples(*, source_path: Path, output_path: Path, seed: int) -> dict[str, Any]:
    """Return a hash-bound artifact with the same examples but mismatched labels.

    Every output meaning is paired with a different source message string. Source
    acquisition costs are preserved: this transform adds no model calls.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("shuffle seed must be a non-negative integer")
    source_path = _inside_project(source_path)
    output_path = _inside_project(output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".control.json")
    if source_path == output_path or output_path.exists() or manifest_path.exists():
        raise FileExistsError("source and control output must be distinct, and control outputs must be new")

    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if not isinstance(source, dict) or source.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("source must be a tlu.usage_examples.v1 artifact")
    examples = source.get("examples")
    if not isinstance(examples, list) or len(examples) < 2:
        raise ValueError("at least two examples are required for a shuffled-pair control")
    for row in examples:
        if (
            not isinstance(row, dict) or set(row) != {"meaning_id", "meaning", "message"}
            or not isinstance(row["meaning_id"], str)
            or not isinstance(row["meaning"], dict)
            or not isinstance(row["message"], str)
        ):
            raise ValueError("source contains a malformed usage example")

    source_digest = hashlib.sha256(source_bytes).hexdigest()
    rng = random.Random(f"tlu-shuffled-pairs-v1:{seed}:{source_digest}")
    order = list(range(len(examples)))
    for _ in range(MAX_SHUFFLE_ATTEMPTS):
        rng.shuffle(order)
        if all(examples[index]["message"] != examples[source_index]["message"]
               for index, source_index in enumerate(order)):
            break
    else:
        raise ValueError("source messages do not permit a complete mismatched pairing")

    control = dict(source)
    control["examples"] = [
        {**example, "message": examples[source_index]["message"]}
        for example, source_index in zip(examples, order)
    ]
    control_bytes = (json.dumps(control, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(control_bytes)
    temporary.replace(output_path)
    manifest = {
        "schema": CONTROL_SCHEMA,
        "control": "shuffled_meaning_message_pairs",
        "algorithm": "seeded-python-random-shuffle-rejecting-matching-labels-v1",
        "seed": seed,
        "source_artifact_sha256": source_digest,
        "control_artifact_sha256": hashlib.sha256(control_bytes).hexdigest(),
        "meaning_to_source_message_index": order,
        "model_calls_added": 0,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    try:
        manifest = shuffle_usage_examples(source_path=args.source, output_path=args.output, seed=args.seed)
    except (OSError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
