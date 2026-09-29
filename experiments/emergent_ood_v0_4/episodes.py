"""Generate evaluator-keyed, role-separated Emergent OOD episodes.

This module creates data ledgers only. It never calls a model and makes no
claim about a communication protocol's performance.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

try:  # Direct script and package/module execution.
    from .split import DEFAULT_ATTRIBUTES, DEFAULT_VALUES, build_split, load_ontology_spec, validate_split
except ImportError:  # pragma: no cover - exercised by the CLI entry point
    from split import DEFAULT_ATTRIBUTES, DEFAULT_VALUES, build_split, load_ontology_spec, validate_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_VERSION = "0.4.1"
SCHEMA = f"tlu.emergent-ood-episodes.v{GENERATOR_VERSION}"


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("output and key paths must stay inside the project directory") from exc
    return resolved


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _key(task_key: bytes) -> None:
    if not isinstance(task_key, bytes) or len(task_key) != 32:
        raise ValueError("task key must be exactly 32 secret bytes")


class _Stream:
    """Domain-separated HMAC stream with unbiased randbelow and shuffle."""

    def __init__(self, key: bytes, domain: str, seed: int) -> None:
        _key(key)
        if not domain.isascii() or not domain:
            raise ValueError("domain must be non-empty ASCII")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.key = key
        self.prefix = f"tlu.emergent-ood-episodes.v{GENERATOR_VERSION}/{domain}/{seed}/".encode("ascii")
        self.counter = 0
        self.buffer = bytearray()

    def _read(self, count: int) -> bytes:
        while len(self.buffer) < count:
            self.buffer.extend(hmac.new(self.key, self.prefix + self.counter.to_bytes(8, "big"), hashlib.sha256).digest())
            self.counter += 1
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def randbelow(self, stop: int) -> int:
        if stop < 1:
            raise ValueError("stop must be positive")
        bits = (stop - 1).bit_length()
        while True:
            nbytes = (bits + 7) // 8
            value = int.from_bytes(self._read(nbytes), "big") & ((1 << bits) - 1) if bits else 0
            if value < stop:
                return value

    def shuffle(self, values: list[Any]) -> None:
        for i in range(len(values) - 1, 0, -1):
            j = self.randbelow(i + 1)
            values[i], values[j] = values[j], values[i]


def create_task_key(path: Path, *, force: bool = False) -> str:
    path = _inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to replace existing key: {path}")
    key = secrets.token_bytes(32)
    if force:
        path.write_bytes(key)
    else:
        with path.open("xb") as stream:
            stream.write(key)
    return hashlib.sha256(key).hexdigest()[:16]


def load_task_key(path: Path) -> bytes:
    key = _inside_project(path).read_bytes()
    _key(key)
    return key


def _partition(split: dict[str, Any], key: bytes, seed: int) -> dict[str, list[str]]:
    held = list(split["held_out_meaning_ids"])
    _Stream(key, f"heldout-partition-{split['split_sha256']}", seed).shuffle(held)
    # The held-out ontology has 64 meanings in the default design.
    validation = held[: max(1, len(held) // 4)]
    test = held[len(validation):]
    return {"train": list(split["train_meaning_ids"]), "validation": validation, "test": test}


def _make_sets(
    *, meaning_ids: Sequence[str], k: int,
    set_count: int, key: bytes, seed: int, stage: str, split_sha256: str,
) -> list[tuple[str, list[str], list[str]]]:
    if k < 2 or k > len(meaning_ids):
        raise ValueError(f"k must be between 2 and the {stage} pool size")
    if set_count < 1:
        raise ValueError("set_count must be positive")
    result = []
    for set_index in range(set_count):
        rng = _Stream(key, f"{stage}-candidate-set-{split_sha256}-{set_index}", seed)
        pool = list(meaning_ids)
        rng.shuffle(pool)
        selected = pool[:k]
        # Candidate IDs and row order are fixed across the k balanced targets.
        id_rng = _Stream(key, f"{stage}-candidate-ids-{split_sha256}-{set_index}", seed)
        order = list(range(k))
        id_rng.shuffle(order)
        candidates = [f"c{order[i]:03d}" for i in range(k)]
        set_id = hashlib.sha256(f"{stage}/{split_sha256}/{seed}/{set_index}".encode()).hexdigest()[:16]
        result.append((set_id, selected, candidates))
    return result


def generate_ledgers(
    *, split: dict[str, Any], task_key: bytes, task_seed: int = 0, k: int = 4,
    sets_per_stage: int = 16,
) -> dict[str, Any]:
    """Return separate sender, receiver and evaluator ledgers plus safe metadata."""
    _key(task_key)
    if isinstance(task_seed, bool) or not isinstance(task_seed, int) or task_seed < 0:
        raise ValueError("task_seed must be a non-negative integer")
    validate_split(split)
    partitions = _partition(split, task_key, task_seed)
    meaning_by_id = {row["meaning_id"]: row for row in split["meanings"]}
    sender: dict[str, list[dict[str, Any]]] = {name: [] for name in partitions}
    receiver: dict[str, list[dict[str, Any]]] = {name: [] for name in partitions}
    gold: dict[str, list[dict[str, Any]]] = {name: [] for name in partitions}

    for stage, pool in partitions.items():
        candidate_sets = _make_sets(
            meaning_ids=pool, k=k, set_count=sets_per_stage,
            key=task_key, seed=task_seed, stage=stage, split_sha256=split["split_sha256"],
        )
        for set_index, (set_id, selected, candidate_ids) in enumerate(candidate_sets):
            candidates = [
                {"candidate_id": candidate_ids[i], "attributes": dict(zip(split["attributes"], meaning_by_id[mid]["values"]))}
                for i, mid in enumerate(selected)
            ]
            # All k targets occur exactly once; shuffle target order independently.
            target_order = list(range(k))
            _Stream(task_key, f"{stage}-target-order-{split['split_sha256']}-{set_index}", task_seed).shuffle(target_order)
            for within_set, target_index in enumerate(target_order):
                episode_id = hmac.new(
                    task_key,
                    f"episode/{split['split_sha256']}/{task_seed}/{stage}/{set_id}/{within_set}".encode("ascii"),
                    hashlib.sha256,
                ).hexdigest()[:24]
                target_mid = selected[target_index]
                target_values = meaning_by_id[target_mid]["values"]
                sender[stage].append({
                    "private_meaning": dict(zip(split["attributes"], target_values)),
                })
                # Candidate table order does not depend on which item is the target.
                receiver[stage].append({"candidates": candidates})
                gold[stage].append({
                    "episode_id": episode_id,
                    "candidate_id": candidates[target_index]["candidate_id"],
                    "meaning_id": target_mid,
                    "candidate_set_id": set_id,
                })

    value_to_id = {tuple(row["values"]): row["meaning_id"] for row in split["meanings"]}
    graph_diagnostics = {}
    for stage in partitions:
        candidate_sets_by_id: dict[str, set[str]] = {}
        vertices: set[str] = set()
        for receiver_row, gold_row in zip(receiver[stage], gold[stage]):
            candidate_meanings = {
                value_to_id[tuple(candidate["attributes"][attribute] for attribute in split["attributes"])]
                for candidate in receiver_row["candidates"]
            }
            vertices.update(candidate_meanings)
            candidate_sets_by_id.setdefault(gold_row["candidate_set_id"], candidate_meanings)
        observed_pairs = {
            pair
            for candidate_meanings in candidate_sets_by_id.values()
            for pair in combinations(sorted(candidate_meanings), 2)
        }
        possible_pairs = len(vertices) * (len(vertices) - 1) // 2
        pool_pairs = len(partitions[stage]) * (len(partitions[stage]) - 1) // 2
        graph_diagnostics[stage] = {
            "available_target_support_size": len(partitions[stage]),
            "observed_target_vertices": len(vertices),
            "cooccurring_target_pairs": len(observed_pairs),
            "possible_pairs_among_observed_vertices": possible_pairs,
            "observed_pair_coverage": (len(observed_pairs) / possible_pairs) if possible_pairs else 1.0,
            "possible_pairs_in_available_support": pool_pairs,
            "available_support_pair_coverage": (len(observed_pairs) / pool_pairs) if pool_pairs else 1.0,
            "complete_conflict_graph_on_observed_vertices": len(observed_pairs) == possible_pairs,
        }

    safe_meta = {
        "schema": SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "split_sha256": split["split_sha256"],
        "split_spec": {
            "seed": split["seed"],
            "attributes": list(split["attributes"]),
            "values_by_attribute": split["values_by_attribute"],
            **({"ontology_id": split["ontology_id"]} if "ontology_id" in split else {}),
        },
        "attributes": list(split["attributes"]),
        "task_key_id": hashlib.sha256(task_key).hexdigest()[:16],
        "task_seed": task_seed,
        "k": k,
        "sets_per_stage": sets_per_stage,
        "partition_sizes": {name: len(ids) for name, ids in partitions.items()},
        "episodes_per_stage": {name: len(rows) for name, rows in sender.items()},
        "conflict_graph_coverage": graph_diagnostics,
        "chance_accuracy": 1.0 / k,
        "warning": "Keep gold ledgers evaluator-only. Do not expose episode IDs or task-key-derived assignments to model tools.",
    }
    return {"sender": sender, "receiver": receiver, "gold": gold, "manifest": safe_meta}


def write_ledgers(bundle: dict[str, Any], output_dir: Path, *, force: bool = False) -> dict[str, Any]:
    output_dir = _inside_project(output_dir)
    expected_outputs = [
        output_dir / f"{role}_{stage}.jsonl"
        for role in ("sender", "receiver", "gold")
        for stage in bundle[role]
    ] + [output_dir / "manifest.json"]
    if not force and any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"refusing to overwrite existing episode bundle: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(bundle["manifest"])
    files = {}
    for role in ("sender", "receiver", "gold"):
        files[role] = {}
        for stage, rows in bundle[role].items():
            filename = f"{role}_{stage}.jsonl"
            data = b"".join(_json_bytes(row) + b"\n" for row in rows)
            (output_dir / filename).write_bytes(data)
            files[role][stage] = {"file": filename, "sha256": hashlib.sha256(data).hexdigest(), "records": len(rows)}
    manifest["files"] = files
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_ledgers(bundle: dict[str, Any]) -> None:
    sender, receiver, gold = bundle["sender"], bundle["receiver"], bundle["gold"]
    if set(sender) != set(receiver) or set(sender) != set(gold):
        raise ValueError("role ledgers have different stage sets")
    heldout_ids_by_stage: dict[str, set[str]] = {}
    attributes = bundle.get("manifest", {}).get("attributes")
    if not isinstance(attributes, list) or not attributes or len(set(attributes)) != len(attributes):
        raise ValueError("episode manifest must declare unique attribute order")
    for stage in sender:
        srows, rrows, grows = sender[stage], receiver[stage], gold[stage]
        if not (len(srows) == len(rrows) == len(grows)):
            raise ValueError("role ledger counts differ")
        targets_by_set: dict[str, list[str]] = {}
        candidates_by_set: dict[str, list[dict[str, Any]]] = {}
        meanings_by_stage = heldout_ids_by_stage.setdefault(stage, set())
        for srow, rrow, grow in zip(srows, rrows, grows):
            candidates = rrow["candidates"]
            ids = [row["candidate_id"] for row in candidates]
            if len(ids) != len(set(ids)) or grow["candidate_id"] not in ids:
                raise ValueError("invalid candidate IDs or target")
            target = next(row for row in candidates if row["candidate_id"] == grow["candidate_id"])
            if set(target.get("attributes", {})) != set(attributes):
                raise ValueError("target attributes do not match manifest order")
            if target["attributes"] != srow["private_meaning"]:
                raise ValueError("sender meaning does not match evaluator target")
            target_tuple = [target["attributes"][attribute] for attribute in attributes]
            canonical_tuple = json.dumps(target_tuple, ensure_ascii=False, separators=(",", ":"))
            expected_meaning_id = "m-" + hashlib.sha256(canonical_tuple.encode("utf-8")).hexdigest()[:16]
            if grow["meaning_id"] != expected_meaning_id:
                raise ValueError("evaluator meaning ID does not match target tuple")
            targets_by_set.setdefault(grow["candidate_set_id"], []).append(grow["candidate_id"])
            signature = candidates_by_set.setdefault(grow["candidate_set_id"], candidates)
            if signature != candidates:
                raise ValueError("candidate table changed within a balanced candidate set")
            meanings_by_stage.add(grow["meaning_id"])
            # No evaluator label appears in either model-facing view.
            if any(key in srow or key in rrow for key in ("meaning_id", "candidate_set_id", "target", "gold")):
                raise ValueError("evaluator label leaked into model view")
        if not rrows:
            raise ValueError("empty stage ledger")
        for target_ids in targets_by_set.values():
            if len(target_ids) != len(set(target_ids)) or len(target_ids) != len(rrows[0]["candidates"]):
                raise ValueError("candidate set is not exactly target-balanced")
    if heldout_ids_by_stage.get("validation", set()) & heldout_ids_by_stage.get("test", set()):
        raise ValueError("validation and test target meanings overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    key_parser = sub.add_parser("keygen")
    key_parser.add_argument("--key", type=Path, default=Path(".cache/emergent_ood_v0_4/evaluator.key"))
    gen = sub.add_parser("generate")
    gen.add_argument("--key", type=Path, default=Path(".cache/emergent_ood_v0_4/evaluator.key"))
    gen.add_argument("--seed", type=int, default=17)
    gen.add_argument("--ontology", type=Path, help="project-local ontology spec; omit for the default fixture")
    gen.add_argument("--task-seed", type=int, default=0)
    gen.add_argument("--k", type=int, default=4)
    gen.add_argument("--sets-per-stage", type=int, default=16)
    gen.add_argument("--output-dir", type=Path, default=Path(".cache/emergent_ood_v0_4/episodes"))
    gen.add_argument("--force", action="store_true", help="overwrite an existing local episode bundle")
    args = parser.parse_args()
    if args.command == "keygen":
        print(create_task_key(args.key))
        return
    ontology = load_ontology_spec(args.ontology) if args.ontology else None
    split = build_split(
        seed=args.seed,
        attributes=ontology["attributes"] if ontology else DEFAULT_ATTRIBUTES,
        values=[ontology["values_by_attribute"][name] for name in ontology["attributes"]]
        if ontology else DEFAULT_VALUES,
        ontology_id=ontology["ontology_id"] if ontology else None,
    )
    key = load_task_key(args.key)
    bundle = generate_ledgers(split=split, task_key=key, task_seed=args.task_seed, k=args.k, sets_per_stage=args.sets_per_stage)
    verify_ledgers(bundle)
    print(json.dumps(write_ledgers(bundle, args.output_dir, force=args.force), indent=2))


if __name__ == "__main__":
    main()
