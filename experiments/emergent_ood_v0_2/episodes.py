"""Generate role-separated receiver-utility episodes for held-out compositions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
from typing import Any


_SPLIT_PATH = Path(__file__).resolve().parents[1] / "emergent_ood_v0_1" / "split.py"
_SPLIT_SPEC = importlib.util.spec_from_file_location("emergent_ood_split", _SPLIT_PATH)
if _SPLIT_SPEC is None or _SPLIT_SPEC.loader is None:
    raise RuntimeError(f"could not load split generator at {_SPLIT_PATH}")
_SPLIT_MODULE = importlib.util.module_from_spec(_SPLIT_SPEC)
_SPLIT_SPEC.loader.exec_module(_SPLIT_MODULE)
AXES = tuple(_SPLIT_MODULE.AXES)
VALUES = tuple(_SPLIT_MODULE.VALUES)


def meaning_id(values: tuple[int, ...]) -> str:
    if len(values) != len(AXES) or any(
        isinstance(value, bool) or not isinstance(value, int) or value not in VALUES
        for value in values
    ):
        raise ValueError("meaning is outside the declared domain")
    return "-".join(f"{axis[0]}{value}" for axis, value in zip(AXES, values))


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_ledgers(seed: int, candidate_count: int = 5) -> dict[str, Any]:
    """Enumerate target-balanced candidate sets from held-out meanings.

    For each k-subset of the nine held-out meanings, emit one episode for each
    member as the private target. Conditional on the receiver's candidate set,
    every candidate is therefore exactly equally likely to be the target. The
    no-message Bayes accuracy is exactly 1/k, not an estimate from sampling.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise ValueError("candidate_count must be an integer")
    heldout_size = 9
    if not 2 <= candidate_count <= heldout_size:
        raise ValueError("candidate_count must be between 2 and 9")

    split = _SPLIT_MODULE.generate_split(seed)
    _SPLIT_MODULE.verify_split(split)
    split_hash = _canonical_hash(split)
    meanings = [tuple(row[axis] for axis in AXES) for row in split["held_out_meanings"]]
    meanings.sort()

    sender_rows: list[dict[str, Any]] = []
    receiver_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for subset in itertools.combinations(meanings, candidate_count):
        subset_hash = _canonical_hash([meaning_id(values) for values in subset])
        for target in subset:
            target_id = meaning_id(target)
            ordered_candidates = sorted(
                subset,
                key=lambda values: hashlib.sha256(
                    f"{seed}|{candidate_count}|{subset_hash}|{target_id}|{meaning_id(values)}".encode("ascii")
                ).digest(),
            )
            candidate_ids = [meaning_id(values) for values in ordered_candidates]
            episode_id = f"eood-v0.2-s{seed}-k{candidate_count}-{subset_hash[:12]}-{target_id}"
            sender_rows.append({
                "schema": "tlu.emergent_ood_sender.v1",
                "episode_id": episode_id,
                "split_sha256": split_hash,
                "candidate_count": candidate_count,
                "private_target": dict(zip(AXES, target)),
                "private_target_id": target_id,
            })
            receiver_rows.append({
                "schema": "tlu.emergent_ood_receiver.v1",
                "episode_id": episode_id,
                "split_sha256": split_hash,
                "candidate_count": candidate_count,
                "candidates": [
                    {"candidate_id": meaning_id(values), **dict(zip(AXES, values))}
                    for values in ordered_candidates
                ],
            })
            gold_rows.append({
                "schema": "tlu.emergent_ood_gold.v1",
                "episode_id": episode_id,
                "candidate_count": candidate_count,
                "candidate_ids": candidate_ids,
                "target_id": target_id,
                "target_position": candidate_ids.index(target_id),
            })

    ledgers = {
        "schema": "tlu.emergent_ood_ledgers.v1",
        "seed": seed,
        "candidate_count": candidate_count,
        "held_out_target_count": len(meanings),
        "split_sha256": split_hash,
        "train_meanings": split["train_meanings"],
        "episode_count": len(gold_rows),
        "no_message_bayes_accuracy": 1 / candidate_count,
        "ideal_zero_error_payload_floor_bits": math.ceil(math.log2(len(meanings))),
        "payload_floor_assumptions": "sender does not know receiver candidates; fixed shared target-index codebook is preinstalled and its setup cost is excluded",
        "sender": sender_rows,
        "receiver": receiver_rows,
        "gold": gold_rows,
    }
    verify_ledgers(ledgers)
    return ledgers


def verify_ledgers(ledgers: dict[str, Any]) -> None:
    """Check role separation, exact episode alignment, and analytic chance."""
    sender, receiver, gold = (ledgers.get(name) for name in ("sender", "receiver", "gold"))
    if not all(isinstance(rows, list) for rows in (sender, receiver, gold)):
        raise ValueError("sender, receiver, and gold ledgers must be lists")
    if any(not isinstance(row, dict) for rows in (sender, receiver, gold) for row in rows):
        raise ValueError("each role-ledger row must be an object")
    if len(sender) != len(receiver) or len(receiver) != len(gold):
        raise ValueError("role ledgers must have equal row counts")
    candidate_count = ledgers.get("candidate_count")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or not 2 <= candidate_count <= 9:
        raise ValueError("candidate_count must be an integer from 2 through 9")
    sender_by_id = {row.get("episode_id"): row for row in sender}
    receiver_by_id = {row.get("episode_id"): row for row in receiver}
    gold_by_id = {row.get("episode_id"): row for row in gold}
    if len(sender_by_id) != len(sender) or len(receiver_by_id) != len(receiver) or len(gold_by_id) != len(gold):
        raise ValueError("episode_id values must be unique within each role ledger")
    if set(sender_by_id) != set(receiver_by_id) or set(receiver_by_id) != set(gold_by_id):
        raise ValueError("role ledgers must contain exactly the same episode IDs")

    split = _SPLIT_MODULE.generate_split(ledgers.get("seed"))
    _SPLIT_MODULE.verify_split(split)
    heldout_ids = {
        meaning_id(tuple(row[axis] for axis in AXES))
        for row in split["held_out_meanings"]
    }
    expected_split_hash = _canonical_hash(split)
    if ledgers.get("split_sha256") != expected_split_hash:
        raise ValueError("split hash does not match the declared seed")

    targets_by_candidate_set: dict[tuple[str, ...], set[str]] = {}
    for episode_id in sender_by_id:
        source = sender_by_id[episode_id]
        choices = receiver_by_id[episode_id]
        answer = gold_by_id[episode_id]
        if "private_target" not in source or "candidates" in source:
            raise ValueError("sender view must contain a private target, not receiver candidates")
        if not isinstance(source["private_target"], dict) or set(source["private_target"]) != set(AXES):
            raise ValueError("sender private target must contain exactly the declared axes")
        if "private_target" in choices or "private_target_id" in choices or "target_id" in choices:
            raise ValueError("receiver view leaks the target label")
        candidate_rows = choices.get("candidates")
        if not isinstance(candidate_rows, list) or any(not isinstance(row, dict) for row in candidate_rows):
            raise ValueError("receiver candidates must be a list of objects")
        candidate_ids = [row.get("candidate_id") for row in candidate_rows]
        if len(candidate_ids) != candidate_count or len(set(candidate_ids)) != candidate_count:
            raise ValueError("receiver candidates must be unique and match candidate_count")
        for row in candidate_rows:
            if not isinstance(row, dict) or set(row) != {"candidate_id", *AXES}:
                raise ValueError("candidate rows must contain one ID and all declared attributes")
            values = tuple(row[axis] for axis in AXES)
            if row["candidate_id"] != meaning_id(values):
                raise ValueError("candidate identity must match its valid attribute values")
            if row["candidate_id"] not in heldout_ids:
                raise ValueError("receiver candidates must all belong to the held-out set")
        if candidate_ids != answer.get("candidate_ids"):
            raise ValueError("gold candidate order must match the receiver view")
        if source.get("private_target_id") != answer.get("target_id"):
            raise ValueError("sender target and gold target disagree")
        if answer.get("target_id") not in candidate_ids:
            raise ValueError("target must occur in receiver candidate set")
        if candidate_ids.index(answer["target_id"]) != answer.get("target_position"):
            raise ValueError("gold target_position is incorrect")
        target_values = tuple(source["private_target"].get(axis) for axis in AXES)
        if meaning_id(target_values) != answer["target_id"]:
            raise ValueError("sender private target attributes and gold target disagree")
        if answer["target_id"] not in heldout_ids:
            raise ValueError("private target must be a held-out meaning")
        if source.get("split_sha256") != choices.get("split_sha256") or source.get("split_sha256") != ledgers.get("split_sha256"):
            raise ValueError("episode rows must refer to the declared split")
        candidate_set = tuple(sorted(candidate_ids))
        target_set = targets_by_candidate_set.setdefault(candidate_set, set())
        if answer["target_id"] in target_set:
            raise ValueError("each candidate set may use each member as target only once")
        target_set.add(answer["target_id"])

    expected_count = candidate_count * math.comb(9, candidate_count)
    if len(gold) != expected_count or ledgers.get("episode_count") != expected_count:
        raise ValueError("episode count must enumerate every target in every candidate subset")
    if len(targets_by_candidate_set) != math.comb(9, candidate_count) or any(
        targets != set(candidate_set) for candidate_set, targets in targets_by_candidate_set.items()
    ):
        raise ValueError("each held-out candidate subset must be enumerated with every member as target")
    if ledgers.get("no_message_bayes_accuracy") != 1 / candidate_count:
        raise ValueError("no-message Bayes accuracy must equal 1/candidate_count")
    if ledgers.get("held_out_target_count") != 9 or ledgers.get("ideal_zero_error_payload_floor_bits") != 4:
        raise ValueError("the declared 9-target task has a 4-bit fixed-width zero-error floor")


def write_ledgers(ledgers: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "schema": "tlu.emergent_ood_split.v1",
        "seed": ledgers["seed"],
        "sha256": ledgers["split_sha256"],
        "train_meanings": ledgers["train_meanings"],
    }
    for filename, rows in (
        ("sender.jsonl", ledgers["sender"]),
        ("receiver.jsonl", ledgers["receiver"]),
        ("gold.jsonl", ledgers["gold"]),
    ):
        (output_dir / filename).write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {key: value for key, value in ledgers.items() if key not in {"sender", "receiver", "gold"}}
    manifest["split"] = split_payload
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ledgers = generate_ledgers(args.seed, args.candidate_count)
    write_ledgers(ledgers, args.output_dir)


if __name__ == "__main__":
    main()
