"""Generate deterministic, role-separated private-record matching tasks.

This is a model-free calibration task inspired by the MT-PingEval name-game
family. It uses no upstream task records or generated benchmark data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "tlu.private-match.v1"
GENERATOR_VERSION = "0.1.0"
ROLE_FILES = ("sender.jsonl", "receiver.jsonl", "gold.jsonl")


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def generate_episode(
    *, episode_id: str, seed: int, candidate_count: int, feature_count: int,
    vocabulary_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return sender view, receiver view, and gold for one episode.

    Candidate rows are generated and shuffled before the target index is drawn.
    Conditional on the receiver's complete view, the target is therefore
    uniform over candidate IDs and the exact no-message Bayes accuracy is 1/n.
    """
    if not episode_id or not isinstance(episode_id, str):
        raise ValueError("episode_id must be a non-empty string")
    for name, value, minimum in (
        ("seed", seed, 0), ("candidate_count", candidate_count, 2),
        ("feature_count", feature_count, 1), ("vocabulary_size", vocabulary_size, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if vocabulary_size ** feature_count < candidate_count:
        raise ValueError("the feature tuple space must contain at least candidate_count rows")

    # Domain separation prevents the deterministic target draw from reusing
    # the random stream that constructed/shuffled the receiver's table.
    table_rng = random.Random(f"tlu.private-match.v1/table/{seed}")
    target_rng = random.Random(f"tlu.private-match.v1/target/{seed}")
    feature_names = [f"f{i}" for i in range(feature_count)]
    rows: set[tuple[int, ...]] = set()
    # Sampling without replacement from the Cartesian product avoids an
    # unbounded collision loop when the requested table is relatively dense.
    space_size = vocabulary_size ** feature_count
    if space_size <= 1_000_000:
        ranks = table_rng.sample(range(space_size), candidate_count)
        for rank in ranks:
            values = []
            remainder = rank
            for _ in range(feature_count):
                values.append(remainder % vocabulary_size)
                remainder //= vocabulary_size
            rows.add(tuple(values))
    else:
        while len(rows) < candidate_count:
            rows.add(tuple(table_rng.randrange(vocabulary_size) for _ in feature_names))
    ordered_rows = sorted(rows)
    table_rng.shuffle(ordered_rows)
    target_index = target_rng.randrange(candidate_count)

    candidates = [
        {
            "candidate_id": f"r{i:04d}",
            "record": {name: f"v{value:04d}" for name, value in zip(feature_names, row)},
        }
        for i, row in enumerate(ordered_rows)
    ]
    target_record = candidates[target_index]["record"].copy()
    sender_view = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "feature_names": feature_names,
        "target_record": target_record,
    }
    receiver_view = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "feature_names": feature_names,
        "candidates": candidates,
    }
    gold = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "target_candidate_id": f"r{target_index:04d}",
    }
    validate_episode(sender_view, receiver_view, gold)
    return sender_view, receiver_view, gold


def validate_episode(sender: dict[str, Any], receiver: dict[str, Any], gold: dict[str, Any]) -> None:
    """Reject malformed or accidentally leaky generated episodes."""
    if not (sender.get("schema_version") == receiver.get("schema_version") == gold.get("schema_version") == SCHEMA_VERSION):
        raise ValueError("schema version mismatch")
    ids = {v.get("episode_id") for v in (sender, receiver, gold)}
    if len(ids) != 1 or None in ids:
        raise ValueError("episode IDs must agree")
    names = sender.get("feature_names")
    if not isinstance(names, list) or not names or receiver.get("feature_names") != names:
        raise ValueError("feature names must be a non-empty matching list")
    target = sender.get("target_record")
    candidates = receiver.get("candidates")
    if not isinstance(target, dict) or set(target) != set(names):
        raise ValueError("sender target record does not match feature schema")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("receiver must have at least two candidates")
    candidate_ids = [c.get("candidate_id") for c in candidates if isinstance(c, dict)]
    if len(candidate_ids) != len(candidates) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate records must have unique IDs")
    records = [c.get("record") for c in candidates]
    if any(not isinstance(r, dict) or set(r) != set(names) for r in records):
        raise ValueError("candidate record does not match feature schema")
    if len({tuple(r[n] for n in names) for r in records}) != len(records):
        raise ValueError("candidate records must be unique")
    matches = [c["candidate_id"] for c in candidates if c["record"] == target]
    if len(matches) != 1 or gold.get("target_candidate_id") != matches[0]:
        raise ValueError("gold must identify the unique exact record match")
    if set(receiver).intersection({"target_record", "target_candidate_id", "target_index"}):
        raise ValueError("receiver view leaks the target")


def score_answer(receiver_view: dict[str, Any], gold: dict[str, Any], answer: str) -> bool:
    """Strictly score one exact candidate ID against scorer-only gold."""
    if not isinstance(answer, str):
        return False
    valid_ids = {c["candidate_id"] for c in receiver_view["candidates"]}
    return answer in valid_ids and answer == gold.get("target_candidate_id")


def no_message_bayes_accuracy(candidate_count: int) -> float:
    """Exact no-message Bayes accuracy under the uniform target-index prior."""
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 2:
        raise ValueError("candidate_count must be an integer >= 2")
    return 1.0 / candidate_count


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_dataset(
    output: Path, *, episodes: int, seed: int, candidate_count: int,
    feature_count: int, vocabulary_size: int, force: bool = False,
) -> dict[str, Any]:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be >= 1")
    output = output.resolve()
    names = (*ROLE_FILES, "manifest.json")
    existing = [output / name for name in names if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite existing dataset files: {existing}")
    output.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[str]] = {name: [] for name in ROLE_FILES}
    for i in range(episodes):
        episode = generate_episode(
            # Do not put generation seeds in agent-visible episode IDs.
            episode_id=f"pm-{i:06d}", seed=seed + i,
            candidate_count=candidate_count, feature_count=feature_count,
            vocabulary_size=vocabulary_size,
        )
        for filename, item in zip(ROLE_FILES, episode):
            rows[filename].append(_json_line(item))

    for filename, lines in rows.items():
        (output / filename).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "python_version": platform.python_version(),
        "randomness": "random.Random with versioned, domain-separated string seeds for table and target streams",
        "episodes": episodes,
        "seed_start": seed,
        "candidate_count": candidate_count,
        "feature_count": feature_count,
        "vocabulary_size_per_feature": vocabulary_size,
        "no_message_bayes_accuracy": {"numerator": 1, "denominator": candidate_count},
        "files_sha256": {filename: _digest(output / filename) for filename in ROLE_FILES},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument("--vocabulary-size", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="replace generated role files and manifest")
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output, episodes=args.episodes, seed=args.seed,
        candidate_count=args.candidates, feature_count=args.features,
        vocabulary_size=args.vocabulary_size, force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
