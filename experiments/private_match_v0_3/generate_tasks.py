"""Generate a deterministic three-agent complementary-evidence task family.

Two source agents each hold one coordinate of a hidden target. A receiver sees
the complete candidate table. Every coordinate class has q candidates, but
the pair of coordinates identifies exactly one row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from fractions import Fraction
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "tlu.private-match.v3"
GENERATOR_VERSION = "0.3.0"
ROLE_FILES = ("sender_x.jsonl", "sender_y.jsonl", "receiver.jsonl", "gold.jsonl")


def _validate_q(q: int) -> None:
    if isinstance(q, bool) or not isinstance(q, int) or q < 2 or q & (q - 1):
        raise ValueError("q must be an integer power of two >= 2")


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def generate_episode(*, episode_id: str, seed: int, q: int) -> tuple[dict[str, Any], ...]:
    """Return x-sender, y-sender, receiver, and gold views for one episode."""
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _validate_q(q)

    table_rng = random.Random(f"tlu.private-match.v3/table/{seed}")
    target_rng = random.Random(f"tlu.private-match.v3/target/{seed}")
    xs = [f"x{i:04d}" for i in range(q)]
    ys = [f"y{i:04d}" for i in range(q)]
    rows = [(x, y) for x in xs for y in ys]
    table_rng.shuffle(rows)
    candidate_ids = [f"r{i:04d}" for i in range(q * q)]
    table_rng.shuffle(candidate_ids)
    candidates = [
        {"candidate_id": candidate_id, "record": {"x": x, "y": y}}
        for candidate_id, (x, y) in zip(candidate_ids, rows)
    ]
    target_index = target_rng.randrange(q * q)
    target = candidates[target_index]
    x_value = target["record"]["x"]
    y_value = target["record"]["y"]

    shared = {"schema_version": SCHEMA_VERSION, "episode_id": episode_id, "q": q}
    sender_x = {**shared, "role": "sender_x", "coordinate": "x", "private_value": x_value}
    sender_y = {**shared, "role": "sender_y", "coordinate": "y", "private_value": y_value}
    receiver = {
        **shared,
        "role": "receiver",
        "candidates": candidates,
    }
    gold = {**shared, "target_candidate_id": target["candidate_id"]}
    validate_episode(sender_x, sender_y, receiver, gold)
    return sender_x, sender_y, receiver, gold


def validate_episode(
    sender_x: dict[str, Any], sender_y: dict[str, Any],
    receiver: dict[str, Any], gold: dict[str, Any],
) -> None:
    """Check role separation and exact no-message/one-source Bayes strata."""
    views = (sender_x, sender_y, receiver, gold)
    if any(view.get("schema_version") != SCHEMA_VERSION for view in views):
        raise ValueError("schema version mismatch")
    episode_ids = {view.get("episode_id") for view in views}
    if len(episode_ids) != 1 or None in episode_ids:
        raise ValueError("episode IDs must agree")
    q_values = [view.get("q") for view in views]
    for value in q_values:
        _validate_q(value)
    if any(value != q_values[0] for value in q_values[1:]):
        raise ValueError("q must agree across views")
    q = q_values[0]
    if sender_x.get("role") != "sender_x" or sender_x.get("coordinate") != "x":
        raise ValueError("sender_x role schema mismatch")
    if sender_y.get("role") != "sender_y" or sender_y.get("coordinate") != "y":
        raise ValueError("sender_y role schema mismatch")
    if receiver.get("role") != "receiver":
        raise ValueError("receiver role schema mismatch")
    candidates = receiver.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != q * q:
        raise ValueError("receiver must have exactly q squared candidates")
    candidate_ids = [row.get("candidate_id") for row in candidates if isinstance(row, dict)]
    if len(candidate_ids) != len(candidates):
        raise ValueError("candidate rows must be objects")
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
        raise ValueError("candidate IDs must be non-empty strings")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    records = [row.get("record") for row in candidates]
    if any(not isinstance(record, dict) or set(record) != {"x", "y"} for record in records):
        raise ValueError("each candidate record must contain x and y")
    x_domain = {f"x{i:04d}" for i in range(q)}
    y_domain = {f"y{i:04d}" for i in range(q)}
    if any(
        not isinstance(record["x"], str)
        or not isinstance(record["y"], str)
        or record["x"] not in x_domain
        or record["y"] not in y_domain
        for record in records
    ):
        raise ValueError("candidate values fall outside the registered coordinate domains")
    if {(record["x"], record["y"]) for record in records} != {
        (x, y) for x in x_domain for y in y_domain
    }:
        raise ValueError("candidate table must contain the complete Cartesian product")
    x_value = sender_x.get("private_value")
    y_value = sender_y.get("private_value")
    if (
        not isinstance(x_value, str)
        or not isinstance(y_value, str)
        or x_value not in x_domain
        or y_value not in y_domain
    ):
        raise ValueError("source private value falls outside its coordinate domain")
    if set(sender_x).intersection({"target_candidate_id", "target_record", "y", "y_value"}):
        raise ValueError("sender_x view leaks the other coordinate or target identity")
    if set(sender_y).intersection({"target_candidate_id", "target_record", "x", "x_value"}):
        raise ValueError("sender_y view leaks the other coordinate or target identity")
    if set(receiver).intersection({"target_candidate_id", "target_record", "private_value"}):
        raise ValueError("receiver view leaks target evidence")
    gold_id = gold.get("target_candidate_id")
    if not isinstance(gold_id, str) or gold_id not in set(candidate_ids):
        raise ValueError("gold target ID must occur in the candidate table")
    matches = [
        row["candidate_id"] for row in candidates
        if row["record"] == {"x": x_value, "y": y_value}
    ]
    if matches != [gold["target_candidate_id"]]:
        raise ValueError("the two private coordinates must identify exactly the gold row")
    if sum(row["record"]["x"] == x_value for row in candidates) != q:
        raise ValueError("sender_x evidence must leave exactly q candidates")
    if sum(row["record"]["y"] == y_value for row in candidates) != q:
        raise ValueError("sender_y evidence must leave exactly q candidates")


def bayes_accuracy_references(q: int) -> dict[str, Fraction]:
    """Exact Bayes accuracies for no message, either one source, or both."""
    _validate_q(q)
    return {
        "no_message": Fraction(1, q * q),
        "sender_x_only": Fraction(1, q),
        "sender_y_only": Fraction(1, q),
        "both_sources": Fraction(1, 1),
    }


def oracle_answer(receiver: dict[str, Any], x_value: str, y_value: str) -> str:
    """Return the unique candidate ID matching both source values."""
    matches = [
        row["candidate_id"] for row in receiver["candidates"]
        if row["record"] == {"x": x_value, "y": y_value}
    ]
    if len(matches) != 1:
        raise ValueError("oracle requires exactly one matching candidate")
    return matches[0]


def score_answer(receiver: dict[str, Any], gold: dict[str, Any], answer: str) -> bool:
    """Strictly score one exact candidate ID against scorer-only gold."""
    if not isinstance(answer, str):
        return False
    candidate_ids = {row["candidate_id"] for row in receiver["candidates"]}
    return answer in candidate_ids and answer == gold.get("target_candidate_id")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_dataset(output: Path, *, episodes: int, seed: int, q: int, force: bool = False) -> dict[str, Any]:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _validate_q(q)
    output = output.resolve()
    names = (*ROLE_FILES, "manifest.json")
    existing = [output / name for name in names if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite existing dataset files: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[str]] = {name: [] for name in ROLE_FILES}
    for index in range(episodes):
        episode = generate_episode(
            episode_id=f"pmt3-{index:06d}", seed=seed + index, q=q
        )
        for filename, item in zip(ROLE_FILES, episode):
            rows[filename].append(_json_line(item))
    for filename, lines in rows.items():
        (output / filename).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    references = bayes_accuracy_references(q)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "python_version": platform.python_version(),
        "randomness": "random.Random with versioned, domain-separated string seeds for table and target streams",
        "episodes": episodes,
        "seed_start": seed,
        "q": q,
        "agent_count": 3,
        "candidate_count": q * q,
        "no_message_bayes_accuracy": f"{references['no_message'].numerator}/{references['no_message'].denominator}",
        "one_source_bayes_accuracy": f"{references['sender_x_only'].numerator}/{references['sender_x_only'].denominator}",
        "both_sources_oracle_accuracy": "1/1",
        "fixed_width_zero_error_payload_bits": 2 * (q.bit_length() - 1),
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
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="replace generated role files and manifest")
    args = parser.parse_args()
    manifest = generate_dataset(args.output, episodes=args.episodes, seed=args.seed, q=args.q, force=args.force)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
