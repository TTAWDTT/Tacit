"""Generate frozen IID private-sum episodes with role-separated ledgers.

This is a data-only utility. It does not contact models or endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import platform
import secrets
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"
SCHEMA = f"tlu.multiparty-private-sum.v{VERSION}"
DOMAIN = 4
MAX_CALLS = 12


def _inside_project(path: Path) -> Path:
    result = path.resolve()
    try:
        result.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("key and output paths must stay inside the project directory") from exc
    return result


def _check_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("task key must be exactly 32 secret bytes")


class _Stream:
    """Domain-separated HMAC-SHA256 stream and unbiased bounded draws."""

    def __init__(self, key: bytes, *, domain: str, seed: int) -> None:
        _check_key(key)
        if not domain.isascii() or not domain:
            raise ValueError("domain must be non-empty ASCII")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.key = key
        self.prefix = f"{SCHEMA}/{domain}/{seed}/".encode("ascii")
        self.counter = 0
        self.buffer = bytearray()

    def _read(self, size: int) -> bytes:
        while len(self.buffer) < size:
            block = hmac.new(self.key, self.prefix + self.counter.to_bytes(8, "big"), hashlib.sha256).digest()
            self.counter += 1
            self.buffer.extend(block)
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def randbelow(self, stop: int) -> int:
        if isinstance(stop, bool) or not isinstance(stop, int) or stop < 1:
            raise ValueError("stop must be a positive integer")
        bits = (stop - 1).bit_length()
        while True:
            count = (bits + 7) // 8
            value = int.from_bytes(self._read(count), "big") & ((1 << bits) - 1) if bits else 0
            if value < stop:
                return value


def create_task_key(path: Path, *, force: bool = False) -> str:
    path = _inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to replace evaluator key: {path}")
    key = secrets.token_bytes(32)
    if force:
        path.write_bytes(key)
    else:
        with path.open("xb") as stream:
            stream.write(key)
    return hashlib.sha256(key).hexdigest()[:16]


def load_task_key(path: Path) -> bytes:
    key = _inside_project(path).read_bytes()
    _check_key(key)
    return key


def _line(row: Any) -> str:
    return json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def model_visible_view(view: dict[str, Any]) -> dict[str, Any]:
    """Strip evaluator alignment metadata before building a model prompt."""
    if not isinstance(view, dict):
        raise ValueError("view must be an object")
    return {key: value for key, value in view.items() if key != "episode_id"}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_dataset(
    output: Path, *, task_key: bytes, task_seed: int = 0,
    agent_counts: tuple[int, ...] = tuple(range(2, 12)), episodes_per_count: int = 32,
    force: bool = False,
) -> dict[str, Any]:
    """Create per-sender, receiver, and scorer ledgers for fixed IID tuples."""
    _check_key(task_key)
    if isinstance(task_seed, bool) or not isinstance(task_seed, int) or task_seed < 0:
        raise ValueError("task_seed must be a non-negative integer")
    if isinstance(episodes_per_count, bool) or not isinstance(episodes_per_count, int) or episodes_per_count < 1:
        raise ValueError("episodes_per_count must be a positive integer")
    counts = tuple(agent_counts)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("agent_counts must be a non-empty sequence without duplicates")
    if any(isinstance(m, bool) or not isinstance(m, int) or m < 2 or m + 1 > MAX_CALLS for m in counts):
        raise ValueError(f"each agent count must be between 2 and {MAX_CALLS - 1} for the frozen call ceiling")
    output = _inside_project(output)
    output.mkdir(parents=True, exist_ok=True)

    files: dict[str, list[str]] = {}
    for m in counts:
        senders = {f"sender_m{m:02d}_s{i + 1:02d}.jsonl": [] for i in range(m)}
        receiver_name, gold_name = f"receiver_m{m:02d}.jsonl", f"gold_m{m:02d}.jsonl"
        receiver_rows: list[str] = []
        gold_rows: list[str] = []
        for episode_index in range(episodes_per_count):
            # Independent stream per (m, episode): public task_seed alone cannot reveal values.
            stream_seed = task_seed * 1_000_000 + m * episodes_per_count + episode_index
            rng = _Stream(task_key, domain=f"inputs-m{m:02d}-ep{episode_index:06d}", seed=stream_seed)
            values = [rng.randbelow(DOMAIN) for _ in range(m)]
            episode_id = f"sum-m{m:02d}-s{task_seed:08d}-e{episode_index:06d}"
            for i, value in enumerate(values):
                name = f"sender_m{m:02d}_s{i + 1:02d}.jsonl"
                senders[name].append(_line({
                    "schema_version": SCHEMA, "episode_id": episode_id,
                    "sender_count": m, "role": f"sender_{i + 1}", "private_value": value,
                }))
            receiver_rows.append(_line({
                "schema_version": SCHEMA, "episode_id": episode_id,
                "sender_count": m, "role": "receiver",
                "prior": "independent_uniform_integer_0_to_3",
            }))
            gold_rows.append(_line({
                "schema_version": SCHEMA, "episode_id": episode_id,
                "values_in_sender_order": values, "exact_sum": sum(values),
            }))
        files.update(senders)
        files[receiver_name] = receiver_rows
        files[gold_name] = gold_rows

    targets = [output / name for name in (*files.keys(), "manifest.json")]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite generated dataset files: {existing}")
    for name, lines in files.items():
        (output / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": SCHEMA,
        "generator_version": VERSION,
        "python_version": platform.python_version(),
        "task_seed": task_seed,
        "agent_counts": list(counts),
        "episodes_per_agent_count": episodes_per_count,
        "input_domain": list(range(DOMAIN)),
        "input_prior": "independent_uniform_per_sender",
        "randomness": "domain-separated HMAC-SHA256 counter streams with rejection-sampled bounded draws; the evaluator-only 256-bit key is not stored in this manifest",
        "task_key_id": hashlib.sha256(task_key).hexdigest()[:16],
        "max_model_calls_per_communicating_episode": MAX_CALLS,
        "call_count_by_agent_count": {str(m): m + 1 for m in counts},
        "pairing": "same keyed episode tuples across all protocol conditions within each agent count; counts represent different task sizes and are not row-paired",
        "files_sha256": {name: _digest(output / name) for name in files},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    validate_dataset(output)
    return manifest


def validate_dataset(output: Path) -> dict[str, Any]:
    """Verify hashes, episode alignment, IID task schema, and role separation."""
    output = _inside_project(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("schema version mismatch")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("manifest file hash map is missing")
    for name, expected in hashes.items():
        path = _inside_project(output / name)
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"file hash mismatch: {name}")
    for m in manifest["agent_counts"]:
        recv_path, gold_path = output / f"receiver_m{m:02d}.jsonl", output / f"gold_m{m:02d}.jsonl"
        recv = [json.loads(line) for line in recv_path.read_text(encoding="utf-8").splitlines()]
        gold = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()]
        if len(recv) != manifest["episodes_per_agent_count"] or len(gold) != len(recv):
            raise ValueError(f"episode count mismatch for m={m}")
        senders = []
        for i in range(m):
            path = output / f"sender_m{m:02d}_s{i + 1:02d}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if len(rows) != len(recv):
                raise ValueError(f"sender episode count mismatch for m={m}, sender={i + 1}")
            senders.append(rows)
        for j, (rrow, grow) in enumerate(zip(recv, gold)):
            eid = rrow.get("episode_id")
            expected_values = []
            for i, rows in enumerate(senders):
                srow = rows[j]
                if (
                    srow.get("schema_version"), srow.get("episode_id"),
                    srow.get("sender_count"), srow.get("role"),
                ) != (SCHEMA, eid, m, f"sender_{i + 1}"):
                    raise ValueError("sender role alignment/schema mismatch")
                value = srow.get("private_value")
                if isinstance(value, bool) or not isinstance(value, int) or value not in range(DOMAIN):
                    raise ValueError("sender private value outside {0,1,2,3}")
                if "exact_sum" in srow or "values_in_sender_order" in srow:
                    raise ValueError("sender ledger leaks evaluator data")
                expected_values.append(value)
            if (rrow.get("schema_version"), rrow.get("role"), rrow.get("sender_count"), rrow.get("prior")) != (
                SCHEMA, "receiver", m, "independent_uniform_integer_0_to_3"
            ) or "exact_sum" in rrow or "values_in_sender_order" in rrow or "private_value" in rrow:
                raise ValueError("receiver schema or privacy violation")
            if (grow.get("episode_id"), grow.get("values_in_sender_order"), grow.get("exact_sum")) != (
                eid, expected_values, sum(expected_values)
            ):
                raise ValueError("gold ledger does not match sender inputs")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-key-file", type=Path, required=True)
    parser.add_argument("--create-task-key", action="store_true")
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--episodes-per-count", type=int, default=32)
    parser.add_argument("--agent-counts", type=int, nargs="+", default=list(range(2, 12)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        if args.create_task_key:
            print(json.dumps({"task_key_file": str(args.task_key_file), "task_key_id": create_task_key(args.task_key_file, force=args.force)}, indent=2))
            return
        if args.output is None:
            parser.error("--output is required unless --create-task-key is supplied")
        manifest = generate_dataset(
            args.output, task_key=load_task_key(args.task_key_file), task_seed=args.task_seed,
            agent_counts=tuple(args.agent_counts), episodes_per_count=args.episodes_per_count, force=args.force,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
