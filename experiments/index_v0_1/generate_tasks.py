"""Generate deterministic synthetic INDEX_m communication episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def episode_seed(master_seed: int, split: str, length: int, episode_index: int) -> int:
    payload = f"INDEX_m-v0.1|{master_seed}|{split}|{length}|{episode_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def generate_episode(master_seed: int, split: str, length: int, episode_index: int) -> dict:
    if length < 1:
        raise ValueError("length must be positive")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    rng = random.Random(episode_seed(master_seed, split, length, episode_index))
    bits = [rng.getrandbits(1) for _ in range(length)]
    index = rng.randrange(length) + 1
    episode = {
        "task": "INDEX_m",
        "version": "0.1",
        "split": split,
        "master_seed": master_seed,
        "episode_id": f"{split}-m{length}-e{episode_index:05d}",
        "length": length,
        "sender_view": {"bits": bits},
        "receiver_view": {"index_1_based": index},
        "gold_bit": bits[index - 1],
    }
    validate_episode(episode)
    return episode


def validate_episode(episode: dict) -> None:
    length = episode["length"]
    sender = episode["sender_view"]
    receiver = episode["receiver_view"]
    bits = sender["bits"]
    index = receiver["index_1_based"]
    if len(bits) != length or any(bit not in (0, 1) for bit in bits):
        raise ValueError("sender bit vector does not match declared length or alphabet")
    if not 1 <= index <= length:
        raise ValueError("receiver index is outside the vector")
    if episode["gold_bit"] != bits[index - 1]:
        raise ValueError("gold answer does not match the indexed sender bit")
    if set(sender) & set(receiver):
        raise ValueError("sender and receiver views must use disjoint fields")
    if any(key in sender for key in ("index_1_based", "gold_bit")):
        raise ValueError("sender view leaks receiver input or answer")
    if any(key in receiver for key in ("bits", "gold_bit")):
        raise ValueError("receiver view leaks sender input or answer")


def parse_exact_bit(output: str) -> int | None:
    value = output.strip()
    if value in {"0", "1"}:
        return int(value)
    return None


def score_episode(episode: dict, output: str) -> dict:
    prediction = parse_exact_bit(output)
    return {
        "episode_id": episode["episode_id"],
        "prediction": prediction,
        "gold_bit": episode["gold_bit"],
        "exact": prediction is not None and prediction == episode["gold_bit"],
    }


def generate_shard(split: str, seed: int, lengths: list[int], episodes_per_length: int) -> list[dict]:
    if episodes_per_length < 1:
        raise ValueError("episodes_per_length must be positive")
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    if len(set(lengths)) != len(lengths):
        raise ValueError("lengths must be unique")
    return [
        generate_episode(seed, split, length, episode_index)
        for length in lengths
        for episode_index in range(episodes_per_length)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "dev", "test", "pilot"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--lengths", required=True, nargs="+", type=int)
    parser.add_argument("--episodes-per-length", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    episodes = generate_shard(args.split, args.seed, args.lengths, args.episodes_per_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "episodes": len(episodes), "split": args.split,
                      "lengths": args.lengths, "episodes_per_length": args.episodes_per_length}))


if __name__ == "__main__":
    main()
