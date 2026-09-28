"""Generate and score deterministic pointer-chasing communication episodes.

This module is model-free. It does not start an inference service or load a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


TASK = "POINTER_CHASING"
VERSION = "0.1"


def episode_seed(master_seed: int, split: str, size: int, depth: int, episode_index: int) -> int:
    if not split:
        raise ValueError("split must be non-empty")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    payload = f"{TASK}-v{VERSION}|{master_seed}|{split}|{size}|{depth}|{episode_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def pointer_trace(function_a: list[int], function_b: list[int], depth: int) -> list[int]:
    """Return p_0,...,p_k for one-based maps over [n]={1,...,n}."""
    if len(function_a) < 2 or len(function_a) != len(function_b):
        raise ValueError("both functions must have the same domain size n >= 2")
    if depth < 1:
        raise ValueError("depth must be positive")
    size = len(function_a)
    for function in (function_a, function_b):
        if any(type(value) is not int or not 1 <= value <= size for value in function):
            raise ValueError("function values must be integers in [1,n]")
    trace = [1]
    for step in range(1, depth + 1):
        function = function_a if step % 2 else function_b
        trace.append(function[trace[-1] - 1])
    return trace


def oracle_answer(episode: dict) -> int:
    trace = pointer_trace(
        episode["agent_a_view"]["function_values_1_based"],
        episode["agent_b_view"]["function_values_1_based"],
        episode["depth"],
    )
    return trace[-1] % 2


def encode_pointer(pointer: int, size: int) -> str:
    """Encode a one-based pointer as a fixed-width binary offset in [0,n)."""
    if size < 2 or type(pointer) is not int or not 1 <= pointer <= size:
        raise ValueError("pointer must be an integer in [1,n] with n >= 2")
    width = (size - 1).bit_length()
    return format(pointer - 1, f"0{width}b")


def decode_pointer(message: str, size: int) -> int:
    """Decode exactly ceil(log2(n)) bits to a one-based pointer."""
    if size < 2:
        raise ValueError("size must be at least 2")
    width = (size - 1).bit_length()
    if len(message) != width or any(bit not in "01" for bit in message):
        raise ValueError("pointer message must contain exactly ceil(log2(n)) binary digits")
    pointer = int(message, 2) + 1
    if pointer > size:
        raise ValueError("binary code is outside the pointer domain")
    return pointer


def oracle_relay(episode: dict) -> dict:
    """Simulate the k-message pointer relay; report payload bits, excluding framing."""
    validate_episode(episode)
    size, depth = episode["size"], episode["depth"]
    function_a = episode["agent_a_view"]["function_values_1_based"]
    function_b = episode["agent_b_view"]["function_values_1_based"]
    pointer = 1
    messages = []
    width = (size - 1).bit_length()
    for step in range(1, depth + 1):
        sender = "agent_a" if step % 2 else "agent_b"
        receiver = "agent_b" if step % 2 else "agent_a"
        function = function_a if step % 2 else function_b
        pointer = function[pointer - 1]
        payload = encode_pointer(pointer, size)
        received_pointer = decode_pointer(payload, size)
        messages.append({
            "step": step,
            "sender": sender,
            "receiver": receiver,
            "payload_bits": payload,
            "decoded_pointer_1_based": received_pointer,
        })
        pointer = received_pointer
    return {
        "answer_bit": pointer % 2,
        "messages": messages,
        "payload_bit_count": depth * width,
        "framing_included": False,
    }


def generate_episode(master_seed: int, split: str, size: int, depth: int, episode_index: int) -> dict:
    if size < 2:
        raise ValueError("size must be at least 2")
    if depth < 1:
        raise ValueError("depth must be positive")
    rng = random.Random(episode_seed(master_seed, split, size, depth, episode_index))
    function_a = [rng.randrange(1, size + 1) for _ in range(size)]
    function_b = [rng.randrange(1, size + 1) for _ in range(size)]
    episode = {
        "task": TASK,
        "version": VERSION,
        "split": split,
        "master_seed": master_seed,
        "episode_id": f"{split}-n{size}-k{depth}-e{episode_index:05d}",
        "size": size,
        "depth": depth,
        "agent_a_view": {"function_values_1_based": function_a},
        "agent_b_view": {"function_values_1_based": function_b},
        "gold_bit": pointer_trace(function_a, function_b, depth)[-1] % 2,
    }
    validate_episode(episode)
    return episode


def validate_episode(episode: dict) -> None:
    expected_keys = {
        "task", "version", "split", "master_seed", "episode_id", "size", "depth",
        "agent_a_view", "agent_b_view", "gold_bit",
    }
    if set(episode) != expected_keys:
        raise ValueError("episode has missing or unexpected top-level fields")
    if episode["task"] != TASK or episode["version"] != VERSION:
        raise ValueError("unsupported task or version")
    size, depth = episode["size"], episode["depth"]
    if type(size) is not int or size < 2 or type(depth) is not int or depth < 1:
        raise ValueError("size and depth are invalid")
    for role in ("agent_a_view", "agent_b_view"):
        view = episode[role]
        if set(view) != {"function_values_1_based"}:
            raise ValueError(f"{role} has missing or unexpected fields")
        values = view["function_values_1_based"]
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"{role} function must contain exactly n values")
        if any(type(value) is not int or not 1 <= value <= size for value in values):
            raise ValueError(f"{role} function values must be integers in [1,n]")
    if type(episode["gold_bit"]) is not int or episode["gold_bit"] not in (0, 1):
        raise ValueError("gold_bit must be 0 or 1")
    if episode["gold_bit"] != oracle_answer(episode):
        raise ValueError("gold_bit does not match the independently recomputed pointer chain")


def parse_exact_bit(output: str) -> int | None:
    value = output.strip()
    if value in {"0", "1"}:
        return int(value)
    return None


def score_episode(episode: dict, output: str) -> dict:
    validate_episode(episode)
    prediction = parse_exact_bit(output)
    return {
        "episode_id": episode["episode_id"],
        "prediction": prediction,
        "gold_bit": episode["gold_bit"],
        "exact": prediction is not None and prediction == episode["gold_bit"],
    }


def generate_shard(
    split: str, seed: int, sizes: list[int], depths: list[int], episodes_per_condition: int
) -> list[dict]:
    if not split:
        raise ValueError("split must be non-empty")
    if episodes_per_condition < 1:
        raise ValueError("episodes_per_condition must be positive")
    if not sizes or any(size < 2 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be unique integers greater than or equal to 2")
    if not depths or any(depth < 1 for depth in depths) or len(set(depths)) != len(depths):
        raise ValueError("depths must be unique positive integers")
    return [
        generate_episode(seed, split, size, depth, episode_index)
        for size in sizes
        for depth in depths
        for episode_index in range(episodes_per_condition)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "dev", "test", "pilot"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--sizes", required=True, nargs="+", type=int)
    parser.add_argument("--depths", required=True, nargs="+", type=int)
    parser.add_argument("--episodes-per-condition", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    episodes = generate_shard(args.split, args.seed, args.sizes, args.depths, args.episodes_per_condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(args.output), "episodes": len(episodes), "split": args.split,
        "sizes": args.sizes, "depths": args.depths,
        "episodes_per_condition": args.episodes_per_condition,
    }))


if __name__ == "__main__":
    main()
