"""Measure exact model-free payload baselines on Private Match episodes.

This is a channel/codec calibration, not an LLM benchmark: it reports wire
bytes and the ideal bit count, but does not estimate model tokens or inference
cost. Sender and receiver functions receive only their role-specific views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import zlib
from pathlib import Path
from typing import Any, Callable

from generate_tasks import SCHEMA_VERSION, generate_episode, score_answer


def _record_values(record: dict[str, str], names: list[str], vocabulary_size: int) -> tuple[int, ...]:
    values = []
    for name in names:
        value = record[name]
        match = re.fullmatch(r"v(\d+)", value)
        if match is None:
            raise ValueError(f"unexpected categorical value: {value!r}")
        number = int(match.group(1))
        if number >= vocabulary_size:
            raise ValueError(f"categorical value outside declared vocabulary: {value!r}")
        values.append(number)
    return tuple(values)


def _record_from_values(values: tuple[int, ...], names: list[str]) -> dict[str, str]:
    return {name: f"v{value:04d}" for name, value in zip(names, values)}


def _rank(values: tuple[int, ...], vocabulary_size: int) -> int:
    rank = 0
    multiplier = 1
    for value in values:
        rank += value * multiplier
        multiplier *= vocabulary_size
    return rank


def _unrank(rank: int, width: int, vocabulary_size: int) -> tuple[int, ...]:
    values = []
    for _ in range(width):
        values.append(rank % vocabulary_size)
        rank //= vocabulary_size
    if rank:
        raise ValueError("rank is outside the declared tuple space")
    return tuple(values)


def _lookup(receiver: dict[str, Any], target: dict[str, str]) -> str:
    matches = [row["candidate_id"] for row in receiver["candidates"] if row["record"] == target]
    if len(matches) != 1:
        raise ValueError("decoded target did not identify exactly one candidate")
    return matches[0]


def encode_labeled_text(sender: dict[str, Any]) -> str:
    record = sender["target_record"]
    fields = "; ".join(f"{name} is {record[name]}" for name in sender["feature_names"])
    return f"Match the candidate with this record: {fields}."


def decode_labeled_text(message: str, receiver: dict[str, Any]) -> str:
    found = dict(re.findall(r"\b(f\d+) is (v\d+)\b", message))
    names = receiver["feature_names"]
    if set(found) != set(names):
        raise ValueError("labeled-text message does not match the receiver schema")
    return _lookup(receiver, found)


def encode_json(sender: dict[str, Any]) -> str:
    return json.dumps(sender["target_record"], ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def decode_json(message: str, receiver: dict[str, Any]) -> str:
    target = json.loads(message)
    if not isinstance(target, dict) or set(target) != set(receiver["feature_names"]):
        raise ValueError("JSON message does not match the receiver schema")
    return _lookup(receiver, target)


def encode_delimited(sender: dict[str, Any]) -> str:
    return ",".join(sender["target_record"][name] for name in sender["feature_names"])


def decode_delimited(message: str, receiver: dict[str, Any]) -> str:
    values = message.split(",")
    names = receiver["feature_names"]
    if len(values) != len(names) or any(re.fullmatch(r"v\d+", v) is None for v in values):
        raise ValueError("delimited tuple does not match the receiver schema")
    return _lookup(receiver, dict(zip(names, values)))


def encode_rank_bytes(sender: dict[str, Any], vocabulary_size: int) -> bytes:
    values = _record_values(sender["target_record"], sender["feature_names"], vocabulary_size)
    bit_width = math.ceil(math.log2(vocabulary_size ** len(values)))
    byte_width = (bit_width + 7) // 8
    return _rank(values, vocabulary_size).to_bytes(byte_width, "big")


def decode_rank_bytes(message: bytes, receiver: dict[str, Any], vocabulary_size: int) -> str:
    names = receiver["feature_names"]
    bit_width = math.ceil(math.log2(vocabulary_size ** len(names)))
    byte_width = (bit_width + 7) // 8
    if len(message) != byte_width:
        raise ValueError("rank payload has an unexpected byte width")
    rank = int.from_bytes(message, "big")
    if rank >= vocabulary_size ** len(names):
        raise ValueError("rank payload encodes an unused code point")
    target = _record_from_values(_unrank(rank, len(names), vocabulary_size), names)
    return _lookup(receiver, target)


def _payload_metrics(payload: str | bytes) -> dict[str, int]:
    if isinstance(payload, str):
        wire = payload.encode("utf-8")
    else:
        wire = payload
    return {"payload_bytes": len(wire), "payload_bits": len(wire) * 8}


def _summarize(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "max": max(values),
    }


def _shared_compression_dictionary(feature_count: int, vocabulary_size: int) -> bytes:
    pieces = [
        "Match the candidate with this record",
        "record",
        " is ",
        "; ",
        '"',
        '":"',
    ]
    for index in range(feature_count):
        pieces.extend((f"f{index}", f"f{index} is"))
    pieces.extend(f"v{value:04d}" for value in range(vocabulary_size))
    return " ".join(pieces).encode("utf-8")


def _compress_stream(messages: list[bytes], dictionary: bytes | None = None) -> bytes:
    kwargs: dict[str, Any] = {}
    if dictionary is not None:
        kwargs["zdict"] = dictionary
    compressor = zlib.compressobj(level=9, **kwargs)
    output = []
    for message in messages:
        # Newline is an in-stream record separator; Z_SYNC_FLUSH makes every
        # completed record visible to a streaming decoder without ending the
        # shared compression context.
        output.append(compressor.compress(message + b"\n"))
        output.append(compressor.flush(zlib.Z_SYNC_FLUSH))
    output.append(compressor.flush(zlib.Z_FINISH))
    return b"".join(output)


def run_comparison(
    *, episodes: int, seed: int, candidate_count: int,
    feature_count: int, vocabulary_size: int,
) -> dict[str, Any]:
    for name, value in (("episodes", episodes), ("candidate_count", candidate_count),
                        ("feature_count", feature_count), ("vocabulary_size", vocabulary_size)):
        minimum = 1 if name in ("episodes", "feature_count") else 2
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    arms: dict[str, tuple[Callable[..., str | bytes], Callable[..., str], str]] = {
        "labeled_natural_language": (encode_labeled_text, decode_labeled_text, "utf-8 text"),
        "compact_json": (encode_json, decode_json, "utf-8 text"),
        "ordered_delimited_tuple": (encode_delimited, decode_delimited, "utf-8 text"),
    }
    aggregate: dict[str, dict[str, Any]] = {
        name: {"correct": 0, "payload_bytes": [], "payload_bits": [], "serialization": kind}
        for name, (_, _, kind) in arms.items()
    }
    aggregate["fixed_width_mixed_radix_rank"] = {
        "correct": 0, "payload_bytes": [], "payload_bits": [], "serialization": "packed binary bytes"
    }
    messages_by_arm: dict[str, list[bytes]] = {name: [] for name in aggregate}
    record_digests = hashlib.sha256()

    for index in range(episodes):
        sender, receiver, gold = generate_episode(
            episode_id=f"pm-{index:06d}", seed=seed + index,
            candidate_count=candidate_count, feature_count=feature_count,
            vocabulary_size=vocabulary_size,
        )
        # Hash only the independently generated public-role inputs and scorer
        # reference so this aggregate can be tied to an exact task sequence.
        for role_view in (sender, receiver, gold):
            record_digests.update(json.dumps(role_view, sort_keys=True, separators=(",", ":")).encode())
            record_digests.update(b"\n")
        for name, (encoder, decoder, _) in arms.items():
            payload = encoder(sender)
            answer = decoder(payload, receiver)
            metrics = _payload_metrics(payload)
            aggregate[name]["correct"] += int(score_answer(receiver, gold, answer))
            aggregate[name]["payload_bytes"].append(metrics["payload_bytes"])
            aggregate[name]["payload_bits"].append(metrics["payload_bits"])
            messages_by_arm[name].append(payload.encode("utf-8"))

        rank_payload = encode_rank_bytes(sender, vocabulary_size)
        rank_answer = decode_rank_bytes(rank_payload, receiver, vocabulary_size)
        rank_metrics = _payload_metrics(rank_payload)
        rank_arm = aggregate["fixed_width_mixed_radix_rank"]
        rank_arm["correct"] += int(score_answer(receiver, gold, rank_answer))
        rank_arm["payload_bytes"].append(rank_metrics["payload_bytes"])
        rank_arm["payload_bits"].append(rank_metrics["payload_bits"])
        messages_by_arm["fixed_width_mixed_radix_rank"].append(rank_payload)

    rows = {}
    ideal_bits = math.ceil(math.log2(vocabulary_size ** feature_count))
    for name, arm in aggregate.items():
        rows[name] = {
            "correct": arm["correct"],
            "episodes": episodes,
            "accuracy": arm["correct"] / episodes,
            "serialization": arm["serialization"],
            "payload_bytes": _summarize(arm["payload_bytes"]),
            "payload_bits": _summarize(arm["payload_bits"]),
            "ideal_worst_case_zero_error_bits": ideal_bits,
            "llm_tokens_measured": False,
            "llm_inference_cost_measured": False,
        }
    dictionary = _shared_compression_dictionary(feature_count, vocabulary_size)
    for name, messages in messages_by_arm.items():
        framed_messages = b"".join(message + b"\n" for message in messages)
        independent_streams = [zlib.compress(message, level=9) for message in messages]
        if any(zlib.decompress(compressed) != message for compressed, message in zip(independent_streams, messages)):
            raise RuntimeError(f"independent zlib codec failed round-trip for arm {name}")
        no_dictionary_stream = _compress_stream(messages)
        with_dictionary_stream = _compress_stream(messages, dictionary)
        if zlib.decompress(no_dictionary_stream) != framed_messages:
            raise RuntimeError(f"persistent zlib codec failed round-trip for arm {name}")
        dictionary_decoder = zlib.decompressobj(zdict=dictionary)
        decoded_with_dictionary = dictionary_decoder.decompress(with_dictionary_stream) + dictionary_decoder.flush()
        if decoded_with_dictionary != framed_messages:
            raise RuntimeError(f"dictionary zlib codec failed round-trip for arm {name}")
        no_dictionary_stream_bytes = len(no_dictionary_stream)
        with_dictionary_stream_bytes = len(with_dictionary_stream)
        dictionary_saving = no_dictionary_stream_bytes - with_dictionary_stream_bytes
        per_message_saving = dictionary_saving / episodes
        rows[name]["zlib_level_9"] = {
            "independent_stream_total_bytes": sum(map(len, independent_streams)),
            "persistent_stream_total_bytes": no_dictionary_stream_bytes,
            "persistent_stream_bytes_per_message": no_dictionary_stream_bytes / episodes,
            "message_delimiter": "LF byte included in compressed stream",
            "flush_per_message": "Z_SYNC_FLUSH",
            "shared_dictionary": {
                "dictionary_bytes": len(dictionary),
                "dictionary_sha256": hashlib.sha256(dictionary).hexdigest(),
                "compressed_stream_total_bytes": with_dictionary_stream_bytes,
                "bytes_if_dictionary_transmitted_at_this_horizon": len(dictionary) + with_dictionary_stream_bytes,
                "net_savings_vs_no_dictionary_at_this_horizon": dictionary_saving - len(dictionary),
                "linear_extrapolation_break_even_messages": (
                    math.ceil(len(dictionary) / per_message_saving)
                    if per_message_saving > 0 else None
                ),
            },
        }
    return {
        "schema_version": "tlu.private-match-codec-comparison.v2",
        "task_schema_version": SCHEMA_VERSION,
        "task_parameters": {
            "episodes": episodes,
            "seed_start": seed,
            "candidate_count": candidate_count,
            "feature_count": feature_count,
            "vocabulary_size_per_feature": vocabulary_size,
        },
        "task_sequence_sha256": record_digests.hexdigest(),
        "no_message_bayes_accuracy": 1 / candidate_count,
        "centralized_exact_information_accuracy": 1.0,
        "compression_runtime": {
            "python_zlib_compile_version": zlib.ZLIB_VERSION,
            "python_zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            "compression_level": 9,
            "dictionary_bytes_in_compressed_stream_totals": False,
            "dictionary_wire_bytes_reported_separately": True,
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument("--vocabulary-size", type=int, default=16)
    args = parser.parse_args()
    result = run_comparison(
        episodes=args.episodes, seed=args.seed, candidate_count=args.candidates,
        feature_count=args.features, vocabulary_size=args.vocabulary_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
