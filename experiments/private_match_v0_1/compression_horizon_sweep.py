"""Measure exact dictionary setup amortization over longer message horizons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zlib
from pathlib import Path
from typing import Any

from compare_codecs import (
    _compress_stream,
    _shared_compression_dictionary,
    encode_delimited,
    encode_json,
    encode_labeled_text,
    encode_rank_bytes,
)
from generate_tasks import SCHEMA_VERSION, generate_episode


CODECS = {
    "labeled_natural_language_template": lambda sender: encode_labeled_text(sender).encode("utf-8"),
    "compact_json": lambda sender: encode_json(sender).encode("utf-8"),
    "ordered_delimited_tuple": lambda sender: encode_delimited(sender).encode("utf-8"),
    "fixed_width_mixed_radix_rank": lambda sender: encode_rank_bytes(sender, 16),
}
DEFAULT_SEEDS = (12000, 22000, 32000, 42000, 52000)
DEFAULT_HORIZONS = (1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384)


def _validate(seeds: tuple[int, ...], horizons: tuple[int, ...]) -> None:
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be a non-empty tuple of non-negative integers")
    if not horizons or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("horizons must be a non-empty tuple of positive integers")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be strictly increasing")


def _round_trip(messages: list[bytes], dictionary: bytes | None) -> int:
    stream = _compress_stream(messages, dictionary)
    if dictionary is None:
        decoded = zlib.decompress(stream)
    else:
        decoder = zlib.decompressobj(zdict=dictionary)
        decoded = decoder.decompress(stream) + decoder.flush()
    expected = b"".join(message + b"\n" for message in messages)
    if decoded != expected:
        raise RuntimeError("persistent zlib stream failed exact round-trip")
    return len(stream)


def run_sweep(
    *, seeds: tuple[int, ...] = DEFAULT_SEEDS,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    _validate(seeds, horizons)
    max_horizon = max(horizons)
    dictionary = _shared_compression_dictionary(feature_count=5, vocabulary_size=16)
    seed_rows: list[dict[str, Any]] = []

    for seed in seeds:
        messages = {name: [] for name in CODECS}
        task_hash = hashlib.sha256()
        for index in range(max_horizon):
            sender, receiver, gold = generate_episode(
                episode_id=f"pm-horizon-{index:06d}", seed=seed + index,
                candidate_count=8, feature_count=5, vocabulary_size=16,
            )
            for role_view in (sender, receiver, gold):
                task_hash.update(json.dumps(role_view, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                task_hash.update(b"\n")
            for codec_name, encoder in CODECS.items():
                messages[codec_name].append(encoder(sender))

        horizon_rows = []
        crossings: dict[str, list[int]] = {name: [] for name in CODECS}
        for horizon in horizons:
            for codec_name, all_messages in messages.items():
                prefix = all_messages[:horizon]
                base_bytes = _round_trip(prefix, None)
                dict_stream_bytes = _round_trip(prefix, dictionary)
                setup_bytes = len(dictionary)
                net_savings = base_bytes - (setup_bytes + dict_stream_bytes)
                raw_bytes = sum(map(len, prefix))
                horizon_rows.append({
                    "horizon": horizon,
                    "codec": codec_name,
                    "raw_payload_bytes": raw_bytes,
                    "no_dictionary_stream_bytes": base_bytes,
                    "dictionary_setup_bytes": setup_bytes,
                    "dictionary_stream_bytes": dict_stream_bytes,
                    "dictionary_total_bytes": setup_bytes + dict_stream_bytes,
                    "net_savings_bytes_vs_no_dictionary": net_savings,
                    "dictionary_wins": net_savings > 0,
                })
                if net_savings > 0:
                    crossings[codec_name].append(horizon)

        seed_rows.append({
            "seed": seed,
            "episodes": max_horizon,
            "task_sequence_sha256": task_hash.hexdigest(),
            "dictionary_sha256": hashlib.sha256(dictionary).hexdigest(),
            "horizons": horizon_rows,
            "first_observed_break_even_horizon": {
                name: (min(points) if points else None) for name, points in crossings.items()
            },
        })

    pooled = []
    for horizon in horizons:
        for codec_name in CODECS:
            matching = [
                row for seed_row in seed_rows for row in seed_row["horizons"]
                if row["horizon"] == horizon and row["codec"] == codec_name
            ]
            deltas = [row["net_savings_bytes_vs_no_dictionary"] for row in matching]
            sorted_deltas = sorted(deltas)
            p95_index = max(0, math.ceil(0.95 * len(sorted_deltas)) - 1)
            pooled.append({
                "horizon": horizon,
                "codec": codec_name,
                "seed_count": len(matching),
                "net_savings_bytes_mean": statistics.fmean(deltas),
                "net_savings_bytes_median": statistics.median(deltas),
                "net_savings_bytes_p95": sorted_deltas[p95_index],
                "seeds_dictionary_wins": sum(delta > 0 for delta in deltas),
            })

    return {
        "schema_version": "tlu.private-match-compression-horizon.v1",
        "task_schema_version": SCHEMA_VERSION,
        "task_parameters": {
            "candidate_count": 8,
            "feature_count": 5,
            "vocabulary_size": 16,
            "episodes_per_seed": max_horizon,
            "seeds": list(seeds),
            "horizons": list(horizons),
        },
        "zlib": {"level": 9, "runtime_version": zlib.ZLIB_RUNTIME_VERSION},
        "dictionary": {
            "bytes": len(dictionary),
            "sha256": hashlib.sha256(dictionary).hexdigest(),
            "setup_transfer_assumption": "one dictionary copy sent once to one receiver per independent stream",
        },
        "seed_results": seed_rows,
        "pooled_results": pooled,
        "llm_tokens_measured": False,
        "llm_inference_cost_measured": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    args = parser.parse_args()
    report = run_sweep(seeds=tuple(args.seeds), horizons=tuple(args.horizons))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "seed_count": len(args.seeds), "max_horizon": max(args.horizons)}))


if __name__ == "__main__":
    main()
