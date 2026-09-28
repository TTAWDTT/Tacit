"""Run a small model-free scaling sweep for Private Match codec baselines."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

from compare_codecs import run_comparison
from generate_tasks import generate_episode


DEFAULT_FEATURE_COUNTS = (1, 2, 4, 8)
DEFAULT_VOCABULARY_SIZES = (2, 4, 16, 256)
DEFAULT_CANDIDATE_COUNTS = (2, 8, 32, 128)


def _compact_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def run_scaling_sweep(
    *, episodes: int = 32, seed: int = 9000,
    feature_counts: tuple[int, ...] = DEFAULT_FEATURE_COUNTS,
    vocabulary_sizes: tuple[int, ...] = DEFAULT_VOCABULARY_SIZES,
    candidate_counts: tuple[int, ...] = DEFAULT_CANDIDATE_COUNTS,
) -> dict[str, Any]:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be an integer >= 1")
    for name, values in (("feature_counts", feature_counts),
                         ("vocabulary_sizes", vocabulary_sizes),
                         ("candidate_counts", candidate_counts)):
        if not values or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
            raise ValueError(f"{name} must be a non-empty sequence of positive integers")
    if any(value < 2 for value in vocabulary_sizes) or any(value < 2 for value in candidate_counts):
        raise ValueError("vocabulary sizes and candidate counts must be >= 2")

    cells = []
    skipped = []
    sweep_hash = hashlib.sha256()
    cell_index = 0
    for feature_count, vocabulary_size, candidate_count in itertools.product(
        feature_counts, vocabulary_sizes, candidate_counts
    ):
        if vocabulary_size ** feature_count < candidate_count:
            skipped.append({
                "feature_count": feature_count,
                "vocabulary_size": vocabulary_size,
                "candidate_count": candidate_count,
                "reason": "fewer distinct tuples than requested receiver candidates",
            })
            continue
        cell_seed = seed + cell_index * 100_000
        codec = run_comparison(
            episodes=episodes, seed=cell_seed, candidate_count=candidate_count,
            feature_count=feature_count, vocabulary_size=vocabulary_size,
        )
        sender_bytes = []
        receiver_bytes = []
        for episode_index in range(episodes):
            sender, receiver, gold = generate_episode(
                episode_id=f"pm-{episode_index:06d}", seed=cell_seed + episode_index,
                candidate_count=candidate_count, feature_count=feature_count,
                vocabulary_size=vocabulary_size,
            )
            sender_bytes.append(_compact_bytes(sender))
            receiver_bytes.append(_compact_bytes(receiver))
            for role_view in (sender, receiver, gold):
                sweep_hash.update(json.dumps(
                    role_view, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"))
                sweep_hash.update(b"\n")
        rank_bits = math.ceil(math.log2(vocabulary_size ** feature_count))
        cells.append({
            "feature_count": feature_count,
            "vocabulary_size": vocabulary_size,
            "candidate_count": candidate_count,
            "episodes": episodes,
            "cell_seed_start": cell_seed,
            "no_message_bayes_accuracy": 1 / candidate_count,
            "centralized_exact_information_accuracy": 1.0,
            "ideal_zero_error_rank_bits": rank_bits,
            "rank_payload_bytes": math.ceil(rank_bits / 8),
            "mean_sender_role_input_bytes": sum(sender_bytes) / episodes,
            "mean_receiver_role_input_bytes": sum(receiver_bytes) / episodes,
            "codec_results": codec["results"],
            "task_sequence_sha256": codec["task_sequence_sha256"],
        })
        cell_index += 1

    return {
        "schema_version": "tlu.private-match-scaling-sweep.v1",
        "task_schema_version": "tlu.private-match.v1",
        "sweep_parameters": {
            "episodes_per_cell": episodes,
            "seed_start": seed,
            "feature_counts": list(feature_counts),
            "vocabulary_sizes": list(vocabulary_sizes),
            "candidate_counts": list(candidate_counts),
            "valid_cell_count": len(cells),
            "skipped_cell_count": len(skipped),
        },
        "aggregate_task_sequence_sha256": sweep_hash.hexdigest(),
        "scaling_claim_boundary": (
            "Exact wire bytes and serialized role-view JSON bytes only; model-native tokens, inference, "
            "semantic decoding difficulty, and multi-agent scaling are not measured."
        ),
        "cells": cells,
        "skipped_cells": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9000)
    args = parser.parse_args()
    report = run_scaling_sweep(episodes=args.episodes, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "schema_version": report["schema_version"],
        "sweep_parameters": report["sweep_parameters"],
        "aggregate_task_sequence_sha256": report["aggregate_task_sequence_sha256"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
