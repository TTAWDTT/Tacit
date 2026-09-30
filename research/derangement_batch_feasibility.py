"""Model-free exact feasibility scan for message-association replay batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.emergent_ood_v0_4.replay_usage_messages import _uniform_perfect_matching
from experiments.emergent_ood_v0_4.runner import load_episode_bundle, select_candidate_sets


def compatibility_graph(
    episodes: Sequence[dict[str, Any]], attributes: Sequence[str],
) -> dict[str, list[str]]:
    """Build candidate-disjoint foreign-donor edges from aligned task rows."""
    targets: dict[str, tuple[str, ...]] = {}
    candidate_tables: dict[str, set[tuple[str, ...]]] = {}
    for episode in episodes:
        episode_id = episode.get("gold", {}).get("episode_id")
        sender_meaning = episode.get("sender", {}).get("private_meaning")
        receiver = episode.get("receiver", {})
        candidates = receiver.get("candidates")
        if not isinstance(episode_id, str) or episode_id in targets:
            raise ValueError("episodes need unique evaluator IDs")
        if not isinstance(sender_meaning, Mapping) or set(sender_meaning) != set(attributes):
            raise ValueError("sender meaning differs from the declared task ontology")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("receiver candidate table is missing")
        targets[episode_id] = tuple(sender_meaning[axis] for axis in attributes)
        candidate_tables[episode_id] = {
            tuple(candidate["attributes"][axis] for axis in attributes)
            for candidate in candidates
        }
        if len(candidate_tables[episode_id]) != len(candidates):
            raise ValueError("receiver candidate meanings must be unique")
        if targets[episode_id] not in candidate_tables[episode_id]:
            raise ValueError("sender meaning is absent from its receiver candidate table")
    return {
        recipient: [
            donor for donor, meaning in targets.items()
            if donor != recipient and meaning not in candidate_tables[recipient]
        ]
        for recipient in targets
    }


def _window_record(
    bundle: dict[str, Any],
    split: dict[str, Any],
    *,
    stage: str,
    batch_sets: int,
    set_offset: int,
) -> dict[str, Any]:
    episodes = select_candidate_sets(bundle, stage, batch_sets, set_offset=set_offset)
    graph = compatibility_graph(episodes, split["attributes"])
    minimum_degree = min(map(len, graph.values()))
    try:
        _, matching_count = _uniform_perfect_matching(
            graph, list(graph), random.Random(0),
        )
    except ValueError as exc:
        if "no complete compatible message derangement" not in str(exc):
            raise
        matching_count = 0
    return {
        "set_offset": set_offset,
        "candidate_sets": batch_sets,
        "episodes": len(episodes),
        "minimum_receiver_degree": minimum_degree,
        "compatible_perfect_matchings": matching_count,
        "feasible": matching_count > 0,
    }


def analyze_bundle(
    input_dir: Path,
    *,
    split_seed: int,
    stage: str,
    batch_sets: int,
) -> dict[str, Any]:
    bundle, split = load_episode_bundle(input_dir, split_seed=split_seed)
    sets_per_stage = bundle["manifest"]["sets_per_stage"]
    if isinstance(batch_sets, bool) or not isinstance(batch_sets, int) or batch_sets < 1:
        raise ValueError("batch_sets must be a positive integer")
    if batch_sets > sets_per_stage:
        raise ValueError("batch_sets exceeds the generated candidate-set count")
    offsets = range(sets_per_stage - batch_sets + 1)
    rolling = [
        _window_record(
            bundle, split, stage=stage, batch_sets=batch_sets, set_offset=offset,
        )
        for offset in offsets
    ]
    canonical_offsets = list(range(0, sets_per_stage - batch_sets + 1, batch_sets))
    canonical = [
        next(record for record in rolling if record["set_offset"] == offset)
        for offset in canonical_offsets
    ]
    manifest_bytes = (input_dir / "manifest.json").read_bytes()
    return {
        "schema": "tlu.derangement-batch-feasibility.v0.1",
        "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "split_sha256": bundle["manifest"]["split_sha256"],
        "split_seed": split_seed,
        "task_seed": bundle["manifest"]["task_seed"],
        "stage": stage,
        "candidate_count_k": bundle["manifest"]["k"],
        "sets_per_stage": sets_per_stage,
        "batch_sets": batch_sets,
        "rolling_windows": rolling,
        "canonical_non_overlapping_offsets": canonical_offsets,
        "canonical_schedule_feasible": bool(canonical) and all(
            item["feasible"] for item in canonical
        ),
        "canonical_covered_sets": len(canonical) * batch_sets,
        "uncovered_tail_sets": sets_per_stage - len(canonical) * batch_sets,
        "model_loaded": False,
        "inference_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count candidate-disjoint message derangements before model runs."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--stage", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-sets", type=int, default=3)
    args = parser.parse_args()
    try:
        report = analyze_bundle(
            args.input_dir, split_seed=args.split_seed,
            stage=args.stage, batch_sets=args.batch_sets,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
