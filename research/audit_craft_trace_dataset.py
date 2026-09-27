"""Audit generated Director posts against per-role Builder transcript fields."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


GROUP_COLUMNS = ["director_model", "structure_id", "run"]
MESSAGE_COLUMNS = ["D1_message", "D2_message", "D3_message"]
DIRECTOR_PREFIXES = ("D1:", "D2:", "D3:")


def parse_snapshot(value: Any) -> list[str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        return None
    return parsed


def audit(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    required = {*GROUP_COLUMNS, "turn_number", "conversation_snapshot", *MESSAGE_COLUMNS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    measurements: list[dict[str, Any]] = []
    missing_snapshots = 0
    unaligned_snapshots = 0
    for keys, game in frame.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        previous: list[str] | None = None
        game = game.sort_values("turn_number")
        for _, row in game.iterrows():
            current = parse_snapshot(row["conversation_snapshot"])
            if current is None:
                missing_snapshots += 1
                previous = None
                continue

            turn = int(row["turn_number"])
            if previous is None:
                if turn != 1:
                    unaligned_snapshots += 1
                    previous = current
                    continue
                prior_context: list[str] = []
            else:
                # The upstream runner records the snapshot, then trims history
                # from >50 entries to its last 40 before the next turn.
                prior_context = previous[-40:] if len(previous) > 50 else previous
                if current[: len(prior_context)] != prior_context:
                    unaligned_snapshots += 1
                    previous = current
                    continue

            new_items = current[len(prior_context) :]
            generated_posts = sum(item.startswith(DIRECTOR_PREFIXES) for item in new_items)
            builder_messages = sum(
                isinstance(row[column], str) and bool(row[column].strip())
                for column in MESSAGE_COLUMNS
            )
            measurements.append(
                {
                    "director_model": keys[0],
                    "turn_number": turn,
                    "generated_posts": generated_posts,
                    "builder_transcript_messages": builder_messages,
                    "collapsed_posts": max(0, generated_posts - builder_messages),
                }
            )
            previous = current

    generated_hist = Counter(row["generated_posts"] for row in measurements)
    transcript_hist = Counter(row["builder_transcript_messages"] for row in measurements)
    collapse_hist = Counter(
        row["generated_posts"] - row["builder_transcript_messages"]
        for row in measurements
    )
    positive_turns = [row for row in measurements if row["collapsed_posts"] > 0]
    generated_total = sum(row["generated_posts"] for row in measurements)
    collapsed_total = sum(row["collapsed_posts"] for row in measurements)
    per_model: dict[str, Any] = {}
    for model in sorted({row["director_model"] for row in measurements}):
        rows = [row for row in measurements if row["director_model"] == model]
        per_model[model] = {
            "aligned_turns": len(rows),
            "turns_with_collapsed_posts": sum(row["collapsed_posts"] > 0 for row in rows),
            "collapsed_posts": sum(row["collapsed_posts"] for row in rows),
            "mean_generated_posts": round(
                sum(row["generated_posts"] for row in rows) / len(rows), 4
            ),
            "mean_builder_transcript_messages": round(
                sum(row["builder_transcript_messages"] for row in rows) / len(rows), 4
            ),
        }

    return {
        "dataset_rows": len(frame),
        "games": int(frame.groupby(GROUP_COLUMNS, dropna=False).ngroups),
        "aligned_turns": len(measurements),
        "missing_snapshots": missing_snapshots,
        "unaligned_snapshots": unaligned_snapshots,
        "generated_post_histogram": dict(sorted(generated_hist.items())),
        "builder_transcript_message_histogram": dict(sorted(transcript_hist.items())),
        "generated_minus_builder_histogram": dict(sorted(collapse_hist.items())),
        "turns_with_at_least_one_collapsed_post": len(positive_turns),
        "collapsed_posts": collapsed_total,
        "generated_posts_in_aligned_turns": generated_total,
        "collapsed_share_of_aligned_generated_posts": round(
            collapsed_total / generated_total if generated_total else 0.0, 6
        ),
        "per_model": per_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path, help="Path to the public CRAFT Parquet file")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    result = audit(args.parquet)
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
