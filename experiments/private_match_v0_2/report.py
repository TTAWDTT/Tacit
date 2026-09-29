"""Summarize Private Match v0.2 task, message-fidelity, and paired outcomes."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cost_report import RecordError, aggregate, read_jsonl
from tools.paired_report import paired_report


SCHEMA_VERSION = "tlu.private-match-report.v1"
DIAGNOSTIC_FIELDS = (
    "answer_is_candidate_id",
    "message_format_valid",
    "message_semantic_fidelity",
    "sender_truncated",
    "receiver_truncated",
)


def _stratum_key(stratum: dict[str, Any]) -> str:
    return json.dumps(stratum, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _task_stratum(record: dict[str, Any]) -> dict[str, Any]:
    stratum = dict(record["_normalized_stratum"])
    stratum.pop("model_population_id", None)
    stratum.pop("agent_models", None)
    return stratum


def _protocol_key(record: dict[str, Any]) -> tuple[str, str, str]:
    protocol = record["protocol"]
    return protocol["policy_id"], protocol["code_id"], protocol["decoder_id"]


def _rate(values: list[bool | None]) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    positives = sum(present)
    return {
        "positive": positives,
        "observed": len(present),
        "missing": len(values) - len(present),
        "rate": positives / len(present) if present else None,
    }


def _quantile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _bootstrap(differences: list[float], *, replicates: int, seed: int) -> dict[str, Any]:
    if not differences:
        return {"paired_episodes": 0, "mean_left_minus_right": None, "ci95_low": None, "ci95_high": None}
    mean = sum(differences) / len(differences)
    if len(differences) == 1:
        low = high = mean
    else:
        rng = random.Random(seed)
        samples = [
            sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
            for _ in range(replicates)
        ]
        samples.sort()
        low, high = _quantile(samples, 0.025), _quantile(samples, 0.975)
    return {
        "paired_episodes": len(differences),
        "mean_left_minus_right": mean,
        "ci95_low": low,
        "ci95_high": high,
    }


def private_match_report(
    records: list[dict[str, Any]], *, replicates: int = 5000, seed: int = 1729,
) -> dict[str, Any]:
    if replicates < 100:
        raise RecordError("bootstrap replicates must be at least 100")
    canonical_groups = aggregate(records)
    if canonical_groups["input_schema_version"] != "tlu.costs.v3":
        raise RecordError("Private Match v0.2 reports require tlu.costs.v3 records")
    for row in records:
        stratum = row.get("_normalized_stratum")
        if not isinstance(stratum, dict) or stratum.get("experiment_id") != "private-match-v0.2":
            raise RecordError("input contains a record outside private-match-v0.2")
        diagnostic = row.get("diagnostics")
        if not isinstance(diagnostic, dict):
            raise RecordError(f"episode {row['episode_id']}: missing diagnostics object")
        generation_seed = diagnostic.get("generation_seed")
        if isinstance(generation_seed, bool) or not isinstance(generation_seed, int) or generation_seed < 0:
            raise RecordError(f"episode {row['episode_id']}: diagnostics.generation_seed must be a non-negative integer")
        for name in ("message_text", "answer_text"):
            value = diagnostic.get(name)
            if value is not None and not isinstance(value, str):
                raise RecordError(f"episode {row['episode_id']}: diagnostics.{name} must be text or null")
        for name in DIAGNOSTIC_FIELDS:
            value = diagnostic.get(name)
            if value is not None and not isinstance(value, bool):
                raise RecordError(f"episode {row['episode_id']}: diagnostics.{name} must be boolean or null")
        if diagnostic.get("message_semantic_fidelity") is True and diagnostic.get("message_format_valid") is not True:
            raise RecordError(f"episode {row['episode_id']}: faithful messages must also pass format validation")

    summaries: dict[tuple[str, tuple[str, str, str]], list[dict[str, Any]]] = defaultdict(list)
    paired_conditions: dict[tuple[str, tuple[str, str, str], str], dict[str, dict[str, Any]]] = defaultdict(dict)
    task_strata: dict[str, dict[str, Any]] = {}
    full_strata: dict[str, dict[str, Any]] = {}
    for record in records:
        task_stratum = _task_stratum(record)
        task_key = _stratum_key(task_stratum)
        full_stratum = record["_normalized_stratum"]
        full_key = _stratum_key(full_stratum)
        protocol_key = _protocol_key(record)
        task_strata[task_key] = task_stratum
        full_strata[full_key] = full_stratum
        summaries[(full_key, protocol_key)].append(record)
        episode_id = record["episode_id"]
        condition_key = (task_key, protocol_key, full_key)
        if episode_id in paired_conditions[condition_key]:
            raise RecordError(f"duplicate episode_id {episode_id!r} within a condition")
        paired_conditions[condition_key][episode_id] = record

    condition_reports = []
    for (full_key, protocol_key), rows in sorted(summaries.items()):
        diag = [row["diagnostics"] for row in rows]
        messages = [item.get("message_text") for item in diag]
        text_bytes = [len(message.encode("utf-8")) for message in messages if isinstance(message, str)]
        condition_reports.append({
            "stratum": full_strata[full_key],
            "protocol": dict(zip(("policy_id", "code_id", "decoder_id"), protocol_key)),
            "episodes": len(rows),
            "task": {
                "joint_successes": sum(row["outcome"]["joint_success"] for row in rows),
                "joint_success_rate": sum(row["outcome"]["joint_success"] for row in rows) / len(rows),
            },
            "diagnostics": {name: _rate([item.get(name) for item in diag]) for name in DIAGNOSTIC_FIELDS},
            "message_text_utf8_bytes": {
                "observed": len(text_bytes),
                "missing_or_no_message": len(rows) - len(text_bytes),
                "mean": sum(text_bytes) / len(text_bytes) if text_bytes else None,
                "max": max(text_bytes) if text_bytes else None,
            },
        })

    groups: dict[str, list[tuple[tuple[str, str, str], str]]] = defaultdict(list)
    for task_key, protocol_key, full_key in paired_conditions:
        groups[task_key].append((protocol_key, full_key))
    paired_reports = []
    metric_index = 0
    for task_key in sorted(groups):
        conditions = sorted(set(groups[task_key]))
        for left_condition, right_condition in itertools.combinations(conditions, 2):
            left_protocol, left_full = left_condition
            right_protocol, right_full = right_condition
            left = paired_conditions[(task_key, left_protocol, left_full)]
            right = paired_conditions[(task_key, right_protocol, right_full)]
            common_ids = sorted(set(left) & set(right))
            metric_results: dict[str, Any] = {}
            for metric in ("joint_success", *DIAGNOSTIC_FIELDS):
                deltas = []
                missing = 0
                for episode_id in common_ids:
                    if metric == "joint_success":
                        left_value = left[episode_id]["outcome"]["joint_success"]
                        right_value = right[episode_id]["outcome"]["joint_success"]
                    else:
                        left_value = left[episode_id]["diagnostics"].get(metric)
                        right_value = right[episode_id]["diagnostics"].get(metric)
                    if left_value is None or right_value is None:
                        missing += 1
                    else:
                        deltas.append(float(left_value) - float(right_value))
                metric_results[metric] = {
                    **_bootstrap(deltas, replicates=replicates, seed=seed + metric_index),
                    "missing_pairs": missing,
                }
                metric_index += 1
            paired_reports.append({
                "task_stratum": task_strata[task_key],
                "left_stratum": full_strata[left_full],
                "right_stratum": full_strata[right_full],
                "model_strata_matched": full_strata[left_full] == full_strata[right_full],
                "left_protocol": dict(zip(("policy_id", "code_id", "decoder_id"), left_protocol)),
                "right_protocol": dict(zip(("policy_id", "code_id", "decoder_id"), right_protocol)),
                "left_only_episodes": len(set(left) - set(right)),
                "right_only_episodes": len(set(right) - set(left)),
                "paired_episode_count": len(common_ids),
                "metrics": metric_results,
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "bootstrap": {"method": "paired episode percentile bootstrap", "replicates": replicates, "seed": seed},
        "conditions": condition_reports,
        "paired_comparisons": paired_reports,
        "cost_and_success_comparisons": paired_report(records, replicates=replicates, seed=seed)["comparisons"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Private Match v0.2 per-episode tlu.costs.v3 JSONL")
    parser.add_argument("-o", "--output", type=Path, help="write JSON report (default: stdout)")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    try:
        report = private_match_report(read_jsonl(args.input), replicates=args.replicates, seed=args.seed)
    except (OSError, RecordError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
