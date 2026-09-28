"""Paired bootstrap comparisons for TLU per-episode cost records."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .cost_report import RecordError, aggregate, read_jsonl
except ImportError:  # Direct ``python tools/paired_report.py`` invocation.
    from cost_report import RecordError, aggregate, read_jsonl


REPORT_SCHEMA_VERSION = "tlu.paired-report.v1"
PROTOCOL_FIELDS = ("policy_id", "code_id", "decoder_id")
BOOTSTRAP_METRICS = (
    "joint_success",
    "wire_bytes",
    "transmissions",
    "input_tokens",
    "output_tokens",
    "model_calls",
    "service_seconds",
    "wall_seconds",
    "critical_path_seconds",
    "amortized_setup_bytes",
)


def _stratum_key(record: dict[str, Any]) -> str:
    stratum = record.get("_normalized_stratum")
    return "" if stratum is None else json.dumps(
        stratum, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _task_stratum(record: dict[str, Any]) -> dict[str, Any] | None:
    stratum = record.get("_normalized_stratum")
    if stratum is None:
        return None
    # Model population is a comparison factor. Match the task/split/scorer, then
    # report both model strata so cross-model comparisons cannot masquerade as
    # a representation-only effect.
    return {key: value for key, value in stratum.items() if key not in {"model_population_id", "agent_models"}}


def _task_stratum_key(record: dict[str, Any]) -> str:
    task_stratum = _task_stratum(record)
    return "" if task_stratum is None else json.dumps(
        task_stratum, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _protocol_key(record: dict[str, Any]) -> tuple[str, str, str]:
    protocol = record["protocol"]
    return tuple(protocol[field] for field in PROTOCOL_FIELDS)  # type: ignore[return-value]


def _episode_metric(record: dict[str, Any], name: str) -> float | None:
    if name == "joint_success":
        return float(record["outcome"]["joint_success"])
    if name == "wire_bytes":
        total = 0
        for transmission in record["transmissions"]:
            framing = transmission.get("framing_utf8_bytes")
            if framing is None:
                return None
            total += transmission["payload_utf8_bytes"] + framing
        return float(total)
    if name == "transmissions":
        return float(len(record["transmissions"]))
    if name in {"input_tokens", "output_tokens"}:
        field = name
        values = [call[field] for call in record["model_calls"]]
        if any(value is None for value in values):
            return None
        return float(sum(values))
    if name == "model_calls":
        return float(len(record["model_calls"]))
    if name == "service_seconds":
        values = [call["service_seconds"] for call in record["model_calls"]]
        if any(value is None for value in values):
            return None
        return float(sum(values))
    if name in {"wall_seconds", "critical_path_seconds"}:
        value = record["runtime"].get(name)
        return None if value is None else float(value)
    if name == "amortized_setup_bytes":
        unique: dict[str, dict[str, Any]] = {}
        for artifact in record["setup"]:
            artifact_id = artifact["artifact_id"]
            if artifact_id in unique and unique[artifact_id] != artifact:
                raise RecordError(f"setup artifact {artifact_id!r} has inconsistent metadata")
            unique[artifact_id] = artifact
        total = 0.0
        for artifact in unique.values():
            one_time_bytes = artifact.get("one_time_bytes")
            if one_time_bytes is None:
                return None
            total += one_time_bytes / artifact["reuse_horizon"]
        return total
    raise RecordError(f"unsupported paired metric {name!r}")


def _quantile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _paired_bootstrap(
    differences: list[float], *, replicates: int, rng: random.Random
) -> dict[str, float | int | None]:
    if not differences:
        return {"paired_episodes": 0, "mean_left_minus_right": None, "ci95_low": None, "ci95_high": None}
    count = len(differences)
    mean = sum(differences) / count
    if count == 1:
        low = high = differences[0]
    else:
        samples = []
        for _ in range(replicates):
            samples.append(sum(differences[rng.randrange(count)] for _ in range(count)) / count)
        samples.sort()
        low, high = _quantile(samples, 0.025), _quantile(samples, 0.975)
    return {
        "paired_episodes": count,
        "mean_left_minus_right": mean,
        "ci95_low": low,
        "ci95_high": high,
    }


def paired_report(
    records: list[dict[str, Any]], *, replicates: int = 5000, seed: int = 1729
) -> dict[str, Any]:
    if replicates < 100:
        raise RecordError("bootstrap replicates must be at least 100")
    # Reuse the canonical validator and duplicate-ID checks before pairing.
    aggregate(records)
    versions = {record["schema_version"] for record in records}
    groups: dict[tuple[str, tuple[str, str, str], str], dict[str, dict[str, Any]]] = defaultdict(dict)
    full_strata: dict[str, dict[str, Any] | None] = {}
    task_strata: dict[str, dict[str, Any] | None] = {}
    for record in records:
        task_key = _task_stratum_key(record)
        full_stratum_key = _stratum_key(record)
        protocol_key = _protocol_key(record)
        full_strata[full_stratum_key] = record.get("_normalized_stratum")
        task_strata[task_key] = _task_stratum(record)
        episode_id = record["episode_id"]
        groups[(task_key, protocol_key, full_stratum_key)][episode_id] = record

    task_groups: dict[str, list[tuple[str, tuple[str, str, str]]]] = defaultdict(list)
    for task_key, protocol_key, full_stratum_key in groups:
        task_groups[task_key].append((full_stratum_key, protocol_key))

    comparisons: list[dict[str, Any]] = []
    for task_key in sorted(task_groups):
        conditions = sorted(set(task_groups[task_key]))
        for left_condition, right_condition in itertools.combinations(conditions, 2):
            left_stratum_key, left_protocol_key = left_condition
            right_stratum_key, right_protocol_key = right_condition
            left = groups[(task_key, left_protocol_key, left_stratum_key)]
            right = groups[(task_key, right_protocol_key, right_stratum_key)]
            common_ids = sorted(set(left) & set(right))
            metrics: dict[str, Any] = {}
            for metric_index, metric_name in enumerate(BOOTSTRAP_METRICS):
                deltas = []
                missing_pairs = 0
                for episode_id in common_ids:
                    left_value = _episode_metric(left[episode_id], metric_name)
                    right_value = _episode_metric(right[episode_id], metric_name)
                    if left_value is None or right_value is None:
                        missing_pairs += 1
                    else:
                        deltas.append(left_value - right_value)
                metrics[metric_name] = {
                    **_paired_bootstrap(
                        deltas,
                        replicates=replicates,
                        rng=random.Random(seed + len(comparisons) * len(BOOTSTRAP_METRICS) + metric_index),
                    ),
                    "missing_pairs": missing_pairs,
                }
            comparisons.append({
                "task_stratum": task_strata[task_key],
                "left_stratum": full_strata[left_stratum_key],
                "right_stratum": full_strata[right_stratum_key],
                "control_alignment": {
                    "model_strata_matched": (
                    None if full_strata[left_stratum_key] is None or full_strata[right_stratum_key] is None
                    else full_strata[left_stratum_key] == full_strata[right_stratum_key]
                    ),
                    "policy_matched": left_protocol_key[0] == right_protocol_key[0],
                    "decoder_matched": left_protocol_key[2] == right_protocol_key[2],
                    "code_differs": left_protocol_key[1] != right_protocol_key[1],
                },
                "pairing_basis": (
                    "episode_id_only_task_identity_unverified"
                    if next(iter(versions)) == "tlu.costs.v1"
                    else "shared_task_stratum_and_episode_id"
                ),
                "left_protocol": dict(zip(PROTOCOL_FIELDS, left_protocol_key)),
                "right_protocol": dict(zip(PROTOCOL_FIELDS, right_protocol_key)),
                "left_only_episodes": len(set(left) - set(right)),
                "right_only_episodes": len(set(right) - set(left)),
                "paired_episode_count": len(common_ids),
                "metrics": metrics,
            })

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_schema_version": next(iter(versions)),
        "method": "paired episode bootstrap; percentile 95% interval; left minus right",
        "bootstrap_replicates": replicates,
        "seed": seed,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="per-episode tlu.costs.v1 or tlu.costs.v2 JSONL")
    parser.add_argument("-o", "--output", type=Path, help="write report JSON to this path (default: stdout)")
    parser.add_argument("--replicates", type=int, default=5000, help="paired bootstrap resamples (minimum 100)")
    parser.add_argument("--seed", type=int, default=1729, help="deterministic bootstrap random seed")
    args = parser.parse_args()
    try:
        report = paired_report(read_jsonl(args.input), replicates=args.replicates, seed=args.seed)
    except (OSError, RecordError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
