"""Compute empirical Pareto frontiers from TLU per-episode cost records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .cost_report import CONDITION_FIELDS, RecordError, aggregate, read_jsonl
except ImportError:  # Direct ``python tools/frontier_report.py`` invocation.
    from cost_report import CONDITION_FIELDS, RecordError, aggregate, read_jsonl


REPORT_SCHEMA_VERSION = "tlu.frontier-report.v1"
FRONTIER_SCOPES = {
    "channel_bytes": ("wire_bytes",),
    "channel_and_inference_tokens": ("wire_bytes", "input_tokens", "output_tokens"),
    "operational": (
        "wire_bytes", "input_tokens", "output_tokens", "model_calls",
        "service_seconds", "wall_seconds", "amortized_setup_bytes",
    ),
    "with_critical_path": (
        "wire_bytes", "input_tokens", "output_tokens", "model_calls",
        "service_seconds", "wall_seconds", "critical_path_seconds", "amortized_setup_bytes",
    ),
    "end_to_end_operational": (
        "wire_bytes", "amortized_setup_bytes", "input_tokens", "output_tokens",
        "amortized_setup_tokens", "model_calls", "amortized_setup_model_calls",
        "service_seconds", "amortized_setup_service_seconds", "wall_seconds",
        "amortized_setup_wall_seconds",
    ),
    "end_to_end_with_critical_path": (
        "wire_bytes", "amortized_setup_bytes", "input_tokens", "output_tokens",
        "amortized_setup_tokens", "model_calls", "amortized_setup_model_calls",
        "service_seconds", "amortized_setup_service_seconds", "wall_seconds",
        "amortized_setup_wall_seconds", "critical_path_seconds",
    ),
}


def _stratum_key(record: dict[str, Any]) -> str:
    stratum = record.get("_normalized_stratum")
    return "" if stratum is None else json.dumps(
        stratum, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _protocol_key(record: dict[str, Any]) -> tuple[str, str, str]:
    protocol = record["protocol"]
    return tuple(protocol[field] for field in CONDITION_FIELDS)  # type: ignore[return-value]


def _mean(values: list[float | None]) -> dict[str, Any]:
    observed = [value for value in values if value is not None]
    return {
        "mean": sum(observed) / len(observed) if observed else None,
        "observed": len(observed),
        "missing": len(values) - len(observed),
        "complete": len(observed) == len(values),
    }


def _setup_bytes(records: list[dict[str, Any]]) -> float | None:
    return _setup_value(records, "one_time_bytes")


def _setup_value(records: list[dict[str, Any]], field: str) -> float | None:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        for artifact in record["setup"]:
            artifact_id = artifact["artifact_id"]
            if artifact_id in unique and unique[artifact_id] != artifact:
                raise RecordError(f"setup artifact {artifact_id!r} has inconsistent metadata")
            unique[artifact_id] = artifact
    total = 0.0
    for artifact in unique.values():
        size = artifact.get(field)
        if size is None:
            return None
        total += size / artifact["reuse_horizon"]
    return total


def _setup_tokens(records: list[dict[str, Any]]) -> float | None:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        for artifact in record["setup"]:
            artifact_id = artifact["artifact_id"]
            if artifact_id in unique and unique[artifact_id] != artifact:
                raise RecordError(f"setup artifact {artifact_id!r} has inconsistent metadata")
            unique[artifact_id] = artifact
    tokenizer_ids = {tokenizer for item in unique.values() for tokenizer in item["one_time_tokens"]}
    if len(tokenizer_ids) > 1:
        return None
    tokenizer = next(iter(tokenizer_ids), None)
    return sum(
        (artifact["one_time_tokens"].get(tokenizer, 0) if tokenizer else 0)
        / artifact["reuse_horizon"]
        for artifact in unique.values()
    )


def _episode_metrics(record: dict[str, Any]) -> dict[str, float | None]:
    wire_bytes: float | None = 0.0
    for transmission in record["transmissions"]:
        framing = transmission.get("_normalized_framing_bytes")
        if framing is None:
            wire_bytes = None
            break
        wire_bytes += transmission["_normalized_payload_bytes"] + framing

    def sum_calls(field: str) -> float | None:
        if len({call["tokenizer"] for call in record["model_calls"]}) > 1:
            return None
        values = [call[field] for call in record["model_calls"]]
        return None if any(value is None for value in values) else float(sum(values))

    runtime = record["runtime"]
    return {
        "wire_bytes": wire_bytes,
        "input_tokens": sum_calls("input_tokens"),
        "output_tokens": sum_calls("output_tokens"),
        "model_calls": float(len(record["model_calls"])),
        "service_seconds": sum_calls("service_seconds"),
        "wall_seconds": None if runtime.get("wall_seconds") is None else float(runtime["wall_seconds"]),
        "critical_path_seconds": (
            None if runtime.get("critical_path_seconds") is None
            else float(runtime["critical_path_seconds"])
        ),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any], dimensions: tuple[str, ...]) -> bool:
    if left["success_rate"] < right["success_rate"]:
        return False
    if any(left["costs"][name]["mean"] > right["costs"][name]["mean"] for name in dimensions):
        return False
    return left["success_rate"] > right["success_rate"] or any(
        left["costs"][name]["mean"] < right["costs"][name]["mean"] for name in dimensions
    )


def _summarize_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    episode_metrics = [_episode_metrics(record) for record in records]
    successes = [record["outcome"]["joint_success"] for record in records]
    tokenizer_units = sorted({
        tokenizer
        for record in records
        for tokenizer in (
            [call["tokenizer"] for call in record["model_calls"]]
            + [tokenizer for artifact in record["setup"]
               for tokenizer in artifact["one_time_tokens"]]
        )
    })
    costs = {
        name: _mean([metrics[name] for metrics in episode_metrics])
        for name in next(iter(episode_metrics)).keys()
    }
    setup_bytes = _setup_bytes(records)
    costs["amortized_setup_bytes"] = {
        "mean": setup_bytes,
        "observed": len(records) if setup_bytes is not None else 0,
        "missing": 0 if setup_bytes is not None else len(records),
        "complete": setup_bytes is not None,
    }
    for field, name in (
        ("one_time_model_calls", "amortized_setup_model_calls"),
        ("one_time_service_seconds", "amortized_setup_service_seconds"),
        ("one_time_wall_seconds", "amortized_setup_wall_seconds"),
    ):
        value = _setup_value(records, field)
        costs[name] = {
            "mean": value,
            "observed": len(records) if value is not None else 0,
            "missing": 0 if value is not None else len(records),
            "complete": value is not None,
        }
    setup_tokens = _setup_tokens(records)
    costs["amortized_setup_tokens"] = {
        "mean": setup_tokens,
        "observed": len(records) if setup_tokens is not None else 0,
        "missing": 0 if setup_tokens is not None else len(records),
        "complete": setup_tokens is not None,
    }
    if len(tokenizer_units) > 1:
        for name in ("input_tokens", "output_tokens"):
            costs[name] = {"mean": None, "observed": 0, "missing": len(records), "complete": False}
    return {
        "episodes": len(records),
        "successes": sum(successes),
        "success_rate": sum(successes) / len(successes),
        "tokenizer_units": tokenizer_units,
        "costs": costs,
    }


def frontier_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate(records)
    versions = {record["schema_version"] for record in records}
    if versions == {"tlu.costs.v1"}:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "input_schema_version": "tlu.costs.v1",
            "method": "empirical Pareto dominance over per-episode means; success maximized and declared cost dimensions minimized; no scalarized cost",
            "uncertainty_note": "Frontier membership is descriptive from observed means. Use paired bootstrap comparisons for uncertainty; this report does not infer population dominance.",
            "groups": [],
            "excluded_reason": "tlu.costs.v1 lacks task and model strata; a stratified frontier cannot be established.",
        }
    groups: dict[tuple[str, tuple[str, str, str]], list[dict[str, Any]]] = defaultdict(list)
    strata: dict[str, dict[str, Any] | None] = {}
    for record in records:
        stratum_key = _stratum_key(record)
        groups[(stratum_key, _protocol_key(record))].append(record)
        strata[stratum_key] = record.get("_normalized_stratum")

    by_stratum: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for stratum_key, protocol_key in groups:
        by_stratum[stratum_key].append(protocol_key)

    reports = []
    for stratum_key in sorted(by_stratum):
        condition_points = []
        for protocol_key in sorted(set(by_stratum[stratum_key])):
            point = {
                "protocol": dict(zip(CONDITION_FIELDS, protocol_key)),
                **_summarize_condition(groups[(stratum_key, protocol_key)]),
            }
            condition_points.append(point)

        episode_sets = [
            {record["episode_id"] for record in groups[(stratum_key, protocol_key)]}
            for protocol_key in sorted(set(by_stratum[stratum_key]))
        ]
        episode_coverage_matched = not episode_sets or all(episode_set == episode_sets[0] for episode_set in episode_sets[1:])

        tokenizer_units = sorted({
            tokenizer
            for point in condition_points
            for tokenizer in point["tokenizer_units"]
        })
        if len(tokenizer_units) > 1:
            for point in condition_points:
                for name in ("input_tokens", "output_tokens"):
                    point["costs"][name] = {
                        "mean": None,
                        "observed": 0,
                        "missing": point["episodes"],
                        "complete": False,
                    }
                point["costs"]["amortized_setup_tokens"] = {
                    "mean": None,
                    "observed": 0,
                    "missing": point["episodes"],
                    "complete": False,
                }

        frontiers = []
        for scope_name, dimensions in FRONTIER_SCOPES.items():
            eligible = [
                point for point in condition_points
                if episode_coverage_matched and all(point["costs"][name]["complete"] for name in dimensions)
            ]
            excluded = [
                {
                    "protocol": point["protocol"],
                    "missing_dimensions": [
                        name for name in dimensions if not point["costs"][name]["complete"]
                    ],
                    "episode_set_mismatch": not episode_coverage_matched,
                }
                for point in condition_points if point not in eligible
            ]
            frontier = [
                point for point in eligible
                if not any(other is not point and _dominates(other, point, dimensions) for other in eligible)
            ]
            frontiers.append({
                "scope": scope_name,
                "maximize": ["success_rate"],
                "minimize": list(dimensions),
                "eligible_conditions": len(eligible),
                "excluded_conditions": excluded,
                "nondominated_protocols": [point["protocol"] for point in frontier],
            })

        reports.append({
            "stratum": strata[stratum_key],
            "aggregation_scope": "protocol_only_legacy_v1" if next(iter(versions)) == "tlu.costs.v1" else "full_stratum_fixed_model_population",
            "tokenizer_units": tokenizer_units,
            "episode_coverage_matched_across_conditions": episode_coverage_matched,
            "conditions": condition_points,
            "frontiers": frontiers,
        })

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_schema_version": next(iter(versions)),
        "method": "empirical Pareto dominance over per-episode means; success maximized and declared cost dimensions minimized; no scalarized cost",
        "uncertainty_note": "Frontier membership is descriptive from observed means. Use paired bootstrap comparisons for uncertainty; this report does not infer population dominance.",
        "groups": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="per-episode tlu.costs.v1, v2, or v3 JSONL")
    parser.add_argument("-o", "--output", type=Path, help="write report JSON to this path (default: stdout)")
    args = parser.parse_args()
    try:
        report = frontier_report(read_jsonl(args.input))
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
