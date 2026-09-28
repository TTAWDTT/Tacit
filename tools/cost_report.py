"""Validate and aggregate TLU per-episode communication-cost JSONL records.

This tool intentionally keeps channel, inference, runtime, and setup costs in
separate dimensions. It uses only the Python standard library and never reads
model prompts or calls an inference endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


LEGACY_SCHEMA_VERSION = "tlu.costs.v1"
SCHEMA_VERSION = "tlu.costs.v2"
REPORT_SCHEMA_VERSION = "tlu.cost-report.v1"
CONDITION_FIELDS = ("policy_id", "code_id", "decoder_id")
STRATUM_FIELDS = ("experiment_id", "task_id", "split", "scorer_id", "model_population_id")
RUNTIME_METRICS = (
    "wall_seconds",
    "critical_path_seconds",
    "tool_seconds",
    "process_cpu_seconds",
    "process_gpu_seconds",
    "peak_rss_bytes",
    "peak_vram_bytes",
)


class RecordError(ValueError):
    """Raised when a record violates the versioned input contract."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecordError(f"{where}: expected object")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{where}: expected non-empty string")
    return value


def _number(value: Any, where: str, *, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"{where}: expected non-negative number or null")
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0 or (integer and not isinstance(value, int)):
        raise RecordError(f"{where}: expected non-negative {'integer' if integer else 'number'} or null")
    return value


def _required_integer(value: Any, where: str) -> int:
    number = _number(value, where, integer=True)
    if number is None:
        raise RecordError(f"{where}: value is required")
    return number


def _metric(values: list[int | float | None]) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    return {
        "observed_sum": sum(present),
        "observed_mean": sum(present) / len(present) if present else None,
        "observed": len(present),
        "missing": len(values) - len(present),
        "complete": len(present) == len(values),
    }


def _peak_metric(values: list[int | float | None]) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    return {
        "observed_max": max(present) if present else None,
        "observed": len(present),
        "missing": len(values) - len(present),
        "complete": len(present) == len(values),
    }


def _validate_record(record: Any, line_number: int) -> dict[str, Any]:
    prefix = f"line {line_number}"
    record = _object(record, prefix)
    input_version = record.get("schema_version")
    if input_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise RecordError(f"{prefix}: schema_version must be {SCHEMA_VERSION!r} or legacy {LEGACY_SCHEMA_VERSION!r}")
    _string(record.get("episode_id"), f"{prefix}.episode_id")

    if input_version == SCHEMA_VERSION:
        stratum = _object(record.get("stratum"), f"{prefix}.stratum")
        for name in STRATUM_FIELDS:
            _string(stratum.get(name), f"{prefix}.stratum.{name}")
        task_parameters = stratum.get("task_parameters")
        if not isinstance(task_parameters, dict):
            raise RecordError(f"{prefix}.stratum.task_parameters: expected object")
        try:
            json.dumps(task_parameters, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RecordError(f"{prefix}.stratum.task_parameters: must contain finite JSON values") from exc
        agent_models = stratum.get("agent_models")
        if not isinstance(agent_models, dict) or not agent_models:
            raise RecordError(f"{prefix}.stratum.agent_models: expected non-empty agent-to-model object")
        for agent, model in agent_models.items():
            _string(agent, f"{prefix}.stratum.agent_models key")
            _string(model, f"{prefix}.stratum.agent_models.{agent}")
    else:
        if "stratum" in record:
            raise RecordError(f"{prefix}: legacy v1 records must not contain v2 stratum metadata")
        stratum = None
    record["_normalized_stratum"] = stratum

    condition = _object(record.get("protocol"), f"{prefix}.protocol")
    for name in CONDITION_FIELDS:
        _string(condition.get(name), f"{prefix}.protocol.{name}")

    outcome = _object(record.get("outcome"), f"{prefix}.outcome")
    if not isinstance(outcome.get("joint_success"), bool):
        raise RecordError(f"{prefix}.outcome.joint_success: expected boolean")
    _number(outcome.get("answer_score"), f"{prefix}.outcome.answer_score")

    transmissions = record.get("transmissions")
    if not isinstance(transmissions, list):
        raise RecordError(f"{prefix}.transmissions: expected array")
    for index, transmission in enumerate(transmissions):
        where = f"{prefix}.transmissions[{index}]"
        transmission = _object(transmission, where)
        _required_integer(transmission.get("round"), f"{where}.round")
        _string(transmission.get("sender"), f"{where}.sender")
        recipients = transmission.get("recipients")
        if not isinstance(recipients, list) or not recipients:
            raise RecordError(f"{where}.recipients: expected non-empty array")
        for recipient in recipients:
            _string(recipient, f"{where}.recipients[]")
        if len(set(recipients)) != len(recipients):
            raise RecordError(f"{where}.recipients: duplicates are not allowed")
        _required_integer(transmission.get("payload_utf8_bytes"), f"{where}.payload_utf8_bytes")
        if input_version == SCHEMA_VERSION:
            _required_integer(transmission.get("framing_utf8_bytes"), f"{where}.framing_utf8_bytes")
        elif "framing_utf8_bytes" in transmission:
            _number(transmission["framing_utf8_bytes"], f"{where}.framing_utf8_bytes", integer=True)

        recipient_tokens = transmission.get("recipient_tokens")
        if not isinstance(recipient_tokens, dict) or set(recipient_tokens) != set(recipients):
            raise RecordError(f"{where}.recipient_tokens: provide one entry per recipient")
        for agent, token_record in recipient_tokens.items():
            token_record = _object(token_record, f"{where}.recipient_tokens.{agent}")
            _string(token_record.get("tokenizer"), f"{where}.recipient_tokens.{agent}.tokenizer")
            _number(token_record.get("tokens"), f"{where}.recipient_tokens.{agent}.tokens", integer=True)

    calls = record.get("model_calls")
    if not isinstance(calls, list):
        raise RecordError(f"{prefix}.model_calls: expected array")
    for index, call in enumerate(calls):
        where = f"{prefix}.model_calls[{index}]"
        call = _object(call, where)
        for name in ("agent", "stage", "model", "tokenizer"):
            _string(call.get(name), f"{where}.{name}")
        for name in ("input_tokens", "output_tokens"):
            _number(call.get(name), f"{where}.{name}", integer=True)
        _number(call.get("service_seconds"), f"{where}.service_seconds")
        for name in ("retry", "truncated"):
            if not isinstance(call.get(name), bool):
                raise RecordError(f"{where}.{name}: expected boolean")
        billing = call.get("billing")
        if billing is not None:
            billing = _object(billing, f"{where}.billing")
            for name in ("input_units", "output_units", "input_cost", "output_cost"):
                _number(billing.get(name), f"{where}.billing.{name}")
            for name in ("unit_label", "currency"):
                value = billing.get(name)
                if value is not None:
                    _string(value, f"{where}.billing.{name}")
            if any(billing.get(name) is not None for name in ("input_units", "output_units")) and billing.get("unit_label") is None:
                raise RecordError(f"{where}.billing.unit_label: required when billed units are present")
            if any(billing.get(name) is not None for name in ("input_cost", "output_cost")) and billing.get("currency") is None:
                raise RecordError(f"{where}.billing.currency: required when billed costs are present")

    runtime = _object(record.get("runtime"), f"{prefix}.runtime")
    for name in RUNTIME_METRICS:
        if name in runtime:
            _number(runtime[name], f"{prefix}.runtime.{name}", integer=name.endswith("_bytes"))

    setup = record.get("setup")
    if not isinstance(setup, list):
        raise RecordError(f"{prefix}.setup: expected array")
    for index, artifact in enumerate(setup):
        where = f"{prefix}.setup[{index}]"
        artifact = _object(artifact, where)
        _string(artifact.get("artifact_id"), f"{where}.artifact_id")
        _number(artifact.get("one_time_bytes"), f"{where}.one_time_bytes", integer=True)
        horizon = _number(artifact.get("reuse_horizon"), f"{where}.reuse_horizon", integer=True)
        if horizon is None or horizon < 1:
            raise RecordError(f"{where}.reuse_horizon: expected positive integer")
        setup_tokens = artifact.get("one_time_tokens")
        if not isinstance(setup_tokens, dict):
            raise RecordError(f"{where}.one_time_tokens: expected tokenizer-to-count object")
        for tokenizer, count in setup_tokens.items():
            _string(tokenizer, f"{where}.one_time_tokens key")
            _number(count, f"{where}.one_time_tokens.{tokenizer}", integer=True)

    return record


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    records = [_validate_record(record, index) for index, record in enumerate(records, start=1)]
    if not records:
        raise RecordError("aggregate requires at least one episode record")
    versions = {record.get("schema_version") for record in records}
    if len(versions) != 1 or not versions <= {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise RecordError("aggregate one input schema version at a time")
    input_version = next(iter(versions))
    groups: dict[tuple[str, tuple[str, str, str]], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        protocol = record["protocol"]
        protocol_key = tuple(protocol[field] for field in CONDITION_FIELDS)
        stratum = record.get("_normalized_stratum")
        stratum_key = "" if stratum is None else json.dumps(stratum, sort_keys=True, separators=(",", ":"), allow_nan=False)
        groups[(stratum_key, protocol_key)].append(record)

    reports: list[dict[str, Any]] = []
    for (stratum_key, protocol_key), episodes in sorted(groups.items()):
        episode_ids = [episode["episode_id"] for episode in episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise RecordError(f"duplicate episode_id within stratum/protocol condition {(stratum_key, protocol_key)!r}")
        transmissions = [item for episode in episodes for item in episode["transmissions"]]
        calls = [item for episode in episodes for item in episode["model_calls"]]
        successes = sum(episode["outcome"]["joint_success"] for episode in episodes)

        recipient_token_values: dict[str, list[int | None]] = defaultdict(list)
        for transmission in transmissions:
            for token_record in transmission["recipient_tokens"].values():
                recipient_token_values[token_record["tokenizer"]].append(token_record["tokens"])

        inference: dict[str, Any] = {"model_calls": len(calls), "retries": sum(call["retry"] for call in calls),
                                     "truncated_calls": sum(call["truncated"] for call in calls)}
        calls_by_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for call in calls:
            calls_by_model[(call["model"], call["tokenizer"])].append(call)
        inference["by_model_and_tokenizer"] = []
        for (model, tokenizer), model_calls in sorted(calls_by_model.items()):
            inference["by_model_and_tokenizer"].append({
                "model": model,
                "tokenizer": tokenizer,
                "model_calls": len(model_calls),
                "input_tokens": _metric([call["input_tokens"] for call in model_calls]),
                "output_tokens": _metric([call["output_tokens"] for call in model_calls]),
                "service_seconds": _metric([call["service_seconds"] for call in model_calls]),
            })
        calls_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for call in calls:
            calls_by_stage[call["stage"]].append(call)
        inference["by_stage"] = [
            {
                "stage": stage,
                "model_calls": len(stage_calls),
                "input_tokens": _metric([call["input_tokens"] for call in stage_calls]),
                "output_tokens": _metric([call["output_tokens"] for call in stage_calls]),
                "service_seconds": _metric([call["service_seconds"] for call in stage_calls]),
            }
            for stage, stage_calls in sorted(calls_by_stage.items())
        ]
        billing_groups: dict[tuple[str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
        unreported_billing_calls = 0
        for call in calls:
            billing = call.get("billing")
            if billing is None:
                unreported_billing_calls += 1
                continue
            if all(billing.get(name) is None for name in ("input_units", "output_units", "input_cost", "output_cost")):
                unreported_billing_calls += 1
            billing_groups[(billing.get("currency"), billing.get("unit_label"))].append(billing)
        billing_report = {
            "calls": len(calls),
            "unreported_calls": unreported_billing_calls,
            "by_currency_and_unit": [
                {
                    "currency": currency,
                    "unit_label": unit_label,
                    "calls": len(billing_records),
                    "input_units": _metric([item.get("input_units") for item in billing_records]),
                    "output_units": _metric([item.get("output_units") for item in billing_records]),
                    "input_cost": _metric([item.get("input_cost") for item in billing_records]),
                    "output_cost": _metric([item.get("output_cost") for item in billing_records]),
                }
                for (currency, unit_label), billing_records in sorted(
                    billing_groups.items(), key=lambda item: (item[0][0] or "", item[0][1] or "")
                )
            ],
        }

        runtime: dict[str, Any] = {}
        for name in RUNTIME_METRICS:
            if any(name in episode["runtime"] for episode in episodes):
                values = [episode["runtime"].get(name) for episode in episodes]
                runtime[name] = _peak_metric(values) if name.startswith("peak_") else _metric(values)

        setup_artifacts: dict[str, dict[str, Any]] = {}
        for episode in episodes:
            for artifact in episode["setup"]:
                artifact_id = artifact["artifact_id"]
                if artifact_id in setup_artifacts and setup_artifacts[artifact_id] != artifact:
                    raise RecordError(f"setup artifact {artifact_id!r} has inconsistent metadata")
                setup_artifacts[artifact_id] = artifact
        setup_bytes = _metric([item["one_time_bytes"] for item in setup_artifacts.values()])
        amortized_setup_bytes = _metric([
            None if item["one_time_bytes"] is None
            else item["one_time_bytes"] / item["reuse_horizon"]
            for item in setup_artifacts.values()
        ])
        setup_tokenizers = sorted({tokenizer for item in setup_artifacts.values()
                                   for tokenizer in item["one_time_tokens"]})
        setup_tokens = {
            tokenizer: _metric([item["one_time_tokens"].get(tokenizer)
                                for item in setup_artifacts.values()])
            for tokenizer in setup_tokenizers
        }
        amortized_setup_tokens = {
            tokenizer: _metric([
                None if item["one_time_tokens"].get(tokenizer) is None
                else item["one_time_tokens"][tokenizer] / item["reuse_horizon"]
                for item in setup_artifacts.values()
            ])
            for tokenizer in setup_tokenizers
        }

        reports.append({
            "stratum": json.loads(stratum_key) if stratum_key else None,
            "aggregation_scope": "protocol_only_legacy_v1" if input_version == LEGACY_SCHEMA_VERSION else "full_stratum_and_protocol",
            "protocol": dict(zip(CONDITION_FIELDS, protocol_key)),
            "episodes": len(episodes),
            "joint_successes": successes,
            "joint_success_rate": successes / len(episodes),
            "answer_score": _metric([episode["outcome"].get("answer_score") for episode in episodes]),
            "channel": {
                "transmissions": len(transmissions),
                "recipient_deliveries": sum(len(item["recipients"]) for item in transmissions),
                "payload_bytes": _metric([item["payload_utf8_bytes"] for item in transmissions]),
                "wire_bytes": _metric([
                    None if item.get("framing_utf8_bytes") is None
                    else item["payload_utf8_bytes"] + item["framing_utf8_bytes"]
                    for item in transmissions
                ]),
                "recipient_delivery_bytes": _metric([
                    None if item.get("framing_utf8_bytes") is None
                    else (item["payload_utf8_bytes"] + item["framing_utf8_bytes"]) * len(item["recipients"])
                    for item in transmissions
                ]),
                "receiver_tokenizers": {
                    tokenizer: _metric(values)
                    for tokenizer, values in sorted(recipient_token_values.items())
                },
            },
            "inference": inference,
            "billing": billing_report,
            "runtime": runtime,
            "setup": {
                "unique_artifacts": len(setup_artifacts),
                "one_time_bytes": setup_bytes,
                "amortized_bytes_per_episode_at_declared_horizons": amortized_setup_bytes,
                "one_time_tokens_by_tokenizer": setup_tokens,
                "amortized_tokens_per_episode_by_tokenizer": amortized_setup_tokens,
            },
        })
    return {"schema_version": REPORT_SCHEMA_VERSION, "input_schema_version": input_version, "groups": reports}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_unique_object_pairs,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise RecordError(f"line {line_number}: invalid JSON: {exc}") from exc
            records.append(_validate_record(value, line_number))
    if not records:
        raise RecordError("input contains no episode records")
    versions = {record["schema_version"] for record in records}
    if len(versions) != 1:
        raise RecordError("input must not mix tlu.costs.v1 and tlu.costs.v2 records")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="per-episode JSONL using one consistent tlu.costs.v1 or tlu.costs.v2 schema")
    parser.add_argument("-o", "--output", type=Path, help="write report JSON to this path (default: stdout)")
    args = parser.parse_args()
    try:
        report = aggregate(read_jsonl(args.input))
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
