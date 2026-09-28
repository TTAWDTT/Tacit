"""Summarize task-grounded atomic-claim labels without judging message text."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


INPUT_SCHEMA_VERSION = "tlu.claim-audit.v1"
REPORT_SCHEMA_VERSION = "tlu.claim-audit-report.v1"
STRATUM_FIELDS = (
    "experiment_id", "task_id", "split", "task_parameters",
    "model_population_id", "sender_model", "receiver_model", "scorer_id",
)
CLAIM_BOOLEAN_FIELDS = (
    "sender_observed", "fact_true", "receiver_knows_before", "task_relevant",
)


class ClaimAuditError(ValueError):
    """Raised when a claim-audit record is incomplete or inconsistent."""


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimAuditError(f"{field}: expected non-empty string")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ClaimAuditError(f"{field}: expected boolean")
    return value


def validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ClaimAuditError("record: expected object")
    if record.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ClaimAuditError(f"schema_version: expected {INPUT_SCHEMA_VERSION!r}")

    _nonempty_string(record.get("episode_id"), "episode_id")
    _nonempty_string(record.get("protocol_id"), "protocol_id")
    _nonempty_string(record.get("claim_extractor_id"), "claim_extractor_id")
    stratum = record.get("stratum")
    if not isinstance(stratum, dict):
        raise ClaimAuditError("stratum: expected object")
    for field in STRATUM_FIELDS:
        if field == "task_parameters":
            if not isinstance(stratum.get(field), dict):
                raise ClaimAuditError("stratum.task_parameters: expected object")
            try:
                json.dumps(stratum[field], sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ClaimAuditError(
                    "stratum.task_parameters: expected finite JSON values"
                ) from error
        else:
            _nonempty_string(stratum.get(field), f"stratum.{field}")

    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ClaimAuditError("messages: expected array")
    seen_messages: set[str] = set()
    seen_claim_instances: set[str] = set()
    for message_index, message in enumerate(messages):
        prefix = f"messages[{message_index}]"
        if not isinstance(message, dict):
            raise ClaimAuditError(f"{prefix}: expected object")
        message_id = _nonempty_string(message.get("message_id"), f"{prefix}.message_id")
        if message_id in seen_messages:
            raise ClaimAuditError(f"{prefix}.message_id: duplicate {message_id!r}")
        seen_messages.add(message_id)
        _nonempty_string(message.get("sender_id"), f"{prefix}.sender_id")
        _nonempty_string(message.get("recipient_id"), f"{prefix}.recipient_id")
        claims = message.get("claims")
        if not isinstance(claims, list):
            raise ClaimAuditError(f"{prefix}.claims: expected array")
        for claim_index, claim in enumerate(claims):
            claim_prefix = f"{prefix}.claims[{claim_index}]"
            if not isinstance(claim, dict):
                raise ClaimAuditError(f"{claim_prefix}: expected object")
            instance_id = _nonempty_string(
                claim.get("claim_instance_id"), f"{claim_prefix}.claim_instance_id"
            )
            _nonempty_string(claim.get("fact_id"), f"{claim_prefix}.fact_id")
            if instance_id in seen_claim_instances:
                raise ClaimAuditError(
                    f"{claim_prefix}.claim_instance_id: duplicate {instance_id!r}"
                )
            seen_claim_instances.add(instance_id)
            for field in CLAIM_BOOLEAN_FIELDS:
                _boolean(claim.get(field), f"{claim_prefix}.{field}")

    return record


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "rate": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _stratum_key(record: dict[str, Any]) -> str:
    try:
        return json.dumps(
            {field: record["stratum"][field] for field in STRATUM_FIELDS},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ClaimAuditError("stratum: values must be finite JSON data") from error


def _protocol_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _stratum_key(record),
        record["protocol_id"],
        record["claim_extractor_id"],
    )


def _claim_flags(claim: dict[str, Any]) -> dict[str, bool]:
    return {
        "sender_observed": claim["sender_observed"],
        "fact_true": claim["fact_true"],
        "receiver_novel": not claim["receiver_knows_before"],
        "task_relevant": claim["task_relevant"],
        "supported": claim["sender_observed"] and claim["fact_true"],
        "grounded_novel_relevant": (
            claim["sender_observed"]
            and claim["fact_true"]
            and not claim["receiver_knows_before"]
            and claim["task_relevant"]
        ),
    }


def aggregate(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_episodes: set[tuple[str, str, str, str]] = set()
    for raw in records:
        record = validate_record(raw)
        key = _protocol_key(record)
        episode_key = (key[0], key[1], key[2], record["episode_id"])
        if episode_key in seen_episodes:
            raise ClaimAuditError(
                "duplicate episode_id within a stratum/protocol/extractor: "
                f"{record['episode_id']!r}"
            )
        seen_episodes.add(episode_key)
        groups[key].append(record)

    output_groups = []
    for (stratum_json, protocol_id, extractor_id), group in sorted(groups.items()):
        totals = {
            "sender_observed": 0,
            "fact_true": 0,
            "receiver_novel": 0,
            "task_relevant": 0,
            "supported": 0,
            "grounded_novel_relevant": 0,
        }
        message_count = 0
        claim_count = 0
        no_message_episodes = 0
        no_claim_episodes = 0
        episode_q: list[float] = []
        for record in group:
            messages = record["messages"]
            message_count += len(messages)
            no_message_episodes += not messages
            claims = [claim for message in messages for claim in message["claims"]]
            claim_count += len(claims)
            no_claim_episodes += not claims
            if not claims:
                continue
            episode_novel_relevant = 0
            for claim in claims:
                flags = _claim_flags(claim)
                for name, value in flags.items():
                    totals[name] += value
                episode_novel_relevant += flags["grounded_novel_relevant"]
            episode_q.append(episode_novel_relevant / len(claims))

        rates = {name: _ratio(count, claim_count) for name, count in totals.items()}
        output_groups.append({
            "stratum": json.loads(stratum_json),
            "protocol_id": protocol_id,
            "claim_extractor_id": extractor_id,
            "episodes": len(group),
            "messages": message_count,
            "claims": claim_count,
            "episodes_without_messages": no_message_episodes,
            "episodes_without_claims": no_claim_episodes,
            "claim_rates": rates,
            "macro_episode_grounded_novel_relevant_rate": (
                sum(episode_q) / len(episode_q) if episode_q else None
            ),
            "episodes_with_claims_for_macro": len(episode_q),
        })

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "group_count": len(output_groups),
        "groups": output_groups,
        "interpretation": (
            "Claim-label rates are descriptive semantic diagnostics, not task utility, "
            "causal effects, or language-superiority evidence. Their validity depends "
            "on the declared atomic-claim extractor and task-ground-truth labels."
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ClaimAuditError(f"line {line_number}: invalid JSON: {error}") from error
                try:
                    records.append(validate_record(value))
                except ClaimAuditError as error:
                    raise ClaimAuditError(f"line {line_number}: {error}") from error
    except OSError as error:
        raise ClaimAuditError(f"cannot read {path}: {error}") from error
    return records


def write_report(report: dict[str, Any], output_path: Path | None) -> None:
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output_path is None:
        sys.stdout.write(serialized)
        return
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    except OSError as error:
        raise ClaimAuditError(f"cannot write {output_path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="input tlu.claim-audit.v1 JSONL")
    parser.add_argument("--output", type=Path, help="write report JSON here (default: stdout)")
    args = parser.parse_args(argv)
    try:
        report = aggregate(read_jsonl(args.records))
        write_report(report, args.output)
    except ClaimAuditError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
