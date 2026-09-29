"""Validate and analyze Private Match v0.3 tlu.costs.v3 ledgers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.private_match_v0_3.bit_frontier import frontier as bit_frontier
from experiments.private_match_v0_3.generate_tasks import (
    bayes_accuracy_references, generate_episode, score_answer,
)
from experiments.private_match_v0_3.protocols import PROTOCOL_IDS, PROMPT_REVISION, parse_coordinate_message
from tools.cost_report import RecordError, aggregate, read_jsonl
from tools.frontier_report import frontier_report
from tools.paired_report import paired_report


EXPERIMENT_ID = "private-match-v0.3"
TASK_ID = "triadic-complementary-coordinate-match-v1"
EXPECTED_SENDERS = {
    "full_information": (),
    "no_message": (),
    "sender_x_only": ("sender_x",),
    "sender_y_only": ("sender_y",),
    "both_sources": ("sender_x", "sender_y"),
}
POLICY_IDS = {
    "full_information": "full-information-control-v1",
    "no_message": "no-message-v1",
    "sender_x_only": "sender-x-only-fixed-v1",
    "sender_y_only": "sender-y-only-fixed-v1",
    "both_sources": "fixed-x-then-y-unicast-v1",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordError(message)


def _rate(values: list[bool | None]) -> dict[str, Any]:
    observed = [value for value in values if value is not None]
    return {
        "positive": sum(observed),
        "observed": len(observed),
        "missing": len(values) - len(observed),
        "rate": sum(observed) / len(observed) if observed else None,
    }


def validate_private_match_records(records: list[dict[str, Any]]) -> None:
    """Reconstruct tasks and fail closed on wrong outcomes or cost attribution."""
    _require(bool(records), "report requires at least one record")
    for index, row in enumerate(records, start=1):
        prefix = f"record {index}"
        _require(row.get("schema_version") == "tlu.costs.v3", f"{prefix}: expected tlu.costs.v3")
        stratum = row.get("_normalized_stratum") or row.get("stratum")
        _require(isinstance(stratum, dict) and stratum.get("experiment_id") == EXPERIMENT_ID,
                 f"{prefix}: wrong experiment_id")
        _require(stratum.get("task_id") == TASK_ID, f"{prefix}: wrong task_id")
        params = stratum.get("task_parameters")
        _require(isinstance(params, dict), f"{prefix}: missing task parameters")
        q = params.get("q")
        _require(isinstance(q, int) and not isinstance(q, bool) and q >= 2 and q & (q - 1) == 0,
                 f"{prefix}: q must be a power of two >= 2")
        _require(params == {"q": q, "candidate_count": q * q, "agent_count": 3},
                 f"{prefix}: malformed task parameters")
        diag = row.get("diagnostics")
        _require(isinstance(diag, dict), f"{prefix}: diagnostics must be an object")
        condition = diag.get("condition")
        _require(condition in EXPECTED_SENDERS, f"{prefix}: unknown condition")
        seed = diag.get("generation_seed")
        _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
                 f"{prefix}: invalid generation_seed")
        episode_id = f"pmt3-{seed:012d}"
        _require(row.get("episode_id") == episode_id, f"{prefix}: episode ID differs from seed")
        split = "calibration" if condition == "full_information" else "evaluation"
        _require(stratum.get("split") == split, f"{prefix}: split does not match condition")

        episode = generate_episode(episode_id=episode_id, seed=seed, q=q)
        sender_x, sender_y, receiver, gold = episode
        expected_ids = [item["candidate_id"] for item in receiver["candidates"]]
        _require(diag.get("candidate_ids_in_receiver_order") == expected_ids,
                 f"{prefix}: receiver candidate table differs from generated task")
        answer_raw = diag.get("answer_text_verbatim")
        _require(isinstance(answer_raw, str), f"{prefix}: missing verbatim answer")
        answer = answer_raw.strip()
        exact_format = answer_raw == answer and answer in set(expected_ids)
        _require(diag.get("answer_format_valid") is exact_format,
                 f"{prefix}: answer format diagnostic is inconsistent")
        success = score_answer(receiver, gold, answer)
        _require(row.get("outcome", {}).get("joint_success") is success,
                 f"{prefix}: recorded success differs from exact scorer")

        protocol = row.get("protocol", {})
        representation = protocol.get("representation_id")
        if condition in {"full_information", "no_message"}:
            _require(representation is None and protocol.get("code_id") == "none",
                     f"{prefix}: control condition must not claim a representation")
        else:
            _require(representation in PROTOCOL_IDS, f"{prefix}: unknown representation ID")
            _require(protocol.get("prompt_revision") == PROMPT_REVISION,
                     f"{prefix}: prompt revision is not registered")
        _require(protocol.get("policy_id") == POLICY_IDS[condition],
                 f"{prefix}: policy ID does not match the frozen schedule")

        messages = diag.get("message_diagnostics")
        transmissions = row.get("transmissions")
        expected_senders = EXPECTED_SENDERS[condition]
        _require(isinstance(messages, list) and len(messages) == len(expected_senders),
                 f"{prefix}: wrong number of message diagnostics")
        _require(isinstance(transmissions, list) and len(transmissions) == len(expected_senders),
                 f"{prefix}: wrong number of transmissions")
        expected_sources = {"sender_x": sender_x, "sender_y": sender_y}
        for round_index, (sender, message_row, transmission) in enumerate(
                zip(expected_senders, messages, transmissions), start=1):
            _require(isinstance(message_row, dict), f"{prefix}: malformed message diagnostic")
            text = message_row.get("message")
            _require(isinstance(text, str) and message_row.get("sender") == sender,
                     f"{prefix}: message sender/order mismatch")
            _require(transmission.get("round") == round_index
                     and transmission.get("sender") == sender
                     and transmission.get("recipients") == ["receiver"],
                     f"{prefix}: transmission route/order mismatch")
            metadata = transmission.get("payload_metadata", {})
            _require(metadata.get("logical_text_utf8_bytes") == len(text.encode("utf-8")),
                     f"{prefix}: logical text byte count differs from exact UTF-8 message")
            parsed = parse_coordinate_message(
                representation, text, q=q, sender=sender
            )
            expected_value = expected_sources[sender]["private_value"]
            expected_fidelity = None if parsed[2] is None else parsed[2] == expected_value
            _require(message_row.get("format_valid") is parsed[0],
                     f"{prefix}: message format diagnostic is inconsistent")
            _require(message_row.get("semantic_fidelity") is expected_fidelity,
                     f"{prefix}: message semantic-fidelity diagnostic is inconsistent")

        calls = row.get("model_calls")
        expected_call_agents = [*expected_senders, "receiver"]
        _require(isinstance(calls, list) and len(calls) == len(expected_call_agents),
                 f"{prefix}: wrong number of model calls")
        _require([call.get("agent") for call in calls] == expected_call_agents,
                 f"{prefix}: model call agent/stage order mismatch")
        _require(calls[-1].get("stage") == "final_answer",
                 f"{prefix}: sealed answer call is missing")


def private_match_report(records: list[dict[str, Any]], *, replicates: int = 5000,
                         seed: int = 1729) -> dict[str, Any]:
    validate_private_match_records(records)
    cost_summary = aggregate(records)
    return {
        "schema_version": "tlu.private-match-report.v2",
        "experiment_id": EXPERIMENT_ID,
        "validation": "generated task reconstruction, exact answer scoring, message/parser diagnostics, routes, and call counts passed",
        "analytic_controls": _analytic_controls(records),
        "representation_diagnostics": _representation_diagnostics(records),
        "cost_summary": cost_summary,
        "paired_comparisons": paired_report(records, replicates=replicates, seed=seed),
        "empirical_frontiers": frontier_report(records),
        "claim_limit": "A feasibility pilot report is descriptive. It does not establish population superiority; inspect policy/decoder alignment and complete cost coverage for every comparison.",
    }


def _analytic_controls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q_values = sorted({
        (row.get("_normalized_stratum") or row["stratum"])["task_parameters"]["q"]
        for row in records
    })
    controls = []
    for q in q_values:
        references = bayes_accuracy_references(q)
        width = q.bit_length() - 1
        controls.append({
            "q": q,
            "candidate_count": q * q,
            "bayes_accuracy": {
                name: f"{value.numerator}/{value.denominator}"
                for name, value in references.items()
            },
            "ideal_fixed_width_total_payload_bits": bit_frontier(
                q=q, max_total_payload_bits=2 * width
            ),
        })
    return controls


def _representation_diagnostics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None, str], list[dict[str, Any]]] = {}
    for row in records:
        diag = row["diagnostics"]
        stratum = row.get("_normalized_stratum") or row["stratum"]
        task_key = json.dumps(stratum, sort_keys=True, separators=(",", ":"))
        key = diag["condition"], row["protocol"].get("representation_id"), task_key
        groups.setdefault(key, []).append(row)
    summaries = []
    for (condition, representation, task_key), rows in sorted(groups.items(), key=lambda x: str(x[0])):
        stratum = rows[0].get("_normalized_stratum") or rows[0]["stratum"]
        q = stratum["task_parameters"]["q"]
        message_diagnostics = [
            message for row in rows for message in row["diagnostics"]["message_diagnostics"]
        ]
        summaries.append({
            "condition": condition,
            "representation_id": representation,
            "stratum": stratum,
            "q": q,
            "episodes": len(rows),
            "joint_success": _rate([row["outcome"]["joint_success"] for row in rows]),
            "answer_format_valid": _rate([
                row["diagnostics"].get("answer_format_valid") for row in rows
            ]),
            "message_format_valid": _rate([
                item.get("format_valid") for item in message_diagnostics
            ]),
            "message_semantic_fidelity": _rate([
                item.get("semantic_fidelity") for item in message_diagnostics
            ]),
            "message_observations": len(message_diagnostics),
        })
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="per-episode v0.3 JSONL ledgers from separately run batches")
    parser.add_argument("-o", "--output", type=Path, help="write JSON report (default: stdout)")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.replicates < 100:
        parser.error("--replicates must be at least 100")
    try:
        records = [record for path in args.inputs for record in read_jsonl(path)]
        report = private_match_report(records, replicates=args.replicates, seed=args.seed)
    except (OSError, RecordError, ValueError) as exc:
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
