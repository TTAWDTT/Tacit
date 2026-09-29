"""Validate and analyze Private Match v0.3 tlu.costs.v3 ledgers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.private_match_v0_3.bit_frontier import frontier as bit_frontier
from experiments.private_match_v0_3.generate_tasks import (
    bayes_accuracy_references, generate_episode, load_task_key, score_answer,
    task_key_id,
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


def validate_private_match_records(records: list[dict[str, Any]], *, task_key: bytes) -> None:
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
        _require(params == {"q": q, "candidate_count": q * q, "agent_count": 3,
                            "task_key_id": task_key_id(task_key)},
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
        expected_split = (
            "calibration" if condition == "full_information" else diag.get("split")
        )
        _require(expected_split in {"calibration", "development", "evaluation"},
                 f"{prefix}: unknown data split")
        _require(stratum.get("split") == expected_split,
                 f"{prefix}: split does not match diagnostics")

        episode = generate_episode(episode_id=episode_id, seed=seed, q=q,
                                   task_key=task_key)
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
                         seed: int = 1729, task_key: bytes,
                         selection_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_private_match_records(records, task_key=task_key)
    if any(
        (row.get("_normalized_stratum") or row["stratum"]).get("split") == "development"
        for row in records
    ):
        raise RecordError(
            "development records are selection-only and cannot enter evaluation reports"
        )
    report_records = records
    selection_setup: dict[str, Any] = {
        "status": "missing",
        "note": "The natural-language development selector setup is not included; NL efficiency comparisons are incomplete.",
    }
    if selection_manifest is not None:
        selection_setup = _validate_selection_manifest(
            selection_manifest, records, task_key=task_key
        )
        report_records = [dict(row) for row in records]
        selected_id = selection_setup["selected_protocol_id"]
        artifact = {
            "artifact_id": f"private-match-v0.3-nl-selector:{selection_setup['manifest_sha256']}",
            "one_time_bytes": selection_setup["raw"]["wire_bytes"],
            "one_time_model_calls": selection_setup["raw"]["model_calls"],
            "one_time_service_seconds": selection_setup["raw"]["model_service_seconds"],
            "one_time_wall_seconds": selection_setup["raw"]["wall_seconds"],
            "reuse_horizon": selection_setup["reuse_horizon_evaluation_episodes"],
            "one_time_tokens": {
                tokenizer: values["input_tokens"] + values["output_tokens"]
                for tokenizer, values in selection_setup["raw"]["model_tokens_by_tokenizer"].items()
            },
        }
        for row in report_records:
            if (row["protocol"].get("representation_id") == selected_id
                    and row["diagnostics"].get("condition") == "both_sources"
                    and row["diagnostics"].get("split") == "evaluation"):
                row["setup"] = [*row.get("setup", []), artifact]
    cost_summary = aggregate(report_records)
    frontier_records = report_records
    frontier_note = ""
    prereg = json.loads(Path(__file__).with_name("preregistration.json").read_text(encoding="utf-8"))
    nl_ids = set(prereg["protocol_freeze"]["development_natural_language_candidates"])
    if selection_manifest is None:
        frontier_records = [row for row in report_records
                            if row["protocol"].get("representation_id") not in nl_ids]
        frontier_note = (
            "Natural-language candidate points are omitted because the selection manifest was not supplied; "
            "their development selection costs are unknown, so treating them as zero would bias the frontier."
        )
    else:
        selected_id = selection_setup["selected_protocol_id"]
        frontier_records = [
            row for row in report_records
            if row["protocol"].get("representation_id") not in nl_ids
            or row["protocol"].get("representation_id") == selected_id
        ]
        frontier_note = (
            "The selected natural-language point includes amortized development-selector setup. "
            "The unselected NL candidate is omitted because its development results were used only for selection."
        )
    return {
        "schema_version": "tlu.private-match-report.v2",
        "experiment_id": EXPERIMENT_ID,
        "validation": "generated task reconstruction, exact answer scoring, message/parser diagnostics, routes, and call counts passed",
        "analytic_controls": _analytic_controls(records),
        "representation_diagnostics": _representation_diagnostics(records),
        "cost_summary": cost_summary,
        "natural_language_selection_setup": selection_setup,
        "paired_comparisons": paired_report(records, replicates=replicates, seed=seed),
        "empirical_frontiers": frontier_report(frontier_records) if frontier_records else {
            "schema_version": "tlu.frontier-report.v1",
            "groups": [],
            "excluded_reason": "No frontier points remain after removing NL candidates with unreported selector setup.",
        },
        "frontier_note": frontier_note,
        "claim_limit": "A feasibility pilot report is descriptive. It does not establish population superiority; inspect policy/decoder alignment and complete cost coverage for every comparison.",
    }


def _validate_selection_manifest(
    manifest: dict[str, Any], records: list[dict[str, Any]], *, task_key: bytes,
) -> dict[str, Any]:
    """Validate and expose the fixed development optimizer cost for the chosen NL arm."""
    _require(manifest.get("schema_version") == "tlu.private-match-nl-selection.v1",
             "unsupported NL selection manifest schema")
    _require(manifest.get("selection_split") == "development_only"
             and manifest.get("evaluation_data_used") is False,
             "NL selector must be development-only")
    _require(manifest.get("task_key_id") == task_key_id(task_key),
             "NL selector uses a different evaluator task key")
    _require(manifest.get("prompt_revision") == PROMPT_REVISION,
             "NL selector prompt revision differs from the registered protocol")
    prereg_path = Path(__file__).with_name("preregistration.json")
    protocols_path = Path(__file__).with_name("protocols.py")
    _require(manifest.get("preregistration_sha256") == hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
             "NL selector preregistration hash is stale")
    _require(manifest.get("protocols_py_sha256") == hashlib.sha256(protocols_path.read_bytes()).hexdigest(),
             "NL selector protocol source hash is stale")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    pilot = prereg["default_pilot"]
    freeze = prereg["protocol_freeze"]
    candidates = set(freeze["development_natural_language_candidates"])
    selected = manifest.get("selected_protocol_id")
    _require(selected in candidates, "NL selector chose an unregistered candidate")
    expected_dev_ids = {
        f"pmt3-{seed:012d}"
        for seed in range(pilot["development_seed_start"],
                          pilot["development_seed_start"] + pilot["development_episodes"])
    }
    _require(set(manifest.get("development_episode_ids", [])) == expected_dev_ids,
             "NL selector development episodes differ from the preregistered shard")
    expected_eval_ids = {
        f"pmt3-{seed:012d}"
        for seed in range(pilot["evaluation_seed_start"],
                          pilot["evaluation_seed_start"] + pilot["evaluation_episodes"])
    }
    selected_rows = [
        row for row in records
        if row["protocol"].get("representation_id") == selected
        and row["diagnostics"].get("condition") == "both_sources"
    ]
    _require({row["episode_id"] for row in selected_rows} == expected_eval_ids,
             "NL selection setup can only be attached to the complete preregistered evaluation shard")
    _require(all(row["diagnostics"].get("split") == "evaluation" for row in selected_rows),
             "NL selection setup may only be attached to evaluation records")
    signatures = {
        tuple((call.get("agent"), call.get("model"), call.get("tokenizer"))
              for call in row["model_calls"])
        for row in selected_rows
    }
    expected_signature = tuple(tuple(item) for item in manifest.get("model_and_tokenizer_signature", []))
    _require(len(signatures) == 1 and next(iter(signatures)) == expected_signature,
             "NL selector and evaluation model/tokenizer signatures differ")
    setup = manifest.get("optimizer_setup_cost")
    _require(isinstance(setup, dict) and setup.get("account_as_optimizer_setup") is True,
             "NL selector manifest is missing optimizer setup accounting")
    horizon = freeze.get("development_selection_reuse_horizon_evaluation_episodes")
    _require(setup.get("reuse_horizon_evaluation_episodes") == horizon
             and horizon == pilot["evaluation_episodes"],
             "NL selector reuse horizon differs from the preregistered pilot horizon")
    integer_fields = (
        "development_model_calls", "development_wire_bytes",
        "development_model_input_tokens", "development_model_output_tokens",
    )
    for field in integer_fields:
        value = setup.get(field)
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                 f"NL selector setup has invalid {field}")
    tokenizer_counts = setup.get("development_model_tokens_by_tokenizer")
    _require(isinstance(tokenizer_counts, dict), "NL selector setup is missing per-tokenizer totals")
    expected_tokenizers = {item[2] for item in expected_signature}
    _require(set(tokenizer_counts) == expected_tokenizers,
             "NL selector per-tokenizer totals differ from its model signature")
    for tokenizer, values in tokenizer_counts.items():
        _require(isinstance(values, dict), f"NL selector totals for {tokenizer} must be an object")
        for field in ("input_tokens", "output_tokens"):
            value = values.get(field)
            _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                     f"NL selector has invalid {tokenizer} {field}")
    for field in ("development_model_service_seconds", "development_wall_seconds"):
        value = setup.get(field)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value) and value >= 0,
                 f"NL selector setup has invalid {field}")
    raw = {
        "model_calls": setup["development_model_calls"],
        "wire_bytes": setup["development_wire_bytes"],
        "model_tokens_by_tokenizer": tokenizer_counts,
        "model_service_seconds": setup.get("development_model_service_seconds"),
        "wall_seconds": setup.get("development_wall_seconds"),
    }
    return {
        "status": "included",
        "selected_protocol_id": selected,
        "manifest_sha256": hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).hexdigest(),
        "reuse_horizon_evaluation_episodes": horizon,
        "raw": raw,
        "amortized_per_evaluation_episode": {
            "model_calls": raw["model_calls"] / horizon,
            "wire_bytes": raw["wire_bytes"] / horizon,
            "model_tokens_by_tokenizer": {
                tokenizer: (counts["input_tokens"] + counts["output_tokens"]) / horizon
                for tokenizer, counts in tokenizer_counts.items()
            },
            "model_service_seconds": (
                None if raw["model_service_seconds"] is None
                else raw["model_service_seconds"] / horizon
            ),
            "wall_seconds": None if raw["wall_seconds"] is None else raw["wall_seconds"] / horizon,
        },
        "limitation": "Fixed setup is reported separately from episode bootstrap uncertainty; amortization uses the preregistered eight-episode pilot horizon, not a deployment lifetime. Token costs remain tokenizer-indexed and the scalar token frontier is suppressed for heterogeneous tokenizer units.",
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
    parser.add_argument("--task-key-file", type=Path,
                        default=Path(".cache/private_match_v0_3/task.key"),
                        help="evaluator-only task key used to reconstruct private targets")
    parser.add_argument("--nl-selection-manifest", type=Path,
                        help="development-only natural-language selection manifest; required for a complete selected-NL cost frontier")
    args = parser.parse_args(argv)
    if args.replicates < 100:
        parser.error("--replicates must be at least 100")
    try:
        records = [record for path in args.inputs for record in read_jsonl(path)]
        task_key = load_task_key(args.task_key_file)
        selection_manifest = (
            json.loads(args.nl_selection_manifest.read_text(encoding="utf-8"))
            if args.nl_selection_manifest else None
        )
        report = private_match_report(records, replicates=args.replicates,
                                      seed=args.seed, task_key=task_key,
                                      selection_manifest=selection_manifest)
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
