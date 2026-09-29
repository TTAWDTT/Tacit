"""Freeze a validation-selected Pareto set of shared protocol cards.

This selector never contacts a model and never opens test-stage role files. It
accepts complete, same-episode validation runs for each frozen candidate card,
checks their runner manifests and hashes, and emits a hash-bound shortlist.
Optional induction manifests make measured protocol-generation setup costs
visible and bind every evaluated card to its source trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.emergent_ood_v0_4.episodes import SCHEMA as EPISODE_SCHEMA
from experiments.emergent_ood_v0_4.induce_protocol_cards import (
    OUTPUT_SCHEMA as INDUCER_SCHEMA, PROTOCOL_FAMILIES,
)
from experiments.emergent_ood_v0_4.runner import (
    EXPERIMENT_ID, ROOT, SCORER_ID, _meaning_id_from_values, load_protocol_card,
)
from experiments.emergent_ood_v0_4.split import build_split


SPEC_SCHEMA = "tlu.emergent-ood-protocol-frontier-candidates.v1"
FREEZE_SCHEMA = "tlu.emergent-ood-protocol-frontier-freeze.v1"
RUN_MANIFEST_SCHEMA = "tlu.emergent-ood-run-manifest.v0.4"


def _project_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("all selector input and output paths must stay inside the project") from exc
    return resolved


def _spec_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    return _project_path(candidate if candidate.is_absolute() else base / candidate)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read valid JSONL: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain non-empty object records: {path}")
    return rows, payload


def _validation_episode_index(bundle_dir: Path, *, split_seed: int) -> tuple[dict[str, str], dict[str, Any]]:
    """Verify and read validation roles only; deliberately never opens train/test roles."""
    bundle_dir = _project_path(bundle_dir)
    manifest_path = _project_path(bundle_dir / "manifest.json")
    manifest = _json(manifest_path, "episode bundle manifest")
    if manifest.get("schema") != EPISODE_SCHEMA:
        raise ValueError("unsupported episode bundle schema")
    split = build_split(seed=split_seed)
    if manifest.get("split_sha256") != split["split_sha256"]:
        raise ValueError("episode bundle does not match the declared validation split seed")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("episode bundle manifest is missing its file table")

    role_rows: dict[str, list[dict[str, Any]]] = {}
    for role in ("sender", "receiver", "gold"):
        entry = manifest["files"].get(role, {}).get("validation")
        expected_name = f"{role}_validation.jsonl"
        if not isinstance(entry, dict) or entry.get("file") != expected_name:
            raise ValueError(f"episode manifest has an invalid {role} validation entry")
        role_path = _project_path(bundle_dir / expected_name)
        rows, payload = _jsonl(role_path)
        if entry.get("sha256") != _sha256_bytes(payload) or entry.get("records") != len(rows):
            raise ValueError(f"{role} validation ledger hash or count does not match its manifest")
        role_rows[role] = rows

    sender, receiver, gold = (role_rows[name] for name in ("sender", "receiver", "gold"))
    if not (len(sender) == len(receiver) == len(gold)):
        raise ValueError("validation role ledger counts differ")
    if manifest.get("episodes_per_stage", {}).get("validation") != len(gold):
        raise ValueError("manifest validation episode count is inconsistent")
    expected: dict[str, str] = {}
    candidate_tables: dict[str, list[dict[str, Any]]] = {}
    targets_by_set: dict[str, list[str]] = {}
    attributes = manifest.get("attributes")
    if not isinstance(attributes, list) or not attributes:
        raise ValueError("episode manifest does not declare the ordered attribute vocabulary")
    heldout_ids = set(split["held_out_meaning_ids"])
    meaning_ids_by_values = {
        tuple(row["values"]): row["meaning_id"] for row in split["meanings"]
    }
    for sender_row, receiver_row, gold_row in zip(sender, receiver, gold):
        episode_id = gold_row.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in expected:
            raise ValueError("validation gold ledger has missing or duplicate episode IDs")
        if any(key in sender_row or key in receiver_row for key in ("meaning_id", "candidate_set_id", "target", "gold")):
            raise ValueError("validation evaluator metadata leaked into a model-facing role view")
        set_id = gold_row.get("candidate_set_id")
        candidates = receiver_row.get("candidates")
        if not isinstance(candidates, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("attributes"), dict)
            for item in candidates
        ):
            raise ValueError("validation receiver candidate table is malformed")
        candidate_ids = [item.get("candidate_id") for item in candidates]
        if not isinstance(set_id, str) or len(candidate_ids) != manifest.get("k"):
            raise ValueError("validation gold or candidate table is malformed")
        if (
            any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or gold_row.get("candidate_id") not in candidate_ids
        ):
            raise ValueError("validation candidate IDs are duplicated or omit the target")
        table = candidate_tables.setdefault(set_id, candidates)
        if table != candidates:
            raise ValueError("validation candidate table changes within a candidate set")
        target = next(row for row in candidates if row["candidate_id"] == gold_row["candidate_id"])
        private_meaning = sender_row.get("private_meaning")
        if (
            not isinstance(private_meaning, dict)
            or set(private_meaning) != set(attributes)
            or target.get("attributes") != private_meaning
            or set(target["attributes"]) != set(attributes)
            or gold_row.get("meaning_id") != _meaning_id_from_values([private_meaning[name] for name in attributes])
            or tuple(private_meaning[name] for name in attributes) not in meaning_ids_by_values
            or meaning_ids_by_values[tuple(private_meaning[name] for name in attributes)] not in heldout_ids
            or any(
                tuple(candidate["attributes"].get(name) for name in attributes) not in meaning_ids_by_values
                or meaning_ids_by_values[tuple(candidate["attributes"].get(name) for name in attributes)] not in heldout_ids
                for candidate in candidates
            )
        ):
            raise ValueError("validation sender meaning, candidate table, and evaluator label do not agree")
        targets_by_set.setdefault(set_id, []).append(gold_row["candidate_id"])
        expected[episode_id] = set_id
    if len({set_id for set_id in expected.values()}) != manifest.get("sets_per_stage"):
        raise ValueError("validation candidate-set count does not match its manifest")
    if any(
        len(targets) != manifest.get("k") or set(targets) != set(candidate_tables[set_id][i]["candidate_id"] for i in range(len(candidate_tables[set_id])))
        for set_id, targets in targets_by_set.items()
    ):
        raise ValueError("validation target allocation is not exactly balanced within every candidate set")
    return expected, {
        "manifest": manifest,
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "bundle_dir": str(bundle_dir),
        "split": split,
    }


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _candidate_run(
    *, protocol_id: str, card_path: Path, ledger_paths: list[Path], expected_ids: dict[str, str],
    bundle_info: dict[str, Any], split_seed: int,
) -> dict[str, Any]:
    card, card_digest = load_protocol_card(card_path)
    if card["protocol_id"] != protocol_id:
        raise ValueError("candidate spec ID differs from the card's protocol_id")
    if not ledger_paths:
        raise ValueError(f"candidate {protocol_id!r} requires at least one validation ledger")

    collected: dict[str, dict[str, Any]] = {}
    batch_ranges: list[tuple[int, int]] = []
    batch_manifest_digests: list[str] = []
    ledger_digests: list[str] = []
    reference_manifest: dict[str, Any] | None = None
    for ledger_path in ledger_paths:
        ledger_path = _project_path(ledger_path)
        rows, payload = _jsonl(ledger_path)
        manifest_path = _project_path(ledger_path.with_suffix(ledger_path.suffix + ".manifest.json"))
        run_manifest = _json(manifest_path, "validation run manifest")
        if (
            run_manifest.get("schema") != RUN_MANIFEST_SCHEMA
            or run_manifest.get("experiment_id") != EXPERIMENT_ID
            or run_manifest.get("results_file") != ledger_path.name
            or run_manifest.get("results_sha256") != _sha256_bytes(payload)
            or run_manifest.get("result_records") != len(rows)
            or run_manifest.get("stage") != "validation"
            or run_manifest.get("conditions") != ["shared_protocol_card"]
            or run_manifest.get("protocol_card_sha256") != card_digest
            or run_manifest.get("split_seed") != split_seed
            or run_manifest.get("split_sha256") != bundle_info["split"]["split_sha256"]
            or run_manifest.get("input_episode_manifest_sha256") != bundle_info["manifest_sha256"]
            or run_manifest.get("task_seed") != bundle_info["manifest"].get("task_seed")
            or run_manifest.get("task_key_id") != bundle_info["manifest"].get("task_key_id")
        ):
            raise ValueError(f"candidate {protocol_id!r} has a mismatched or invalid validation manifest")
        _integer(run_manifest.get("communication_budget_bytes"), "communication_budget_bytes")
        if reference_manifest is None:
            reference_manifest = run_manifest
        else:
            invariant_fields = (
                "experiment_id", "stage", "conditions", "split_seed", "split_sha256", "task_seed",
                "task_key_id", "input_episode_manifest_sha256", "communication_budget_bytes",
                "sender_model", "receiver_model", "sender_tokenizer_id", "receiver_tokenizer_id",
                "model_population_id",
            )
            if any(run_manifest.get(field) != reference_manifest.get(field) for field in invariant_fields):
                raise ValueError("validation batches for one card use inconsistent task/model settings")

        offset = _integer(run_manifest.get("candidate_set_offset"), "candidate_set_offset")
        count = _integer(run_manifest.get("candidate_sets"), "candidate_sets", minimum=1)
        batch_ranges.append((offset, offset + count))
        batch_manifest_digests.append(_sha256_bytes(manifest_path.read_bytes()))
        ledger_digests.append(_sha256_bytes(payload))
        for row in rows:
            episode_id = row.get("episode_id")
            if not isinstance(episode_id, str) or episode_id in collected:
                raise ValueError(f"candidate {protocol_id!r} has duplicate or missing episode IDs")
            outcome = row.get("outcome")
            stratum = row.get("stratum")
            protocol = row.get("protocol")
            if not all(isinstance(value, dict) for value in (outcome, stratum, protocol)):
                raise ValueError("validation result is missing its outcome, stratum, or protocol object")
            if (
                row.get("experiment_id") != EXPERIMENT_ID
                or row.get("stage") != "validation"
                or row.get("condition") != "shared_protocol_card"
                or row.get("protocol_id") != protocol_id
                or protocol.get("policy_id") != "shared_protocol_card"
                or protocol.get("code_id") != protocol_id
                or row.get("split_seed") != split_seed
                or row.get("task_seed") != bundle_info["manifest"].get("task_seed")
                or row.get("communication_budget_bytes") != run_manifest.get("communication_budget_bytes")
                or episode_id not in expected_ids
                or row.get("candidate_set_id") != expected_ids[episode_id]
                or stratum.get("task_id") != "four-attribute-higher-order-meaning-matching-v1"
                or stratum.get("scorer_id") != SCORER_ID
                or not isinstance(outcome.get("exact_selection"), bool)
            ):
                raise ValueError(f"candidate {protocol_id!r} row is not a matching validation result")
            calls = row.get("model_calls")
            if not isinstance(calls, list) or len(calls) != 2 or any(not isinstance(call, dict) for call in calls):
                raise ValueError("each shared-card validation episode must have exactly two model calls")
            if stratum.get("model_population_id") != run_manifest.get("model_population_id"):
                raise ValueError("validation row and manifest model population differ")
            cost = row.get("costs")
            if not isinstance(cost, dict):
                raise ValueError("validation row is missing a cost object")
            _integer(cost.get("application_wire_bytes"), "application_wire_bytes")
            collected[episode_id] = row

    if reference_manifest is None:
        raise ValueError(f"candidate {protocol_id!r} has no validation run manifest")
    available_sets = _integer(bundle_info["manifest"].get("sets_per_stage"), "sets_per_stage", minimum=1)
    expected_ranges = sorted(batch_ranges)
    cursor = 0
    for start, end in expected_ranges:
        if start != cursor or end > available_sets:
            raise ValueError("validation batches overlap, omit candidate sets, or exceed the declared stage")
        cursor = end
    if cursor != available_sets:
        raise ValueError("validation ledgers do not cover every declared validation candidate set")
    if set(collected) != set(expected_ids):
        raise ValueError(f"candidate {protocol_id!r} does not contain every validation episode exactly once")

    successes = sum(row["outcome"]["exact_selection"] is True for row in collected.values())
    wire_bytes = sum(row["costs"]["application_wire_bytes"] for row in collected.values())
    model_calls = 0
    token_totals: dict[str, dict[str, int]] = {}
    tokens_complete = True
    call_signature: set[tuple[str, str, str]] = set()
    service_seconds = 0.0
    service_complete = True
    for row in collected.values():
        calls = row["model_calls"]
        model_calls += len(calls)
        for call in calls:
            agent, model, tokenizer = call.get("agent"), call.get("model"), call.get("tokenizer")
            if not all(isinstance(value, str) and value for value in (agent, model, tokenizer)):
                raise ValueError("development records require model/tokenizer IDs on every call")
            call_signature.add((agent, model, tokenizer))
            key = f"{agent}|{model}|{tokenizer}"
            counts = token_totals.setdefault(key, {"input_tokens": 0, "output_tokens": 0})
            for field in ("input_tokens", "output_tokens"):
                value = call.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    tokens_complete = False
                else:
                    counts[field] += value
            service = call.get("service_seconds")
            if isinstance(service, bool) or not isinstance(service, (int, float)) or not math.isfinite(service) or service < 0:
                service_complete = False
            else:
                service_seconds += float(service)
    if {row[0] for row in call_signature} != {"sender", "receiver"} or len(call_signature) != 2:
        raise ValueError("each candidate must use exactly one frozen sender and one frozen receiver stratum")
    expected_call_signature = {
        ("sender", reference_manifest.get("sender_model"), reference_manifest.get("sender_tokenizer_id")),
        ("receiver", reference_manifest.get("receiver_model"), reference_manifest.get("receiver_tokenizer_id")),
    }
    if call_signature != expected_call_signature:
        raise ValueError("validation model calls do not match the run manifest model/tokenizer IDs")

    return {
        "protocol_id": protocol_id,
        "protocol_card_sha256": card_digest,
        "protocol_card_bytes": card_path.stat().st_size,
        "card_path": card_path.relative_to(ROOT).as_posix(),
        "communication_budget_bytes": reference_manifest["communication_budget_bytes"],
        "model_population_id": reference_manifest["model_population_id"],
        "validation_episode_count": len(collected),
        "validation_successes": successes,
        "validation_success_rate": successes / len(collected),
        "application_wire_bytes_total": wire_bytes,
        "application_wire_bytes_per_episode": wire_bytes / len(collected),
        "model_calls": model_calls,
        "input_output_tokens_by_agent_model_tokenizer": token_totals if tokens_complete else None,
        "token_usage_complete": tokens_complete,
        "service_seconds_total": service_seconds if service_complete else None,
        "ledger_sha256": ledger_digests,
        "run_manifest_sha256": batch_manifest_digests,
        "validation_episode_ids_sha256": _sha256_bytes("\n".join(sorted(collected)).encode("utf-8")),
        "call_signature": sorted([list(row) for row in call_signature]),
        "_token_totals": token_totals if tokens_complete else None,
    }


def _pareto_protocols(candidates: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    token_complete = all(candidate["token_usage_complete"] for candidate in candidates)
    for candidate in candidates:
        metrics: dict[str, float] = {
            "validation_success_rate": candidate["validation_success_rate"],
            "application_wire_bytes_per_episode": candidate["application_wire_bytes_per_episode"],
        }
        if token_complete:
            for label, counts in candidate["_token_totals"].items():
                for field, total in counts.items():
                    metrics[f"{field}_per_episode|{label}"] = total / candidate["validation_episode_count"]
        candidate["_pareto_metrics"] = metrics

    # Tokenizer/call populations must be identical before token axes can be compared.
    if token_complete and any(set(item["_pareto_metrics"]) != set(candidates[0]["_pareto_metrics"]) for item in candidates[1:]):
        token_complete = False
        for candidate in candidates:
            candidate["_pareto_metrics"] = {
                key: value for key, value in candidate["_pareto_metrics"].items()
                if key in {"validation_success_rate", "application_wire_bytes_per_episode"}
            }
    costs = [key for key in candidates[0]["_pareto_metrics"] if key != "validation_success_rate"]

    nondominated: list[str] = []
    dominated: list[str] = []
    for candidate in candidates:
        metrics = candidate["_pareto_metrics"]
        is_dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_metrics = other["_pareto_metrics"]
            no_worse = (
                other_metrics["validation_success_rate"] >= metrics["validation_success_rate"]
                and all(other_metrics[key] <= metrics[key] for key in costs)
            )
            strictly_better = (
                other_metrics["validation_success_rate"] > metrics["validation_success_rate"]
                or any(other_metrics[key] < metrics[key] for key in costs)
            )
            if no_worse and strictly_better:
                is_dominated = True
                break
        (dominated if is_dominated else nondominated).append(candidate["protocol_id"])
    dimensions = ["validation_success_rate|max", *(f"{name}|min" for name in costs)]
    return sorted(nondominated), dimensions


def _induction_setup(
    *, paths: list[str], base: Path, bundle_info: dict[str, Any], split_seed: int,
    expected_cards: dict[str, str],
) -> dict[str, Any]:
    if not paths:
        return {
            "status": "not_supplied",
            "model_calls": None,
            "input_tokens_by_model_tokenizer": None,
            "output_tokens_by_model_tokenizer": None,
            "service_seconds": None,
            "prompt_completion_utf8_bytes": None,
            "induction_manifest_sha256": [],
            "candidate_protocol_families": {},
        }
    total_calls = 0
    input_tokens: dict[str, int] = {}
    output_tokens: dict[str, int] = {}
    tokens_complete = True
    service_seconds = 0.0
    service_complete = True
    prompt_completion_bytes = 0
    manifests_seen: set[str] = set()
    verified_cards: dict[str, tuple[str, str]] = {}
    manifest_digests: list[str] = []

    for value in paths:
        manifest_path = _spec_path(base, value)
        manifest_payload = manifest_path.read_bytes()
        digest = _sha256_bytes(manifest_payload)
        if digest in manifests_seen:
            raise ValueError("induction manifests must be unique")
        manifests_seen.add(digest)
        manifest = _json(manifest_path, "protocol induction manifest")
        if (
            manifest.get("schema") != INDUCER_SCHEMA
            or manifest.get("mode") != "execute"
            or manifest.get("inference_started") is not True
            or manifest.get("model_calls") != 1
            or manifest.get("split_seed") != split_seed
            or manifest.get("split_sha256") != bundle_info["split"]["split_sha256"]
            or manifest.get("task_key_id") != bundle_info["manifest"].get("task_key_id")
        ):
            raise ValueError("induction manifest is incomplete or uses a different split/task key")
        model = manifest.get("inducer_model")
        tokenizer = manifest.get("tokenizer_id")
        if not isinstance(model, str) or not model or not isinstance(tokenizer, str) or not tokenizer:
            raise ValueError("induction manifest lacks its model/tokenizer identity")
        family = manifest.get("protocol_family")
        if family not in PROTOCOL_FAMILIES:
            raise ValueError("induction manifest lacks a supported protocol family")
        call_key = f"protocol_inducer|{family}|{model}|{tokenizer}"
        total_calls += manifest["model_calls"]
        for source_name, totals in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
            count = manifest.get(source_name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                tokens_complete = False
            else:
                totals[call_key] = totals.get(call_key, 0) + count
        service = manifest.get("service_seconds")
        if isinstance(service, bool) or not isinstance(service, (int, float)) or not math.isfinite(service) or service < 0:
            service_complete = False
        else:
            service_seconds += float(service)

        for field, digest_field, size_field in (
            ("prompt_path", "prompt_sha256", "prompt_utf8_bytes"),
            ("completion_path", "completion_sha256", "completion_utf8_bytes"),
        ):
            rel = manifest.get(field)
            if not isinstance(rel, str):
                raise ValueError("induction manifest is missing its prompt/completion artifact path")
            artifact_path = _project_path(ROOT / rel)
            payload = artifact_path.read_bytes()
            if manifest.get(digest_field) != _sha256_bytes(payload) or manifest.get(size_field) != len(payload):
                raise ValueError("induction prompt/completion hash or byte count is invalid")
            prompt_completion_bytes += len(payload)

        entries = manifest.get("candidate_cards")
        if not isinstance(entries, list) or not entries:
            raise ValueError("induction manifest has no candidate card records")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("protocol_id"), str) or not isinstance(entry.get("path"), str):
                raise ValueError("induction candidate card record is malformed")
            protocol_id = entry["protocol_id"]
            card_path = _project_path(ROOT / entry["path"])
            card_payload = card_path.read_bytes()
            card, card_digest = load_protocol_card(card_path)
            if (
                card["protocol_id"] != protocol_id
                or card_digest != entry.get("sha256")
                or entry.get("bytes") != len(card_payload)
            ):
                raise ValueError("induced protocol card does not match its content hash/identity")
            if protocol_id in verified_cards and verified_cards[protocol_id] != (card_digest, family):
                raise ValueError("induction manifests assign conflicting content/family to one protocol ID")
            verified_cards[protocol_id] = (card_digest, family)
        manifest_digests.append(digest)

    for protocol_id, expected_digest in expected_cards.items():
        actual = verified_cards.get(protocol_id)
        if actual is None or actual[0] != expected_digest:
            raise ValueError(f"candidate {protocol_id!r} is not covered by the supplied induction manifest(s)")
    return {
        "status": "verified_induction_manifests",
        "model_calls": total_calls,
        "input_tokens_by_model_tokenizer": input_tokens if tokens_complete else None,
        "output_tokens_by_model_tokenizer": output_tokens if tokens_complete else None,
        "token_usage_complete": tokens_complete,
        "service_seconds": service_seconds if service_complete else None,
        "prompt_completion_utf8_bytes": prompt_completion_bytes,
        "induction_manifest_sha256": manifest_digests,
        "candidate_protocol_families": {
            protocol_id: verified_cards[protocol_id][1] for protocol_id in expected_cards
        },
    }


def freeze_protocol_frontier(spec_path: Path) -> dict[str, Any]:
    spec_path = _project_path(spec_path)
    spec = _json(spec_path, "candidate specification")
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("unsupported protocol candidate spec schema")
    base = spec_path.parent
    split_seed = _integer(spec.get("split_seed"), "split_seed")
    reuse_horizon = _integer(spec.get("reuse_horizon_evaluation_episodes"), "reuse_horizon_evaluation_episodes", minimum=1)
    bundle_path = _spec_path(base, spec.get("input_dir", ""))
    expected_ids, bundle_info = _validation_episode_index(bundle_path, split_seed=split_seed)
    candidate_specs = spec.get("candidates")
    if not isinstance(candidate_specs, list) or len(candidate_specs) < 2:
        raise ValueError("at least two frozen candidate cards are required")

    candidates = []
    seen_ids: set[str] = set()
    for item in candidate_specs:
        if not isinstance(item, dict) or not isinstance(item.get("protocol_id"), str):
            raise ValueError("each candidate must declare a protocol_id, card, and validation_ledgers")
        protocol_id = item["protocol_id"]
        if protocol_id in seen_ids:
            raise ValueError("candidate protocol IDs must be unique")
        seen_ids.add(protocol_id)
        ledgers = item.get("validation_ledgers")
        if not isinstance(ledgers, list) or not ledgers or any(not isinstance(path, str) for path in ledgers):
            raise ValueError(f"candidate {protocol_id!r} needs a list of validation ledger paths")
        candidates.append(_candidate_run(
            protocol_id=protocol_id,
            card_path=_spec_path(base, item.get("card", "")),
            ledger_paths=[_spec_path(base, path) for path in ledgers],
            expected_ids=expected_ids,
            bundle_info=bundle_info,
            split_seed=split_seed,
        ))

    induction_paths = spec.get("induction_manifests", [])
    if not isinstance(induction_paths, list) or any(not isinstance(path, str) for path in induction_paths):
        raise ValueError("induction_manifests must be a list of project-relative paths")
    induction_setup = _induction_setup(
        paths=induction_paths,
        base=base,
        bundle_info=bundle_info,
        split_seed=split_seed,
        expected_cards={candidate["protocol_id"]: candidate["protocol_card_sha256"] for candidate in candidates},
    )
    for candidate in candidates:
        candidate["protocol_family"] = induction_setup["candidate_protocol_families"].get(candidate["protocol_id"])

    signatures = {tuple(tuple(call) for call in candidate["call_signature"]) for candidate in candidates}
    budgets = {candidate["communication_budget_bytes"] for candidate in candidates}
    model_populations = {candidate["model_population_id"] for candidate in candidates}
    if len(signatures) != 1 or None in budgets or len(budgets) != 1 or None in model_populations or len(model_populations) != 1:
        raise ValueError("candidate cards must use the same model/tokenizer population and wire budget")
    episode_hashes = {candidate["validation_episode_ids_sha256"] for candidate in candidates}
    if len(episode_hashes) != 1:
        raise ValueError("candidate cards are not paired over identical validation episodes")

    pareto_ids, pareto_dimensions = _pareto_protocols(candidates)
    total_dev_calls = sum(candidate["model_calls"] for candidate in candidates)
    input_setup: dict[str, int] | None = {}
    output_setup: dict[str, int] | None = {}
    service_setup = 0.0
    service_setup_complete = True
    for candidate in candidates:
        if input_setup is None:
            pass
        elif candidate["_token_totals"] is None:
            input_setup = output_setup = None
        else:
            for label, totals in candidate["_token_totals"].items():
                input_setup[label] = input_setup.get(label, 0) + totals["input_tokens"]
                output_setup[label] = output_setup.get(label, 0) + totals["output_tokens"]
        if candidate["service_seconds_total"] is None:
            service_setup_complete = False
        else:
            service_setup += candidate["service_seconds_total"]
    for candidate in candidates:
        candidate.pop("_token_totals", None)
        candidate.pop("_pareto_metrics", None)
        candidate.pop("call_signature", None)

    development_wire_bytes = sum(candidate["application_wire_bytes_total"] for candidate in candidates)
    amortized_development_cost: dict[str, Any] = {
        "model_calls": total_dev_calls / reuse_horizon,
        "application_wire_bytes": development_wire_bytes / reuse_horizon,
        "model_input_tokens_by_agent_model_tokenizer": (
            {label: value / reuse_horizon for label, value in input_setup.items()}
            if input_setup is not None else None
        ),
        "model_output_tokens_by_agent_model_tokenizer": (
            {label: value / reuse_horizon for label, value in output_setup.items()}
            if output_setup is not None else None
        ),
        "model_service_seconds": service_setup / reuse_horizon if service_setup_complete else None,
    }
    amortized_induction_cost = {
        "model_calls": induction_setup["model_calls"] / reuse_horizon if induction_setup["model_calls"] is not None else None,
        "application_wire_bytes": None,
        "prompt_completion_utf8_bytes": (
            induction_setup["prompt_completion_utf8_bytes"] / reuse_horizon
            if induction_setup["prompt_completion_utf8_bytes"] is not None else None
        ),
        "model_input_tokens_by_model_tokenizer": (
            {key: value / reuse_horizon for key, value in induction_setup["input_tokens_by_model_tokenizer"].items()}
            if induction_setup["input_tokens_by_model_tokenizer"] is not None else None
        ),
        "model_output_tokens_by_model_tokenizer": (
            {key: value / reuse_horizon for key, value in induction_setup["output_tokens_by_model_tokenizer"].items()}
            if induction_setup["output_tokens_by_model_tokenizer"] is not None else None
        ),
        "model_service_seconds": induction_setup["service_seconds"] / reuse_horizon if induction_setup["service_seconds"] is not None else None,
    }
    return {
        "schema": FREEZE_SCHEMA,
        "selection_split": "validation_only",
        "selection_rule": "retain non-dominated candidates over validation exact success, application wire bytes, and complete per-tokenizer input/output token costs; do not choose a single winner",
        "pareto_protocol_ids": pareto_ids,
        "pareto_dimensions": pareto_dimensions,
        "candidate_metrics": candidates,
        "evaluation_data_used": False,
        "test_stage_files_opened": False,
        "validation_episode_ids_sha256": next(iter(episode_hashes)),
        "validation_episode_count": len(expected_ids),
        "split_seed": split_seed,
        "split_sha256": bundle_info["split"]["split_sha256"],
        "input_episode_manifest_sha256": bundle_info["manifest_sha256"],
        "communication_budget_bytes": next(iter(budgets)),
        "reuse_horizon_evaluation_episodes": reuse_horizon,
        "selection_setup_cost": {
            "candidate_protocols_evaluated": len(candidates),
            "validation_candidate_model_calls": total_dev_calls,
            "model_input_tokens_by_agent_model_tokenizer": input_setup,
            "model_output_tokens_by_agent_model_tokenizer": output_setup,
            "application_wire_bytes": development_wire_bytes,
            "model_service_seconds": service_setup if service_setup_complete else None,
            "amortized_validation_candidate_evaluation_cost_per_reuse_episode": amortized_development_cost,
            "protocol_induction": induction_setup,
            "amortized_protocol_induction_cost_per_reuse_episode": amortized_induction_cost,
        },
        "limits": [
            "validation outcomes select a Pareto shortlist and are not confirmatory evidence",
            "candidate cards must be generated or revised without using validation/test target-answer pairs",
            "the selector verifies artifact/run hashes and split coverage but cannot prove semantic absence of hidden-target leakage in card text",
            "induction traces are optional; without them protocol discovery/generation costs remain unaccounted",
            "inference service time is reported per candidate but excluded from dominance because endpoint load can vary",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="project-local candidate spec JSON")
    parser.add_argument("--output", type=Path, required=True, help="project-local freeze manifest JSON")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        output = _project_path(args.output)
        if output.exists() and not args.force:
            raise ValueError(f"refusing to overwrite {output}; pass --force explicitly")
        result = freeze_protocol_frontier(args.spec)
        spec_path = _project_path(args.spec)
        spec = _json(spec_path, "candidate specification")
        base = spec_path.parent
        bundle_manifest_path = _project_path(_spec_path(base, spec.get("input_dir", "")) / "manifest.json")
        protected = {spec_path, bundle_manifest_path}
        for role in ("sender", "receiver", "gold"):
            protected.add(_project_path(_spec_path(base, spec["input_dir"]) / f"{role}_validation.jsonl"))
        for item in spec.get("candidates", []):
            protected.add(_spec_path(base, item["card"]))
            for ledger in item["validation_ledgers"]:
                ledger_path = _spec_path(base, ledger)
                protected.add(ledger_path)
                protected.add(_project_path(ledger_path.with_suffix(ledger_path.suffix + ".manifest.json")))
        for induction_manifest in spec.get("induction_manifests", []):
            induction_path = _spec_path(base, induction_manifest)
            protected.add(induction_path)
            trace = _json(induction_path, "protocol induction manifest")
            for field in ("prompt_path", "completion_path"):
                if isinstance(trace.get(field), str):
                    protected.add(_project_path(ROOT / trace[field]))
            for entry in trace.get("candidate_cards", []):
                if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                    protected.add(_project_path(ROOT / entry["path"]))
        if output in protected:
            raise ValueError("freeze manifest output cannot overwrite any selection input")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "freeze_manifest": str(output),
        "pareto_protocol_ids": result["pareto_protocol_ids"],
        "validation_episode_count": result["validation_episode_count"],
        "test_stage_files_opened": result["test_stage_files_opened"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
