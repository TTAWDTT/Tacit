"""Build hash-bound, training-only exact-score feedback for NL prompt search.

This utility reads only the episode manifest and the three train ledgers. It
never opens validation or test episode files. Its output may be supplied to
the plain-English card inducer for a later search round.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.emergent_ood_v0_4.episodes import PROJECT_ROOT, SCHEMA as EPISODE_SCHEMA
from experiments.emergent_ood_v0_4.episodes import verify_ledgers
from experiments.emergent_ood_v0_4.runner import (
    CALLS_PER_EPISODE,
    EXPERIMENT_ID,
    MAX_MODEL_CALLS_PER_BATCH,
    SCORER_ID,
)
from experiments.emergent_ood_v0_4.split import build_split_from_spec, split_task_id


FEEDBACK_SCHEMA = "tlu.emergent-ood-nl-feedback.v0.2"
RUN_SCHEMA = "tlu.emergent-ood-run-manifest.v0.4"
RUN_RESULT_SCHEMA = "tlu.emergent-ood-run.v0.4"
MAX_FAILURES = 24
MAX_TEXT_BYTES = 4096


def _project_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("feedback inputs and outputs must stay inside the project") from exc
    return resolved


def _json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _project_path(path).read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, payload


def _jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSONL") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must be JSON objects")
    return rows


def _load_train_bundle(directory: Path, *, split_seed: int) -> tuple[dict[str, Any], dict[str, Any], str]:
    directory = _project_path(directory)
    manifest_path = _project_path(directory / "manifest.json")
    manifest, manifest_bytes = _json(manifest_path, "episode manifest")
    if manifest.get("schema") != EPISODE_SCHEMA:
        raise ValueError("unsupported episode manifest schema")
    split_spec = manifest.get("split_spec")
    split = build_split_from_spec(seed=split_seed, spec=split_spec)
    if manifest.get("split_sha256") != split["split_sha256"] or manifest.get("attributes") != split["attributes"]:
        raise ValueError("episode manifest does not match its declared training split")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("episode manifest has no role-file table")

    # Deliberately request only the `train` entries below. In particular, do
    # not resolve/open/verify the validation or test paths in this function.
    train: dict[str, list[dict[str, Any]]] = {}
    for role in ("sender", "receiver", "gold"):
        role_table = files.get(role)
        entry = role_table.get("train") if isinstance(role_table, dict) else None
        if not isinstance(entry, dict) or entry.get("file") != f"{role}_train.jsonl":
            raise ValueError(f"episode manifest has an invalid {role} training entry")
        payload = _project_path(directory / entry["file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
            raise ValueError(f"{role} training ledger hash mismatch")
        rows = _jsonl_bytes(payload, f"{role} training ledger")
        if len(rows) != entry.get("records") or len(rows) != manifest.get("episodes_per_stage", {}).get("train"):
            raise ValueError(f"{role} training ledger record count mismatch")
        train[role] = rows

    if not train["gold"] or not (len(train["sender"]) == len(train["receiver"]) == len(train["gold"])):
        raise ValueError("training role ledgers must be non-empty and aligned")
    train_ids = set(split["train_meaning_ids"])
    meaning_by_values = {tuple(row["values"]): row["meaning_id"] for row in split["meanings"]}
    for sender, receiver, gold in zip(train["sender"], train["receiver"], train["gold"]):
        if not isinstance(gold.get("episode_id"), str) or not isinstance(gold.get("candidate_set_id"), str):
            raise ValueError("training gold rows are missing episode or set identity")
        if not isinstance(sender.get("private_meaning"), dict) or not isinstance(receiver.get("candidates"), list):
            raise ValueError("training role rows have an invalid shape")
        target = sender["private_meaning"]
        if set(target) != set(split["attributes"]):
            raise ValueError("training sender meaning does not match the declared schema")
        target_values = tuple(target[name] for name in split["attributes"])
        if meaning_by_values.get(target_values) not in train_ids:
            raise ValueError("training sender meaning is outside the training partition")
        candidates = receiver["candidates"]
        candidate_ids = [candidate.get("candidate_id") for candidate in candidates if isinstance(candidate, dict)]
        if len(candidate_ids) != len(candidates) or len(candidate_ids) < 2 or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("training candidate table has invalid or duplicate IDs")
        if gold.get("candidate_id") not in candidate_ids:
            raise ValueError("training target ID is absent from its candidate table")
        matching = next(candidate for candidate in candidates if candidate["candidate_id"] == gold["candidate_id"])
        if matching.get("attributes") != target:
            raise ValueError("training gold candidate does not match the sender meaning")
        for candidate in candidates:
            attributes = candidate.get("attributes")
            if not isinstance(attributes, dict) or set(attributes) != set(split["attributes"]):
                raise ValueError("training candidate tuple does not match the declared schema")
            values = tuple(attributes[name] for name in split["attributes"])
            if meaning_by_values.get(values) not in train_ids:
                raise ValueError("training candidate is outside the training partition")
    verify_ledgers({
        "sender": {"train": train["sender"]},
        "receiver": {"train": train["receiver"]},
        "gold": {"train": train["gold"]},
        "manifest": {"attributes": split["attributes"]},
    })

    return {**train, "manifest": manifest}, split, hashlib.sha256(manifest_bytes).hexdigest()


def build_feedback(
    *, episode_dir: Path, card_path: Path, induction_manifest_path: Path, split_seed: int,
    run_path: Path | None = None, run_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Validate paired training-only batches for one card and summarize exact errors."""
    if run_paths is not None and run_path is not None:
        raise ValueError("supply run_path or run_paths, not both")
    paths = list(run_paths) if run_paths is not None else ([run_path] if run_path is not None else [])
    if not paths or any(path is None for path in paths):
        raise ValueError("at least one source training run is required")
    bundle, split, bundle_manifest_sha256 = _load_train_bundle(episode_dir, split_seed=split_seed)
    card, card_bytes = _json(card_path, "protocol card")
    required_card_fields = {"schema", "protocol_id", "sender_instruction", "receiver_instruction"}
    if set(card) != required_card_fields or card.get("schema") != "tlu.shared_protocol_card.v1":
        raise ValueError("protocol card schema or fields are invalid")
    card_sha256 = hashlib.sha256(card_bytes).hexdigest()
    induction_manifest, induction_bytes = _json(induction_manifest_path, "induction manifest")
    if (
        induction_manifest.get("schema") != "tlu.emergent-ood-induced-card-set.v1"
        or induction_manifest.get("split_seed") != split_seed
        or induction_manifest.get("split_sha256") != split["split_sha256"]
        or induction_manifest.get("protocol_family") != "plain_english"
    ):
        raise ValueError("induction manifest is not for this plain-English training split")
    produced_cards = induction_manifest.get("candidate_cards")
    declared_candidate_count = induction_manifest.get("candidate_count")
    if (
        isinstance(declared_candidate_count, bool)
        or not isinstance(declared_candidate_count, int)
        or not 2 <= declared_candidate_count <= 8
        or not isinstance(produced_cards, list)
        or len(produced_cards) != declared_candidate_count
    ):
        raise ValueError("induction manifest does not contain its complete declared candidate set")
    produced_ids = [item.get("protocol_id") for item in produced_cards if isinstance(item, dict)]
    produced_hashes = [item.get("sha256") for item in produced_cards if isinstance(item, dict)]
    if len(produced_ids) != len(produced_cards) or len(set(produced_ids)) != len(produced_ids) or len(set(produced_hashes)) != len(produced_hashes):
        raise ValueError("induction manifest candidate IDs and hashes must be unique")
    if not any(
        item.get("protocol_id") == card.get("protocol_id") and item.get("sha256") == card_sha256
        for item in produced_cards if isinstance(item, dict)
    ):
        raise ValueError("the supplied card is not hash-bound to the induction manifest")

    expected_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "train",
        "conditions": ["shared_protocol_card"],
        "split_seed": split_seed,
        "split_sha256": split["split_sha256"],
        "task_id": split_task_id(split),
        "input_episode_manifest_sha256": bundle_manifest_sha256,
        "protocol_card_sha256": card_sha256,
        "task_key_id": bundle["manifest"].get("task_key_id"),
        "task_seed": bundle["manifest"].get("task_seed"),
        "candidate_count": bundle["manifest"].get("k"),
    }
    ordered_sets = list(dict.fromkeys(row["candidate_set_id"] for row in bundle["gold"]))
    run_manifest: dict[str, Any] | None = None
    source_runs: list[dict[str, Any]] = []
    selected: set[str] = set()
    all_results: dict[str, dict[str, Any]] = {}
    occupied_offsets: set[int] = set()
    run_settings = (
        "model_population_id", "sender_model", "receiver_model", "sender_tokenizer_id",
        "receiver_tokenizer_id", "communication_budget_bytes",
    )
    for source_path in paths:
        result_path = _project_path(source_path)
        result_bytes = result_path.read_bytes()
        results = _jsonl_bytes(result_bytes, "run result")
        manifest_path = _project_path(result_path.with_suffix(result_path.suffix + ".manifest.json"))
        current_manifest, current_manifest_bytes = _json(manifest_path, "run manifest")
        if (
            current_manifest.get("schema") != RUN_SCHEMA
            or current_manifest.get("results_file") != result_path.name
            or current_manifest.get("results_sha256") != hashlib.sha256(result_bytes).hexdigest()
            or current_manifest.get("result_records") != len(results)
        ):
            raise ValueError("run result manifest or content hash is invalid")
        for key, expected in expected_manifest.items():
            if current_manifest.get(key) != expected:
                raise ValueError(f"run manifest {key} does not match the training bundle/card")
        for key in ("model_population_id", "sender_model", "receiver_model", "sender_tokenizer_id", "receiver_tokenizer_id"):
            if not isinstance(current_manifest.get(key), str) or not current_manifest[key].strip():
                raise ValueError(f"run manifest {key} is missing")
        wire_budget = current_manifest.get("communication_budget_bytes")
        if isinstance(wire_budget, bool) or not isinstance(wire_budget, int) or wire_budget < 0:
            raise ValueError("run manifest has an invalid communication budget")
        if run_manifest is None:
            run_manifest = current_manifest
        elif any(current_manifest.get(key) != run_manifest.get(key) for key in run_settings):
            raise ValueError("source training batches use different model populations or communication budgets")

        set_offset = current_manifest.get("candidate_set_offset")
        set_count = current_manifest.get("candidate_sets")
        if (
            isinstance(set_offset, bool) or not isinstance(set_offset, int) or set_offset < 0
            or isinstance(set_count, bool) or not isinstance(set_count, int) or set_count < 1
        ):
            raise ValueError("run manifest has an invalid candidate-set range")
        offsets = set(range(set_offset, set_offset + set_count))
        if offsets & occupied_offsets or set_offset + set_count > len(ordered_sets):
            raise ValueError("training batches overlap or exceed the available train candidate sets")
        occupied_offsets.update(offsets)
        batch_selected = set(ordered_sets[set_offset:set_offset + set_count])
        if len(batch_selected) != set_count:
            raise ValueError("run candidate-set range exceeds the training partition")
        selected.update(batch_selected)
        batch_expected = {
            gold["episode_id"]: (sender, receiver, gold)
            for sender, receiver, gold in zip(bundle["sender"], bundle["receiver"], bundle["gold"])
            if gold["candidate_set_id"] in batch_selected
        }
        if len(batch_expected) * CALLS_PER_EPISODE["shared_protocol_card"] > MAX_MODEL_CALLS_PER_BATCH:
            raise ValueError("source run candidate-set batch exceeds the frozen 12-call limit")
        batch_indexed = {row.get("episode_id"): row for row in results}
        if len(batch_indexed) != len(results) or set(batch_indexed) != set(batch_expected):
            raise ValueError("run result episode coverage does not match its declared train candidate sets")
        if set(all_results) & set(batch_indexed):
            raise ValueError("training runs contain duplicate episode results")
        all_results.update(batch_indexed)
        source_runs.append({
            "results": result_path.relative_to(PROJECT_ROOT).as_posix(),
            "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "run_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "run_manifest_sha256": hashlib.sha256(current_manifest_bytes).hexdigest(),
            "candidate_set_offset": set_offset,
            "candidate_sets": set_count,
        })

    if run_manifest is None:
        raise ValueError("no valid training run was supplied")
    source_runs.sort(key=lambda row: row["candidate_set_offset"])
    expected_rows = {
        gold["episode_id"]: (sender, receiver, gold)
        for sender, receiver, gold in zip(bundle["sender"], bundle["receiver"], bundle["gold"])
        if gold["candidate_set_id"] in selected
    }
    indexed = all_results
    if set(indexed) != set(expected_rows):
        raise ValueError("combined training batches do not cover the exact selected episodes")

    failures: list[dict[str, Any]] = []
    successes = 0
    invalid = 0
    application_wire_bytes = 0
    input_tokens = 0
    output_tokens = 0
    input_tokens_complete = True
    output_tokens_complete = True
    service_seconds = 0.0
    service_seconds_complete = True
    wall_seconds = 0.0
    model_calls = 0
    for episode_id, (sender, receiver, gold) in expected_rows.items():
        result = indexed[episode_id]
        outcome = result.get("outcome", {})
        trace = result.get("trace", {})
        costs = result.get("costs", {})
        candidates = receiver["candidates"]
        if (
            result.get("schema") != RUN_RESULT_SCHEMA
            or result.get("experiment_id") != EXPERIMENT_ID
            or result.get("stage") != "train"
            or result.get("condition") != "shared_protocol_card"
            or result.get("protocol_id") != card["protocol_id"]
            or result.get("candidate_set_id") != gold["candidate_set_id"]
            or result.get("meaning_id") != gold["meaning_id"]
            or result.get("split_seed") != split_seed
            or result.get("task_seed") != bundle["manifest"].get("task_seed")
            or result.get("stratum", {}).get("scorer_id") != SCORER_ID
            or result.get("stratum", {}).get("task_id") != split_task_id(split)
            or result.get("stratum", {}).get("model_population_id") != run_manifest["model_population_id"]
            or result.get("stratum", {}).get("agent_models", {}).get("sender") != run_manifest["sender_model"]
            or result.get("stratum", {}).get("agent_models", {}).get("receiver") != run_manifest["receiver_model"]
            or result.get("protocol", {}).get("decoder_id") != SCORER_ID
            or trace.get("target_tuple_for_evaluator") != sender["private_meaning"]
            or trace.get("candidate_ids_in_receiver_order") != [c["candidate_id"] for c in candidates]
            or outcome.get("target_candidate_id") != gold["candidate_id"]
        ):
            raise ValueError("a run row does not match its training-only source episode")
        predicted_id = outcome.get("answer_candidate_id")
        predicted = next((c for c in candidates if c["candidate_id"] == predicted_id), None)
        valid = predicted is not None and outcome.get("answer_format_valid") is True
        success = valid and predicted_id == gold["candidate_id"]
        if (
            outcome.get("exact_selection") is not success
            or outcome.get("joint_success") is not success
            or outcome.get("answer_score") != (1.0 if success else 0.0)
        ):
            raise ValueError("training result's exact task-score fields disagree with the reconstructed outcome")
        successes += int(success)
        invalid += int(not valid)
        wire_bytes = costs.get("application_wire_bytes")
        if isinstance(wire_bytes, bool) or not isinstance(wire_bytes, int) or wire_bytes < 0:
            raise ValueError("training result has invalid complete application wire-byte cost")
        application_wire_bytes += wire_bytes
        call_count = costs.get("model_call_count")
        if isinstance(call_count, bool) or not isinstance(call_count, int) or call_count != CALLS_PER_EPISODE["shared_protocol_card"]:
            raise ValueError("training result does not report the expected two model calls per shared-card episode")
        model_calls += call_count
        for field, total_name in (
            ("complete_input_tokens", "input"), ("complete_output_tokens", "output"),
        ):
            value = costs.get(field)
            if value is None:
                if total_name == "input":
                    input_tokens_complete = False
                else:
                    output_tokens_complete = False
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"training result has invalid {field}")
            elif total_name == "input":
                input_tokens += value
            else:
                output_tokens += value
        reported_service = costs.get("complete_reported_service_seconds")
        if reported_service is None:
            service_seconds_complete = False
        elif isinstance(reported_service, bool) or not isinstance(reported_service, (int, float)) or reported_service < 0:
            raise ValueError("training result has invalid service-time cost")
        else:
            service_seconds += reported_service
        row_wall_seconds = result.get("runtime", {}).get("wall_seconds")
        if isinstance(row_wall_seconds, bool) or not isinstance(row_wall_seconds, (int, float)) or row_wall_seconds < 0:
            raise ValueError("training result has invalid wall-time cost")
        wall_seconds += row_wall_seconds
        if not success:
            failure = {
                "private_meaning": sender["private_meaning"],
                "message": (trace.get("message") or "")[:MAX_TEXT_BYTES],
                "candidates": [candidate["attributes"] for candidate in candidates],
                "target_meaning": sender["private_meaning"],
                "predicted_meaning": predicted["attributes"] if predicted else None,
                "answer_format_valid": outcome.get("answer_format_valid") is True,
                "failure_type": "invalid_or_missing_candidate_id" if not valid else "wrong_candidate",
            }
            failures.append(failure)

    total = len(expected_rows)
    ordered_episode_ids = [gold["episode_id"] for gold in bundle["gold"] if gold["candidate_set_id"] in selected]
    induction_manifest = json.loads(_project_path(induction_manifest_path).read_bytes())
    return {
        "schema": FEEDBACK_SCHEMA,
        "split": "train",
        "split_seed": split_seed,
        "split_sha256": split["split_sha256"],
        "input_episode_dir": _project_path(episode_dir).relative_to(PROJECT_ROOT).as_posix(),
        "input_episode_manifest_sha256": bundle_manifest_sha256,
        "training_meaning_ids_sha256": hashlib.sha256(
            "\n".join(sorted(split["train_meaning_ids"])).encode("ascii")
        ).hexdigest(),
        "task_key_id": bundle["manifest"].get("task_key_id"),
        "task_seed": bundle["manifest"].get("task_seed"),
        "model_population_id": run_manifest.get("model_population_id"),
        "sender_model": run_manifest.get("sender_model"),
        "receiver_model": run_manifest.get("receiver_model"),
        "sender_tokenizer_id": run_manifest.get("sender_tokenizer_id"),
        "receiver_tokenizer_id": run_manifest.get("receiver_tokenizer_id"),
        "communication_budget_bytes": run_manifest.get("communication_budget_bytes"),
        "protocol_family": "plain_english",
        "protocol_id": card["protocol_id"],
        "optimization_round": induction_manifest.get("optimization_round", 1),
        "max_optimization_rounds": induction_manifest.get("max_optimization_rounds", 2),
        "candidate_count": induction_manifest.get("candidate_count"),
        "parent_feedback_sha256": induction_manifest.get("feedback_sha256"),
        "source_induction_manifest_sha256": hashlib.sha256(induction_bytes).hexdigest(),
        "source_induction_manifest": _project_path(induction_manifest_path).relative_to(PROJECT_ROOT).as_posix(),
        "source_induction_model_calls": induction_manifest.get("model_calls"),
        "source_induction_input_tokens": induction_manifest.get("input_tokens"),
        "source_induction_output_tokens": induction_manifest.get("output_tokens"),
        "source_induction_prompt_completion_utf8_bytes": (
            induction_manifest.get("prompt_utf8_bytes", 0) + induction_manifest.get("completion_utf8_bytes", 0)
        ),
        "protocol_card_sha256": card_sha256,
        "source_card": _project_path(card_path).relative_to(PROJECT_ROOT).as_posix(),
        "sender_instruction": card["sender_instruction"],
        "receiver_instruction": card["receiver_instruction"],
        "source_runs": source_runs,
        "candidate_sets": len(selected),
        "episodes": total,
        "training_episode_ids_sha256": hashlib.sha256("\n".join(ordered_episode_ids).encode("ascii")).hexdigest(),
        "exact_successes": successes,
        "exact_success_rate": successes / total,
        "invalid_answers": invalid,
        "application_wire_bytes": application_wire_bytes,
        "model_calls": model_calls,
        "complete_input_tokens": input_tokens if input_tokens_complete else None,
        "complete_output_tokens": output_tokens if output_tokens_complete else None,
        "complete_service_seconds": service_seconds if service_seconds_complete else None,
        "wall_seconds": wall_seconds,
        "failure_count": total - successes,
        "failure_examples_truncated": total - successes > len(failures[:MAX_FAILURES]),
        "failure_examples": failures[:MAX_FAILURES],
        "validation_files_opened": False,
        "test_files_opened": False,
        "contains_evaluator_labels": True,
        "limits": [
            "Feedback is training-only and may guide proposals, but it is not confirmatory evidence.",
            "Only the first 24 failure rows in source episode order are included.",
            "Training gold labels are included to support exact error diagnosis; never expose this artifact to evaluated agents.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True, help="training run; repeat for 12-call batches")
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--induction-manifest", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _project_path(args.output)
        if output.exists():
            raise ValueError(f"refusing to overwrite existing feedback artifact: {output}")
        feedback = build_feedback(
            episode_dir=args.episode_dir, run_paths=args.run, card_path=args.card,
            induction_manifest_path=args.induction_manifest,
            split_seed=args.split_seed,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(feedback, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "feedback": output.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "episodes": feedback["episodes"],
        "exact_success_rate": feedback["exact_success_rate"],
        "failure_examples": len(feedback["failure_examples"]),
        "validation_files_opened": False,
        "test_files_opened": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
