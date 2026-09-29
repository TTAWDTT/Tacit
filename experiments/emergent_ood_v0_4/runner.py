"""Leakage-aware model runner for Emergent OOD v0.4 episodes.

Dry-run is the default. Execution requires an explicit flag, a fresh passing
resource preflight, loopback endpoints, and an existing local episode bundle.
The runner never starts a model server or downloads a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacit.runtime import ChatCompletion, ChatModel, DialogueResult, OpenAICompatibleClient, exchange_dialogue

from experiments.emergent_ood_v0_3.runner import validate_resource_preflight
from experiments.emergent_ood_v0_4.episodes import SCHEMA as EPISODE_SCHEMA
from experiments.emergent_ood_v0_4.episodes import verify_ledgers
from experiments.emergent_ood_v0_4.split import build_split


EXPERIMENT_ID = "emergent-ood-v0.4-receiver-utility"
SCORER_ID = "strict-candidate-id-and-canonical-sender-fidelity-v1"
CONDITIONS = ("full_information", "no_message", "natural_language", "json", "symbolic", "shared_protocol_card")
CALLS_PER_EPISODE = {"full_information": 1, "no_message": 1, "natural_language": 2, "json": 2, "symbolic": 2, "shared_protocol_card": 2}
MAX_MODEL_CALLS_PER_BATCH = 12
MAX_CANDIDATE_SETS_PER_BATCH = 3
MAX_EXACT_COLORING_VERTICES = 20
MAX_COLORING_SEARCH_NODES_PER_COLOR_COUNT = 10000
REQUEST_TIMEOUT_SECONDS = 45.0
DEFAULT_WIRE_BUDGET_BYTES = 4096
STAGES = ("train", "validation", "test")


class _Protocol:
    def __init__(
        self, condition: str, attributes: Sequence[str], values: Mapping[str, Sequence[str]],
        protocol_card: dict[str, str] | None = None,
    ) -> None:
        if condition == "shared_protocol_card":
            if protocol_card is None:
                raise ValueError("shared_protocol_card requires a validated protocol card")
            self.protocol_id = protocol_card["protocol_id"]
            role_boundary = (
                "Use only the information in your private context and the fixed one-message schedule. "
                "Do not claim access to the other agent's private context or to evaluator data."
            )
            sender_instruction = (
                "You are the sender. You have one private meaning and one message. " + role_boundary
                + "\nShared protocol card (follow exactly):\n" + protocol_card["sender_instruction"]
                + "\nReturn only the message payload."
            )
            receiver_instruction = (
                "You are the receiver. The one received message and your candidate table are your only task information. "
                + role_boundary + "\nShared protocol card (follow exactly):\n"
                + protocol_card["receiver_instruction"]
                + "\nSelect the exact matching candidate and return only its candidate_id, with no explanation."
            )
            self.agent_instructions = {"sender": sender_instruction, "receiver": receiver_instruction}
            return
        self.protocol_id = f"tlu.emergent-ood.v0.4.{condition}.v1"
        self.agent_instructions = {
            "sender": self._sender_instruction(condition, attributes, values),
            "receiver": self._receiver_instruction(condition, attributes, values),
        }

    @staticmethod
    def _sender_instruction(condition: str, attributes: Sequence[str], values: Mapping[str, Sequence[str]]) -> str:
        common = (
            "You are the sender in a one-message exact-match task. Use only the private meaning in your input. "
            "Do not guess the receiver's candidate table or add facts. Return only the message payload, with no explanation."
        )
        if condition in {"full_information", "no_message"}:
            return common + " No sender message is required for this control."
        if condition == "natural_language":
            return common + " Describe all four attribute names and their exact values in one short sentence."
        if condition == "json":
            return common + " Return one compact JSON object with exactly the four attribute names as keys and their exact string values."
        if condition == "symbolic":
            codebook = {name: {value: str(index) for index, value in enumerate(values[name])} for name in attributes}
            return common + (
                " Encode one digit per attribute in this fixed order: " + ", ".join(attributes) + ". "
                "For each attribute, its allowed values map to digits in the listed order: "
                + json.dumps(codebook, ensure_ascii=False, separators=(",", ":"))
                + ". Emit exactly four digits and no separators."
            )
        raise ValueError(f"condition has no sender: {condition}")

    @staticmethod
    def _receiver_instruction(condition: str, attributes: Sequence[str], values: Mapping[str, Sequence[str]]) -> str:
        common = (
            "You are the receiver. The other agent's message is the only information it sent. "
            "Choose the candidate whose complete meaning matches the message. Return only that candidate_id exactly. "
            "If uncertain, still choose one candidate_id. Do not include punctuation, JSON, or explanation."
        )
        if condition == "full_information":
            return (
                "You are the receiver in a full-information capability control. "
                "Use the calibration_target tuple in your private context and select the candidate with the exact same four values. "
                "Return only its candidate_id, with no punctuation or explanation."
            )
        if condition == "no_message":
            return (
                "You are the receiver in a no-message control. No sender message was sent. "
                "Choose one candidate from the private candidate table and return only its candidate_id, with no explanation."
            )
        if condition == "natural_language":
            return common + " Interpret ordinary English descriptions of the named attributes."
        if condition == "json":
            return common + " Interpret the sender's JSON object as attribute names and exact values."
        if condition == "symbolic":
            codebook = {name: {str(index): value for index, value in enumerate(values[name])} for name in attributes}
            return common + (
                " Decode exactly four digits in this attribute order: " + ", ".join(attributes) + ". "
                "The per-attribute digit maps are: " + json.dumps(codebook, ensure_ascii=False, separators=(",", ":")) + "."
            )
        raise ValueError(f"condition has no receiver protocol: {condition}")


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("all episode and output paths must stay inside the project") from exc
    return resolved


def load_protocol_card(path: Path) -> tuple[dict[str, str], str]:
    path = _inside_project(path)
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("protocol card is missing or invalid JSON") from exc
    required = {"schema", "protocol_id", "sender_instruction", "receiver_instruction"}
    if not isinstance(card, dict) or set(card) != required or card.get("schema") != "tlu.shared_protocol_card.v1":
        raise ValueError("protocol card fields or schema are invalid")
    for field in ("protocol_id", "sender_instruction", "receiver_instruction"):
        if not isinstance(card[field], str) or not card[field].strip():
            raise ValueError(f"protocol card {field} must be a non-empty string")
    if len(card["protocol_id"]) > 128:
        raise ValueError("protocol card ID is too long")
    if any(len(card[field].encode("utf-8")) > 32768 for field in ("sender_instruction", "receiver_instruction")):
        raise ValueError("protocol card instructions exceed 32 KiB per role")
    return {field: card[field] for field in required if field != "schema"}, hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read valid JSONL: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def load_episode_bundle(directory: Path, *, split_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load all role files, verify hashes/split/labels, and return bundle + split."""
    directory = _inside_project(directory)
    try:
        manifest_path = _inside_project(directory / "manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("episode manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != EPISODE_SCHEMA:
        raise ValueError("unsupported episode manifest schema")
    split = build_split(seed=split_seed)
    if manifest.get("split_sha256") != split["split_sha256"]:
        raise ValueError("episode manifest does not match the declared split seed")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("episode manifest file table is missing")
    bundle: dict[str, Any] = {"manifest": {key: value for key, value in manifest.items() if key != "files"}}
    for role in ("sender", "receiver", "gold"):
        role_files = files.get(role)
        if not isinstance(role_files, dict) or set(role_files) != set(STAGES):
            raise ValueError(f"manifest does not contain all {role} stages")
        bundle[role] = {}
        for stage in STAGES:
            entry = role_files[stage]
            if not isinstance(entry, dict) or entry.get("file") != f"{role}_{stage}.jsonl":
                raise ValueError("manifest contains an unexpected role filename")
            file_path = _inside_project(directory / entry["file"])
            payload = file_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
                raise ValueError(f"{role} {stage} ledger hash mismatch")
            rows = _jsonl(file_path)
            if len(rows) != entry.get("records"):
                raise ValueError(f"{role} {stage} ledger record count mismatch")
            bundle[role][stage] = rows
    verify_ledgers(bundle)
    train_ids = set(split["train_meaning_ids"])
    heldout_ids = set(split["held_out_meaning_ids"])
    meaning_ids_by_values = {tuple(row["values"]): row["meaning_id"] for row in split["meanings"]}
    for stage in STAGES:
        allowed = train_ids if stage == "train" else heldout_ids
        if any(row["meaning_id"] not in allowed for row in bundle["gold"][stage]):
            raise ValueError(f"{stage} ledger includes a target from the wrong split")
        for sender_row, receiver_row in zip(bundle["sender"][stage], bundle["receiver"][stage]):
            sender_tuple = tuple(sender_row["private_meaning"].get(attribute) for attribute in split["attributes"])
            if meaning_ids_by_values.get(sender_tuple) not in allowed:
                raise ValueError(f"{stage} sender input is not in its declared meaning partition")
            for candidate in receiver_row["candidates"]:
                if set(candidate.get("attributes", {})) != set(split["attributes"]):
                    raise ValueError("candidate tuple does not match the declared schema")
                candidate_tuple = tuple(candidate["attributes"].get(attribute) for attribute in split["attributes"])
                if meaning_ids_by_values.get(candidate_tuple) not in allowed:
                    raise ValueError(f"{stage} receiver candidate is not in its declared meaning partition")
    for role in ("sender", "receiver", "gold"):
        for stage in STAGES:
            declared = manifest["files"][role][stage]["records"]
            if manifest.get("episodes_per_stage", {}).get(stage) != declared:
                raise ValueError("manifest stage counts do not match role ledger records")
    if manifest.get("partition_sizes") != {"train": 192, "validation": 16, "test": 48}:
        raise ValueError("manifest has unexpected default partition sizes")
    return bundle, split


def select_candidate_sets(bundle: dict[str, Any], stage: str, set_count: int) -> list[dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError("stage must be train, validation, or test")
    if isinstance(set_count, bool) or not isinstance(set_count, int) or not 1 <= set_count <= MAX_CANDIDATE_SETS_PER_BATCH:
        raise ValueError(f"set_count must be in 1..{MAX_CANDIDATE_SETS_PER_BATCH} per batch")
    sender, receiver, gold = (bundle[role][stage] for role in ("sender", "receiver", "gold"))
    if not (len(sender) == len(receiver) == len(gold)):
        raise ValueError("role ledgers are not aligned")
    picked = []
    selected_set_ids: list[str] = []
    for index, gold_row in enumerate(gold):
        set_id = gold_row["candidate_set_id"]
        if set_id not in selected_set_ids:
            if len(selected_set_ids) >= set_count:
                continue
            selected_set_ids.append(set_id)
        if set_id in selected_set_ids:
            picked.append({"sender": sender[index], "receiver": receiver[index], "gold": gold_row})
    expected_k = bundle["manifest"]["k"]
    if len(picked) != set_count * expected_k:
        raise ValueError("selected candidate sets are incomplete or the stage has too few sets")
    return picked


def _meaning_id_from_values(values: Sequence[str]) -> str:
    canonical = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return "m-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def conflict_graph_summary(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Report co-occurrence coverage and exact chromatic number for small batches."""
    vertices: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    candidate_sets: list[set[str]] = []
    for episode in episodes:
        candidate_meanings = {
            _meaning_id_from_values(tuple(row["attributes"][axis] for axis in sorted(row["attributes"])))
            for row in episode["receiver"]["candidates"]
        }
        candidate_sets.append(candidate_meanings)
        vertices.update(candidate_meanings)
        for vertex in candidate_meanings:
            adjacency.setdefault(vertex, set()).update(candidate_meanings - {vertex})
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    possible_pairs = len(vertices) * (len(vertices) - 1) // 2
    lower_bound = max((len(candidate_set) for candidate_set in candidate_sets), default=0)
    if not vertices:
        return {"observed_target_vertices": 0, "cooccurring_target_pairs": 0, "possible_pairs": 0,
                "pair_coverage": 1.0, "chromatic_number": 0, "chromatic_number_exact": True}

    order = sorted(vertices, key=lambda vertex: (-len(adjacency[vertex]), vertex))
    greedy_colors: dict[str, int] = {}
    for vertex in order:
        unavailable = {greedy_colors[neighbor] for neighbor in adjacency[vertex] if neighbor in greedy_colors}
        color = 0
        while color in unavailable:
            color += 1
        greedy_colors[vertex] = color
    upper_bound = max(greedy_colors.values()) + 1
    chromatic = lower_bound if lower_bound == upper_bound else None
    exact = chromatic is not None or len(vertices) <= MAX_EXACT_COLORING_VERTICES

    def is_colorable(color_count: int) -> bool:
        colors: dict[str, int] = {}
        visited = 0

        def visit() -> bool | None:
            nonlocal visited
            visited += 1
            if visited > MAX_COLORING_SEARCH_NODES_PER_COLOR_COUNT:
                return None
            if len(colors) == len(vertices):
                return True
            uncolored = [vertex for vertex in vertices if vertex not in colors]
            vertex = max(
                uncolored,
                key=lambda item: (
                    len({colors[n] for n in adjacency[item] if n in colors}),
                    len(adjacency[item]),
                    item,
                ),
            )
            forbidden = {colors[n] for n in adjacency[vertex] if n in colors}
            # Color-name symmetry: introduce at most the next unused label.
            next_unused = max(colors.values(), default=-1) + 1
            for color in range(min(color_count, next_unused + 1)):
                if color in forbidden:
                    continue
                colors[vertex] = color
                result = visit()
                if result is True or result is None:
                    return result
                del colors[vertex]
            return False

        return visit()

    search_exhausted = False
    if exact and chromatic is None:
        for candidate in range(lower_bound, upper_bound + 1):
            result = is_colorable(candidate)
            if result is None:
                exact = False
                search_exhausted = True
                break
            if result:
                chromatic = candidate
                break
    return {
        "observed_target_vertices": len(vertices),
        "cooccurring_target_pairs": edge_count,
        "possible_pairs": possible_pairs,
        "pair_coverage": edge_count / possible_pairs if possible_pairs else 1.0,
        "minimum_clique_lower_bound": lower_bound,
        "greedy_coloring_upper_bound": upper_bound,
        "chromatic_number": chromatic,
        "chromatic_number_exact": exact and chromatic is not None,
        "chromatic_search_budget_exhausted": search_exhausted,
        "fixed_width_zero_error_payload_floor_bits": (
            (chromatic - 1).bit_length() if chromatic is not None else None
        ),
        "complete_pair_graph_on_observed_vertices": edge_count == possible_pairs,
    }


def _parse_choice(answer_text: str, candidate_ids: Sequence[str]) -> tuple[str | None, bool]:
    value = answer_text.strip()
    if value in candidate_ids:
        return value, True
    return None, False


def _canonical_label_fidelity(text: str, target: Mapping[str, str], values: Mapping[str, Sequence[str]]) -> bool | None:
    """Conservative exact-label audit; this is not a general semantic judge."""
    lowered = text.casefold()
    for axis, expected in target.items():
        present = []
        for value in values[axis]:
            if re.search(rf"(?<![a-z]){re.escape(value.casefold())}(?![a-z])", lowered):
                present.append(value)
        if present != [expected]:
            return False
    return True


def _strict_json_tuple(message: str, attributes: Sequence[str], target: Mapping[str, str]) -> tuple[bool, bool]:
    duplicate = False

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        parsed = json.loads(message, object_pairs_hook=object_pairs)
    except (json.JSONDecodeError, TypeError):
        return False, False
    semantic = not duplicate and isinstance(parsed, dict) and set(parsed) == set(attributes) and parsed == dict(target)
    canonical = {attribute: target[attribute] for attribute in attributes}
    exact_format = semantic and message == json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return semantic, exact_format


def _sender_audit(condition: str, message: str | None, target: Mapping[str, str], attributes: Sequence[str], values: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    if message is None:
        return {"canonical_label_fidelity": None, "semantic_parse_valid": None, "exact_format_valid": None}
    label_fidelity = _canonical_label_fidelity(message, target, values) if condition == "natural_language" else None
    semantic_valid = exact_format_valid = None
    if condition == "json":
        semantic_valid, exact_format_valid = _strict_json_tuple(message, attributes, target)
    elif condition == "symbolic":
        encoded = "".join(str(list(values[axis]).index(target[axis])) for axis in attributes)
        semantic_valid = message == encoded
        exact_format_valid = semantic_valid
    return {
        "canonical_label_fidelity": label_fidelity,
        "semantic_parse_valid": semantic_valid,
        "exact_format_valid": exact_format_valid,
    }


def run_condition(
    *, episode: dict[str, Any], condition: str, stage: str,
    sender_model: ChatModel | None, receiver_model: ChatModel,
    sender_tokenizer_id: str | None = None, receiver_tokenizer_id: str = "not_reported",
    attributes: Sequence[str], values: Mapping[str, Sequence[str]],
    protocol_card: dict[str, str] | None = None,
    split_seed: int, task_seed: int, model_population_id: str,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    target = episode["sender"]["private_meaning"]
    candidates = episode["receiver"]["candidates"]
    candidate_ids = [row["candidate_id"] for row in candidates]
    gold = episode["gold"]
    if len(candidate_ids) != len(set(candidate_ids)) or gold["candidate_id"] not in candidate_ids:
        raise ValueError("episode candidate view or evaluator target is malformed")

    receiver_context: dict[str, Any] = {"candidates": candidates}
    if condition == "full_information":
        receiver_context["calibration_target"] = target
        protocol = _Protocol("full_information", attributes, values)
        schedule: tuple[tuple[str, str], ...] = ()
        sender_model = None
    elif condition == "no_message":
        protocol = _Protocol("no_message", attributes, values)
        schedule = ()
        sender_model = None
    else:
        if sender_model is None:
            raise ValueError(f"{condition} requires a sender model")
        protocol = _Protocol(condition, attributes, values, protocol_card=protocol_card)
        schedule = (("sender", "receiver"),)

    # The receiver gets the private tuple only in the explicit full-information control.
    private_contexts = {
        "sender": json.dumps({"private_meaning": target}, ensure_ascii=False, separators=(",", ":")),
        "receiver": json.dumps(receiver_context, ensure_ascii=False, separators=(",", ":")),
    }
    started = time.perf_counter()
    result: DialogueResult = exchange_dialogue(
        {"sender": sender_model or receiver_model, "receiver": receiver_model},
        protocol=protocol,
        private_contexts=private_contexts,
        schedule=schedule,
        task="Select the candidate ID whose full tuple is the sender's private meaning.",
        max_turns=1,
        wire_budget_bytes=DEFAULT_WIRE_BUDGET_BYTES,
        final_answer_agent="receiver",
        final_answer_instruction="Return only the exact candidate_id of your selected candidate for external scoring.",
    )
    answer_text = result.final_submission.text if result.final_submission else ""
    answer_id, answer_valid = _parse_choice(answer_text, candidate_ids)
    target_row = next(row for row in candidates if row["candidate_id"] == gold["candidate_id"])
    message = result.turns[0].completion.text if result.turns else None
    sender_audit = _sender_audit(condition, message, target, attributes, values)
    tokenizer_ids = {"receiver": receiver_tokenizer_id}
    if result.turns:
        tokenizer_ids["sender"] = sender_tokenizer_id or "not_reported"
    calls = result.model_call_records(tokenizers=tokenizer_ids)
    def complete_total(name: str) -> int | float | None:
        measured = [row[name] for row in calls]
        if any(value is None for value in measured):
            return None
        return sum(measured)

    complete_service_total = complete_total("service_seconds")
    message_delivered = bool(result.turns and result.turns[0].transmission is not None)
    return {
        "schema": "tlu.emergent-ood-run.v0.4",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "split_seed": split_seed,
        "task_seed": task_seed,
        "episode_id": gold["episode_id"],
        "candidate_set_id": gold["candidate_set_id"],
        "meaning_id": gold["meaning_id"],
        "condition": condition,
        "protocol_id": protocol.protocol_id,
        "outcome": {
            "answer_format_valid": answer_valid,
            "answer_candidate_id": answer_id,
            "target_candidate_id": gold["candidate_id"],
            "exact_selection": answer_id == gold["candidate_id"],
            "bayes_no_message_reference": 1 / len(candidate_ids),
        },
        "sender_audit": sender_audit,
        "costs": {
            "model_calls": calls,
            "model_call_count": result.model_calls,
            "complete_input_tokens": complete_total("input_tokens"),
            "complete_output_tokens": complete_total("output_tokens"),
            "calls_with_input_token_usage": sum(row["input_tokens"] is not None for row in calls),
            "calls_with_output_token_usage": sum(row["output_tokens"] is not None for row in calls),
            "generated_message_bytes": len(message.encode("utf-8")) if message is not None else 0,
            "message_delivered": message_delivered,
            "delivered_payload_bytes": len(message.encode("utf-8")) if message is not None and message_delivered else 0,
            "application_wire_bytes": result.wire_bytes,
            "complete_reported_service_seconds": complete_service_total,
            "calls_with_service_time": sum(row["service_seconds"] is not None for row in calls),
            "wall_seconds": time.perf_counter() - started,
            "protocol_card_bytes": {
                "sender_instruction": len(protocol.agent_instructions["sender"].encode("utf-8")),
                "receiver_instruction": len(protocol.agent_instructions["receiver"].encode("utf-8")),
            },
        },
        "trace": {
            "message": message,
            "answer": answer_text,
            "candidate_ids_in_receiver_order": candidate_ids,
            "target_tuple_for_evaluator": target_row["attributes"],
            "transmissions": result.transmission_records(),
            "stop_reason": result.stop_reason,
        },
        "stratum": {
            "model_population_id": model_population_id,
            "candidate_count": len(candidate_ids),
            "inference_cluster_id": f"split={split_seed}",
            "candidate_set_cluster_id": gold["candidate_set_id"],
            "scorer_id": SCORER_ID,
        },
    }


def _loopback_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("only loopback model endpoints are permitted")
    return value.rstrip("/")


def _endpoint_port(value: str) -> int:
    parsed = urlparse(value)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def _write_results(
    rows: Sequence[dict[str, Any]], output: Path, *, manifest: dict[str, Any], force: bool,
) -> str:
    output = _inside_project(output)
    manifest_path = _inside_project(output.with_suffix(output.suffix + ".manifest.json"))
    if not force and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing output or manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    output.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    complete_manifest = {
        "schema": "tlu.emergent-ood-run-manifest.v0.4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **manifest,
        "results_file": output.name,
        "results_sha256": digest,
        "result_records": len(rows),
        "contains_evaluator_only_labels": True,
    }
    manifest_path.write_text(json.dumps(complete_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(".cache/emergent_ood_v0_4/episodes"))
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--stage", choices=STAGES, default="validation")
    parser.add_argument("--sets", type=int, default=1)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["natural_language"])
    parser.add_argument("--protocol-card", type=Path, help="shared sender/receiver card JSON for the shared_protocol_card condition")
    parser.add_argument("--output", type=Path, default=Path(".cache/emergent_ood_v0_4/runs/validation.jsonl"))
    parser.add_argument("--execute", action="store_true", help="contact the configured local loopback model endpoints")
    parser.add_argument("--resource-preflight", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("TLU_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--sender-base-url", default=os.environ.get("TLU_SENDER_BASE_URL", ""))
    parser.add_argument("--receiver-base-url", default=os.environ.get("TLU_RECEIVER_BASE_URL", ""))
    parser.add_argument("--sender-model", default=os.environ.get("TLU_SENDER_MODEL", ""))
    parser.add_argument("--receiver-model", default=os.environ.get("TLU_RECEIVER_MODEL", ""))
    parser.add_argument("--sender-tokenizer-id", default=os.environ.get("TLU_SENDER_TOKENIZER_ID", ""))
    parser.add_argument("--receiver-tokenizer-id", default=os.environ.get("TLU_RECEIVER_TOKENIZER_ID", ""))
    parser.add_argument("--model-population-id", default=os.environ.get("TLU_MODEL_POPULATION_ID", "local-unspecified"))
    args = parser.parse_args()

    try:
        bundle, split = load_episode_bundle(args.input_dir, split_seed=args.split_seed)
        episodes = select_candidate_sets(bundle, args.stage, args.sets)
        protocol_card, protocol_card_digest = (
            load_protocol_card(args.protocol_card) if args.protocol_card else (None, None)
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if "shared_protocol_card" in args.conditions and protocol_card is None:
        parser.error("shared_protocol_card requires --protocol-card")
    if args.protocol_card is not None and "shared_protocol_card" not in args.conditions:
        parser.error("--protocol-card is only valid with shared_protocol_card")
    calls_planned = len(episodes) * sum(CALLS_PER_EPISODE[name] for name in args.conditions)
    if calls_planned > MAX_MODEL_CALLS_PER_BATCH:
        parser.error(f"batch plans {calls_planned} model calls; hard cap is {MAX_MODEL_CALLS_PER_BATCH}")
    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "experiment_id": EXPERIMENT_ID,
            "stage": args.stage,
            "conditions": args.conditions,
            "candidate_sets": args.sets,
            "episodes": len(episodes),
            "candidate_count": bundle["manifest"]["k"],
            "analytic_no_message_accuracy": 1 / bundle["manifest"]["k"],
            "planned_model_calls": calls_planned,
            "maximum_model_calls_per_batch": MAX_MODEL_CALLS_PER_BATCH,
            "input_dir": str(_inside_project(args.input_dir)),
            "output": str(_inside_project(args.output)),
            "model_loaded": False,
            "inference_started": False,
        }, indent=2))
        return 0

    if not args.receiver_model:
        parser.error("--execute requires --receiver-model")
    if not args.receiver_tokenizer_id:
        parser.error("--execute requires --receiver-tokenizer-id")
    needs_sender = any(condition in {"natural_language", "json", "symbolic"} for condition in args.conditions)
    if needs_sender and not args.sender_model:
        parser.error("message conditions require --sender-model")
    if needs_sender and not args.sender_tokenizer_id:
        parser.error("message conditions require --sender-tokenizer-id")
    try:
        receiver_endpoint = _loopback_url(args.receiver_base_url or args.base_url)
        sender_endpoint = _loopback_url(args.sender_base_url or args.base_url) if needs_sender else receiver_endpoint
        preflight_path = _inside_project(args.resource_preflight) if args.resource_preflight else None
        validate_resource_preflight(
            preflight_path,
            required_ports={_endpoint_port(receiver_endpoint), *([_endpoint_port(sender_endpoint)] if needs_sender else [])},
        )
        output_path = _inside_project(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    output_manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if not args.force and (output_path.exists() or output_manifest_path.exists()):
        parser.error(f"refusing to overwrite {output_path}; pass --force explicitly")

    sender_client = OpenAICompatibleClient(
        sender_endpoint, args.sender_model, timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_tokens=160, follow_redirects=False, temperature=0.0,
    ) if needs_sender else None
    receiver_client = OpenAICompatibleClient(
        receiver_endpoint, args.receiver_model, timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_tokens=48, follow_redirects=False, temperature=0.0,
    )
    rows = []
    candidate_graph = conflict_graph_summary(episodes)
    for episode in episodes:
        for condition in args.conditions:
            row = run_condition(
                episode=episode, condition=condition, stage=args.stage,
                sender_model=sender_client, receiver_model=receiver_client,
                sender_tokenizer_id=args.sender_tokenizer_id or None,
                receiver_tokenizer_id=args.receiver_tokenizer_id,
                attributes=split["attributes"], values=split["values_by_attribute"],
                protocol_card=protocol_card if condition == "shared_protocol_card" else None,
                split_seed=args.split_seed, task_seed=bundle["manifest"]["task_seed"],
                model_population_id=args.model_population_id,
            )
            row["stratum"]["candidate_suite_conflict_graph"] = candidate_graph
            rows.append(row)
    preflight_digest = hashlib.sha256(preflight_path.read_bytes()).hexdigest() if preflight_path else None
    output_digest = _write_results(
        rows,
        output_path,
        force=args.force,
        manifest={
            "experiment_id": EXPERIMENT_ID,
            "stage": args.stage,
            "conditions": args.conditions,
            "protocol_card_sha256": protocol_card_digest,
            "candidate_sets": args.sets,
            "candidate_count": bundle["manifest"]["k"],
            "split_seed": args.split_seed,
            "split_sha256": split["split_sha256"],
            "task_seed": bundle["manifest"]["task_seed"],
            "task_key_id": bundle["manifest"]["task_key_id"],
            "model_population_id": args.model_population_id,
            "sender_model": args.sender_model if needs_sender else None,
            "receiver_model": args.receiver_model,
            "sender_tokenizer_id": args.sender_tokenizer_id if needs_sender else None,
            "receiver_tokenizer_id": args.receiver_tokenizer_id,
            "temperature": 0.0,
            "sender_max_tokens": 160 if needs_sender else None,
            "receiver_max_tokens": 48,
            "resource_preflight_sha256": preflight_digest,
            "input_episode_manifest_sha256": hashlib.sha256(
                (_inside_project(args.input_dir) / "manifest.json").read_bytes()
            ).hexdigest(),
        },
    )
    print(json.dumps({
        "mode": "executed", "episodes": len(episodes), "conditions": args.conditions,
        "model_calls": calls_planned, "output": str(output_path), "sha256": output_digest,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
