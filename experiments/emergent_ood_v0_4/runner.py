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
CONDITIONS = ("full_information", "no_message", "natural_language", "autoform", "json", "symbolic", "shared_protocol_card", "usage_only_transfer")
CALLS_PER_EPISODE = {condition: (1 if condition in {"full_information", "no_message"} else 2) for condition in CONDITIONS}
MAX_MODEL_CALLS_PER_BATCH = 12
MAX_CANDIDATE_SETS_PER_BATCH = 3
CAPABILITY_CALIBRATION_SETS = 3
MAX_EXACT_COLORING_VERTICES = 20
MAX_COLORING_SEARCH_NODES_PER_COLOR_COUNT = 10000
REQUEST_TIMEOUT_SECONDS = 45.0
DEFAULT_WIRE_BUDGET_BYTES = 4096
STAGES = ("train", "validation", "test")
CHECKPOINT_SCHEMA = "tlu.emergent-ood-run-checkpoint.v1"


class _Protocol:
    def __init__(
        self, condition: str, attributes: Sequence[str], values: Mapping[str, Sequence[str]],
        protocol_card: dict[str, str] | None = None,
        usage_examples: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if condition in {"shared_protocol_card", "usage_only_transfer"}:
            if protocol_card is None:
                raise ValueError(f"{condition} requires a validated protocol card")
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
            if condition == "usage_only_transfer":
                if not usage_examples:
                    raise ValueError("usage_only_transfer requires non-empty training usage examples")
                receiver_instruction = (
                    "You are a new receiver. Infer the sender's message convention only from the training examples "
                    "in your private context. Do not assume access to the protocol card or any external decoder. "
                    "Apply the inferred mapping to the one received message, match the full tuple against your candidate "
                    "table, and return only the exact candidate_id with no explanation. " + role_boundary
                )
            else:
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
        if condition == "autoform":
            return common + (
                " Choose a concise, efficient communication medium other than ordinary prose, suited to this exact tuple. "
                "You may use structured data, a table, mathematical notation, pseudocode, or a compact code. "
                "Your partner has the attribute vocabulary and candidate table, but not your private tuple; no fixed message grammar "
                "is shared, so make your representation "
                "decodable and preserve all four exact attribute values. Return only the message."
            )
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
        if condition == "autoform":
            return common + (
                " The sender may choose an open concise format, including structured data, tables, equations, pseudocode, "
                "or code; no fixed syntax is guaranteed. Infer the four exact attribute values from the message and match "
                "the complete tuple against your candidate table."
            )
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


def load_usage_examples(
    path: Path, *, split: dict[str, Any], bundle: dict[str, Any],
    protocol_card: dict[str, str], protocol_card_sha256: str,
    training_episode_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Load a protocol-bound, training-only exemplar trace for receiver onboarding."""
    path = _inside_project(path)
    try:
        payload = path.read_bytes()
        artifact = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("usage-example artifact is missing or invalid JSON") from exc
    if len(payload) > 1_048_576:
        raise ValueError("usage-example artifact exceeds 1 MiB")
    required = {
        "schema", "protocol_id", "protocol_card_sha256", "training_split_sha256",
        "training_episode_manifest_sha256", "acquisition", "examples",
    }
    if not isinstance(artifact, dict) or set(artifact) != required or artifact.get("schema") != "tlu.usage_examples.v1":
        raise ValueError("usage-example artifact fields or schema are invalid")
    if artifact["protocol_id"] != protocol_card["protocol_id"]:
        raise ValueError("usage examples name a different protocol card")
    if artifact["protocol_card_sha256"] != protocol_card_sha256:
        raise ValueError("usage examples are not bound to the supplied protocol card")
    if artifact["training_split_sha256"] != split["split_sha256"]:
        raise ValueError("usage examples are not bound to this training split")
    if artifact["training_episode_manifest_sha256"] != training_episode_manifest_sha256:
        raise ValueError("usage examples are not bound to this exact training episode manifest")
    acquisition = artifact["acquisition"]
    if not isinstance(acquisition, dict) or set(acquisition) != {"method", "model_id", "generation_calls", "input_tokens", "output_tokens"}:
        raise ValueError("usage-example acquisition accounting is malformed")
    if acquisition["method"] not in {"model_generated", "human_authored", "programmatic"}:
        raise ValueError("usage-example acquisition method is invalid")
    if not isinstance(acquisition["model_id"], str):
        raise ValueError("usage-example acquisition model_id must be a string")
    for field in ("generation_calls", "input_tokens", "output_tokens"):
        value = acquisition[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"usage-example acquisition {field} must be a non-negative integer or null")
    examples = artifact["examples"]
    if not isinstance(examples, list) or not 1 <= len(examples) <= 192:
        raise ValueError("usage examples must contain between 1 and 192 rows")
    train_rows = bundle["sender"]["train"]
    train_meanings = {
        tuple(row["private_meaning"].get(axis) for axis in split["attributes"])
        for row in train_rows
    }
    meanings_by_id = {
        row["meaning_id"]: tuple(row["values"])
        for row in split["meanings"] if row["meaning_id"] in set(split["train_meaning_ids"])
    }
    seen_ids: set[str] = set()
    seen_meanings: set[tuple[str, ...]] = set()
    clean: list[dict[str, Any]] = []
    for example in examples:
        if not isinstance(example, dict) or set(example) != {"meaning_id", "meaning", "message"}:
            raise ValueError("each usage example must have exactly meaning_id, meaning, and message")
        meaning_id, meaning, message = example["meaning_id"], example["meaning"], example["message"]
        if not isinstance(meaning_id, str) or meaning_id not in meanings_by_id or meaning_id in seen_ids:
            raise ValueError("usage example contains a duplicate or non-training meaning ID")
        if not isinstance(meaning, dict) or set(meaning) != set(split["attributes"]):
            raise ValueError("usage-example meaning does not match the task ontology")
        meaning_tuple = tuple(meaning[axis] for axis in split["attributes"])
        if meaning_tuple != meanings_by_id[meaning_id] or meaning_tuple not in train_meanings:
            raise ValueError("usage example meaning does not match the bound training ledger")
        if not isinstance(message, str) or not message.strip() or len(message.encode("utf-8")) > 4096:
            raise ValueError("usage-example message must be non-empty and at most 4096 UTF-8 bytes")
        seen_ids.add(meaning_id)
        seen_meanings.add(meaning_tuple)
        clean.append({"meaning_id": meaning_id, "meaning": dict(meaning), "message": message})
    if acquisition["method"] == "model_generated" and acquisition["generation_calls"] != len(clean):
        raise ValueError("model-generated usage traces must account for one generation call per example")
    digest = hashlib.sha256(payload).hexdigest()
    return clean, digest, {
        "acquisition": acquisition,
        "example_count": len(clean),
        "training_support_covered": len(seen_meanings),
    }


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


def select_candidate_sets(
    bundle: dict[str, Any], stage: str, set_count: int, *, set_offset: int = 0
) -> list[dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError("stage must be train, validation, or test")
    if isinstance(set_count, bool) or not isinstance(set_count, int) or not 1 <= set_count <= MAX_CANDIDATE_SETS_PER_BATCH:
        raise ValueError(f"set_count must be in 1..{MAX_CANDIDATE_SETS_PER_BATCH} per batch")
    if isinstance(set_offset, bool) or not isinstance(set_offset, int) or set_offset < 0:
        raise ValueError("set_offset must be a non-negative integer")
    sender, receiver, gold = (bundle[role][stage] for role in ("sender", "receiver", "gold"))
    if not (len(sender) == len(receiver) == len(gold)):
        raise ValueError("role ledgers are not aligned")
    ordered_set_ids = list(dict.fromkeys(row["candidate_set_id"] for row in gold))
    selected_set_ids = ordered_set_ids[set_offset:set_offset + set_count]
    if len(selected_set_ids) != set_count:
        raise ValueError("candidate-set offset/count exceeds the available stage sets")
    selected_set_id_set = set(selected_set_ids)
    picked = []
    for index, gold_row in enumerate(gold):
        set_id = gold_row["candidate_set_id"]
        if set_id in selected_set_id_set:
            picked.append({"sender": sender[index], "receiver": receiver[index], "gold": gold_row})
    expected_k = bundle["manifest"]["k"]
    if len(picked) != set_count * expected_k:
        raise ValueError("selected candidate sets are incomplete or the stage has too few sets")
    return picked


def validate_capability_ledger(
    path: Path | None,
    *,
    expected_calibration_episodes: Sequence[dict[str, Any]],
    evaluation_episodes: Sequence[dict[str, Any]],
    input_manifest_sha256: str,
    split_seed: int,
    split_sha256: str,
    evaluation_split_seed: int,
    task_seed: int,
    task_key_id: str,
    receiver_model: str,
    receiver_tokenizer_id: str,
    model_population_id: str,
) -> None:
    """Require a verified, all-correct train-only receiver screen before comparison runs."""
    if path is None:
        raise ValueError("communication conditions require --capability-ledger from an independent train-stage full_information screen")
    if split_seed == evaluation_split_seed:
        raise ValueError("capability screen must use an independent split seed from evaluation")
    candidate_count = (
        len(expected_calibration_episodes[0]["receiver"]["candidates"])
        if expected_calibration_episodes else 0
    )
    if candidate_count < 2 or len(expected_calibration_episodes) != CAPABILITY_CALIBRATION_SETS * candidate_count:
        raise ValueError("expected calibration episodes must contain exactly three complete balanced candidate sets")
    evaluation_candidate_count = (
        len(evaluation_episodes[0]["receiver"]["candidates"]) if evaluation_episodes else 0
    )
    if candidate_count != evaluation_candidate_count:
        raise ValueError("capability and evaluation candidate counts differ")
    try:
        result_path = _inside_project(path)
        rows = _jsonl(result_path)
        manifest_path = _inside_project(result_path.with_suffix(result_path.suffix + ".manifest.json"))
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"could not read capability result and manifest: {path}") from exc
    payload = result_path.read_bytes()
    if (
        not isinstance(run_manifest, dict)
        or run_manifest.get("schema") != "tlu.emergent-ood-run-manifest.v0.4"
        or run_manifest.get("results_file") != result_path.name
        or run_manifest.get("results_sha256") != hashlib.sha256(payload).hexdigest()
        or run_manifest.get("result_records") != len(rows)
    ):
        raise ValueError("capability result manifest or result hash is invalid")
    if len(rows) != len(expected_calibration_episodes):
        raise ValueError("capability ledger must contain exactly three complete balanced train candidate sets")
    expected_rows = {episode["gold"]["episode_id"]: episode for episode in expected_calibration_episodes}
    indexed = {row.get("episode_id"): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(expected_rows):
        raise ValueError("capability episode IDs do not match the expected training-only calibration block")
    eval_ids = {episode["gold"]["episode_id"] for episode in evaluation_episodes}
    if set(indexed) & eval_ids:
        raise ValueError("capability episodes overlap evaluation episode IDs")
    expected_manifest_fields = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "train",
        "conditions": ["full_information"],
        "candidate_sets": CAPABILITY_CALIBRATION_SETS,
        "candidate_count": candidate_count,
        "split_seed": split_seed,
        "split_sha256": split_sha256,
        "task_seed": task_seed,
        "task_key_id": task_key_id,
        "input_episode_manifest_sha256": input_manifest_sha256,
        "receiver_model": receiver_model,
        "receiver_tokenizer_id": receiver_tokenizer_id,
        "model_population_id": model_population_id,
    }
    for key, expected_value in expected_manifest_fields.items():
        if run_manifest.get(key) != expected_value:
            raise ValueError(f"capability run manifest {key} does not match this evaluation run")

    for episode_id, episode in expected_rows.items():
        row = indexed[episode_id]
        calls = row.get("model_calls", [])
        expected_candidates = [
            candidate["candidate_id"] for candidate in episode["receiver"]["candidates"]
        ]
        expected_target = next(
            candidate["attributes"]
            for candidate in episode["receiver"]["candidates"]
            if candidate["candidate_id"] == episode["gold"]["candidate_id"]
        )
        if (
            row.get("stage") != "train"
            or row.get("condition") != "full_information"
            or row.get("protocol", {}).get("policy_id") != "full_information"
            or row.get("experiment_id") != EXPERIMENT_ID
            or row.get("stratum", {}).get("task_id") != "four-attribute-higher-order-meaning-matching-v1"
            or row.get("stratum", {}).get("scorer_id") != SCORER_ID
            or row.get("outcome", {}).get("joint_success") is not True
            or row.get("outcome", {}).get("answer_format_valid") is not True
            or row.get("outcome", {}).get("answer_candidate_id") != episode["gold"]["candidate_id"]
            or row.get("meaning_id") != episode["gold"]["meaning_id"]
            or row.get("split_seed") != split_seed
            or row.get("task_seed") != task_seed
            or row.get("candidate_set_id") != episode["gold"]["candidate_set_id"]
            or row.get("trace", {}).get("candidate_ids_in_receiver_order") != expected_candidates
            or row.get("trace", {}).get("target_tuple_for_evaluator") != expected_target
            or row.get("transmissions") != []
            or len(calls) != 1
            or calls[0].get("agent") != "receiver"
            or calls[0].get("stage") != "final_answer"
            or calls[0].get("model") != receiver_model
            or calls[0].get("tokenizer") != receiver_tokenizer_id
            or calls[0].get("truncated") is not False
            or row.get("stratum", {}).get("model_population_id") != model_population_id
            or row.get("stratum", {}).get("task_parameters", {}).get("target_support_size") != 192
        ):
            raise ValueError("capability ledger is not a perfect, format-valid, matching full-information receiver screen")


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
    usage_examples: Sequence[dict[str, Any]] | None = None,
    split_seed: int, task_seed: int, model_population_id: str,
    wire_budget_bytes: int = DEFAULT_WIRE_BUDGET_BYTES,
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
    if condition == "usage_only_transfer":
        if not usage_examples:
            raise ValueError("usage_only_transfer requires validated training usage examples")
        # Deliberately expose only the semantic tuple and observed message. Provenance IDs
        # stay in the evaluator artifact and never cross the model boundary.
        receiver_context["training_examples"] = [
            {"meaning": example["meaning"], "message": example["message"]}
            for example in usage_examples
        ]
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
        protocol = _Protocol(
            condition, attributes, values, protocol_card=protocol_card,
            usage_examples=usage_examples,
        )
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
        wire_budget_bytes=wire_budget_bytes,
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
    exact_selection = answer_id == gold["candidate_id"]
    sender_model_id = getattr(sender_model, "model", getattr(sender_model, "model_name", "absent")) if sender_model is not None else "absent"
    receiver_model_id = getattr(receiver_model, "model", getattr(receiver_model, "model_name", "unknown"))
    transmissions = result.transmission_records()
    wall_seconds = time.perf_counter() - started
    return {
        "schema_version": "tlu.costs.v3",
        "schema": "tlu.emergent-ood-run.v0.4",
        "experiment_id": EXPERIMENT_ID,
        "inference_cluster_id": f"split={split_seed}",
        "candidate_set_cluster_id": gold["candidate_set_id"],
        "stage": stage,
        "split_seed": split_seed,
        "task_seed": task_seed,
        "episode_id": gold["episode_id"],
        "candidate_set_id": gold["candidate_set_id"],
        "meaning_id": gold["meaning_id"],
        "condition": condition,
        "communication_budget_bytes": wire_budget_bytes,
        "protocol_id": protocol.protocol_id,
        "outcome": {
            "answer_format_valid": answer_valid,
            "answer_candidate_id": answer_id,
            "target_candidate_id": gold["candidate_id"],
            "exact_selection": exact_selection,
            "joint_success": exact_selection,
            "answer_score": 1.0 if exact_selection else 0.0,
            "bayes_no_message_reference": 1 / len(candidate_ids),
        },
        "protocol": {
            "policy_id": condition,
            "code_id": protocol.protocol_id,
            "decoder_id": SCORER_ID,
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
            "wall_seconds": wall_seconds,
            "communication_budget_bytes": wire_budget_bytes,
            "protocol_card_bytes": {
                "sender_instruction": len(protocol.agent_instructions["sender"].encode("utf-8")),
                "receiver_instruction": len(protocol.agent_instructions["receiver"].encode("utf-8")),
            },
            "usage_example_count": len(usage_examples or []) if condition == "usage_only_transfer" else 0,
            "usage_example_bytes_per_receiver_request": (
                len(json.dumps(receiver_context["training_examples"], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                if condition == "usage_only_transfer" else 0
            ),
        },
        "trace": {
            "message": message,
            "answer": answer_text,
            "candidate_ids_in_receiver_order": candidate_ids,
            "target_tuple_for_evaluator": target_row["attributes"],
            "transmissions": transmissions,
            "stop_reason": result.stop_reason,
        },
        "stratum": {
            "experiment_id": EXPERIMENT_ID,
            "task_id": "four-attribute-higher-order-meaning-matching-v1",
            "split": stage,
            "scorer_id": SCORER_ID,
            "model_population_id": model_population_id,
            "agent_models": {"sender": sender_model_id, "receiver": receiver_model_id},
            "task_parameters": {
                "candidate_count": len(candidate_ids),
                "target_support_size": {"train": 192, "validation": 16, "test": 48}[stage],
                "communication_budget_bytes": wire_budget_bytes,
            },
        },
        "transmissions": transmissions,
        "model_calls": calls,
        "runtime": {"wall_seconds": wall_seconds},
        "setup": [],
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


def _checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".checkpoint.json")


def _checkpoint_rows_digest(rows: Sequence[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = {
        **checkpoint,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_sha256": _checkpoint_rows_digest(checkpoint["rows"]),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("run checkpoint is missing or invalid") from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or not isinstance(checkpoint.get("rows"), list)
        or any(not isinstance(row, dict) for row in checkpoint["rows"])
        or checkpoint.get("rows_sha256") != _checkpoint_rows_digest(checkpoint["rows"])
        or not isinstance(checkpoint.get("resource_preflight_sha256s"), list)
        or any(not isinstance(digest, str) or len(digest) != 64 for digest in checkpoint["resource_preflight_sha256s"])
        or not isinstance(checkpoint.get("run_config"), dict)
        or checkpoint.get("run_signature") != _run_signature(checkpoint["run_config"])
    ):
        raise ValueError("run checkpoint failed schema or content-integrity validation")
    return checkpoint


def _run_signature(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_checkpoint_prefix(
    rows: Sequence[dict[str, Any]], expected_pairs: Sequence[tuple[str, str]],
) -> None:
    if len(rows) > len(expected_pairs):
        raise ValueError("run checkpoint contains more result rows than this batch")
    for index, row in enumerate(rows):
        if (row.get("episode_id"), row.get("condition")) != expected_pairs[index]:
            raise ValueError("run checkpoint rows are not an exact prefix of this batch order")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(".cache/emergent_ood_v0_4/episodes"))
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--stage", choices=STAGES, default="validation")
    parser.add_argument("--sets", type=int, default=1)
    parser.add_argument(
        "--set-offset", type=int, default=0,
        help="number of complete candidate sets to skip within the stage (default: 0)",
    )
    parser.add_argument(
        "--wire-budget-bytes", type=int, default=DEFAULT_WIRE_BUDGET_BYTES,
        help="maximum complete application-layer message bytes per episode (default: 4096)",
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["natural_language"])
    parser.add_argument("--protocol-card", type=Path, help="shared sender/receiver card JSON for the shared_protocol_card condition")
    parser.add_argument("--usage-examples", type=Path, help="train-only meaning/message artifact required by usage_only_transfer")
    parser.add_argument("--output", type=Path, default=Path(".cache/emergent_ood_v0_4/runs/validation.jsonl"))
    parser.add_argument("--resume", action="store_true", help="resume the matching atomic checkpoint for this output path")
    parser.add_argument("--capability-ledger", type=Path, help="verified train-only full_information screen required before message conditions")
    parser.add_argument("--capability-input-dir", type=Path, help="episode bundle used for the independent train-only receiver screen")
    parser.add_argument("--capability-split-seed", type=int, help="split seed for the independent train-only receiver screen")
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
    if args.wire_budget_bytes < 0:
        parser.error("--wire-budget-bytes must be non-negative")

    try:
        bundle, split = load_episode_bundle(args.input_dir, split_seed=args.split_seed)
        episodes = select_candidate_sets(bundle, args.stage, args.sets, set_offset=args.set_offset)
        protocol_card, protocol_card_digest = (
            load_protocol_card(args.protocol_card) if args.protocol_card else (None, None)
        )
        input_manifest_sha256 = hashlib.sha256(
            (_inside_project(args.input_dir) / "manifest.json").read_bytes()
        ).hexdigest()
        usage_examples, usage_examples_digest, usage_metadata = (None, None, None)
        if "usage_only_transfer" in args.conditions:
            if args.usage_examples is None or protocol_card is None:
                raise ValueError("usage_only_transfer requires both --protocol-card and --usage-examples")
            usage_examples, usage_examples_digest, usage_metadata = load_usage_examples(
                args.usage_examples, split=split, bundle=bundle,
                protocol_card=protocol_card, protocol_card_sha256=protocol_card_digest,
                training_episode_manifest_sha256=input_manifest_sha256,
            )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if any(condition in args.conditions for condition in ("shared_protocol_card", "usage_only_transfer")) and protocol_card is None:
        parser.error("shared_protocol_card and usage_only_transfer require --protocol-card")
    if args.protocol_card is not None and not any(condition in args.conditions for condition in ("shared_protocol_card", "usage_only_transfer")):
        parser.error("--protocol-card requires shared_protocol_card or usage_only_transfer")
    if "usage_only_transfer" not in args.conditions and args.usage_examples is not None:
        parser.error("--usage-examples is only valid with usage_only_transfer")
    if args.stage == "train" and (
        args.conditions != ["full_information"] or args.sets != CAPABILITY_CALIBRATION_SETS
        or args.set_offset != 0
    ):
        parser.error(
            f"train stage is reserved for exactly {CAPABILITY_CALIBRATION_SETS} full-information capability sets at offset zero"
        )
    if args.capability_ledger is not None and args.stage == "train":
        parser.error("train-stage calibration cannot consume another capability ledger")
    if args.stage == "train" and (args.capability_input_dir is not None or args.capability_split_seed is not None):
        parser.error("train-stage calibration cannot consume an independent capability bundle")
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
            "candidate_set_offset": args.set_offset,
            "available_candidate_sets": len({row["candidate_set_id"] for row in bundle["gold"][args.stage]}),
            "episodes": len(episodes),
            "candidate_count": bundle["manifest"]["k"],
            "analytic_no_message_accuracy": 1 / bundle["manifest"]["k"],
            "wire_budget_bytes": args.wire_budget_bytes,
            "protocol_card_sha256": protocol_card_digest,
            "usage_examples_sha256": usage_examples_digest,
            "usage_example_count": usage_metadata["example_count"] if usage_metadata else 0,
            "planned_model_calls": calls_planned,
            "maximum_model_calls_per_batch": MAX_MODEL_CALLS_PER_BATCH,
            "resume_supported": True,
            "checkpoint_path": str(_checkpoint_path(_inside_project(args.output))),
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
    needs_capability = any(
        condition in {"natural_language", "autoform", "json", "symbolic", "shared_protocol_card", "usage_only_transfer"}
        for condition in args.conditions
    )
    if needs_capability:
        try:
            if args.capability_input_dir is None or args.capability_split_seed is None:
                raise ValueError("message conditions require --capability-input-dir and --capability-split-seed for independent calibration")
            if args.capability_split_seed == args.split_seed:
                raise ValueError("capability split seed must differ from the evaluation split seed")
            calibration_bundle, calibration_split = load_episode_bundle(
                args.capability_input_dir, split_seed=args.capability_split_seed
            )
            calibration_manifest = calibration_bundle["manifest"]
            evaluation_manifest = bundle["manifest"]
            if (
                calibration_manifest["task_key_id"] != evaluation_manifest["task_key_id"]
                or calibration_manifest["task_seed"] != evaluation_manifest["task_seed"]
                or calibration_manifest["k"] != evaluation_manifest["k"]
                or calibration_split["attributes"] != split["attributes"]
                or calibration_split["values_by_attribute"] != split["values_by_attribute"]
            ):
                raise ValueError("independent calibration bundle does not match the evaluation task, key, and ontology")
            calibration_episodes = select_candidate_sets(
                calibration_bundle, "train", CAPABILITY_CALIBRATION_SETS
            )
            validate_capability_ledger(
                args.capability_ledger,
                expected_calibration_episodes=calibration_episodes,
                evaluation_episodes=episodes,
                input_manifest_sha256=hashlib.sha256(
                    (_inside_project(args.capability_input_dir) / "manifest.json").read_bytes()
                ).hexdigest(),
                split_seed=args.capability_split_seed,
                split_sha256=calibration_split["split_sha256"],
                evaluation_split_seed=args.split_seed,
                task_seed=calibration_manifest["task_seed"],
                task_key_id=calibration_manifest["task_key_id"],
                receiver_model=args.receiver_model,
                receiver_tokenizer_id=args.receiver_tokenizer_id,
                model_population_id=args.model_population_id,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    needs_sender = any(
        condition in {"natural_language", "autoform", "json", "symbolic", "shared_protocol_card", "usage_only_transfer"}
        for condition in args.conditions
    )
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
    checkpoint_path = _inside_project(_checkpoint_path(output_path))
    if args.resume and (output_path.exists() or output_manifest_path.exists()):
        parser.error("cannot resume because a completed output or manifest already exists")
    if not args.force and not args.resume and (output_path.exists() or output_manifest_path.exists()):
        parser.error(f"refusing to overwrite {output_path}; pass --force explicitly")

    if args.resume and not checkpoint_path.exists():
        parser.error(f"no checkpoint exists to resume: {checkpoint_path}")
    if not args.resume and checkpoint_path.exists():
        parser.error(f"checkpoint already exists; pass --resume to continue it: {checkpoint_path}")

    input_manifest_path = _inside_project(_inside_project(args.input_dir) / "manifest.json")
    capability_ledger_sha256 = (
        hashlib.sha256(_inside_project(args.capability_ledger).read_bytes()).hexdigest()
        if needs_capability and args.capability_ledger is not None else None
    )
    capability_ledger_manifest_sha256 = (
        hashlib.sha256(_inside_project(args.capability_ledger).with_suffix(
            _inside_project(args.capability_ledger).suffix + ".manifest.json"
        ).read_bytes()).hexdigest()
        if needs_capability and args.capability_ledger is not None else None
    )
    capability_bundle_manifest_sha256 = (
        hashlib.sha256((_inside_project(args.capability_input_dir) / "manifest.json").read_bytes()).hexdigest()
        if needs_capability and args.capability_input_dir is not None else None
    )
    runtime_source = ROOT / "tacit" / "runtime.py"
    run_config = {
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "conditions": args.conditions,
        "sets": args.sets,
        "set_offset": args.set_offset,
        "wire_budget_bytes": args.wire_budget_bytes,
        "split_seed": args.split_seed,
        "split_sha256": split["split_sha256"],
        "task_seed": bundle["manifest"]["task_seed"],
        "task_key_id": bundle["manifest"]["task_key_id"],
        "input_episode_manifest_sha256": input_manifest_sha256,
        "protocol_card_sha256": protocol_card_digest,
        "usage_examples_sha256": usage_examples_digest,
        "capability_ledger_sha256": capability_ledger_sha256,
        "capability_ledger_manifest_sha256": capability_ledger_manifest_sha256,
        "capability_bundle_manifest_sha256": capability_bundle_manifest_sha256,
        "capability_split_seed": args.capability_split_seed if needs_capability else None,
        "model_population_id": args.model_population_id,
        "sender_endpoint": sender_endpoint,
        "receiver_endpoint": receiver_endpoint,
        "sender_model": args.sender_model if needs_sender else None,
        "receiver_model": args.receiver_model,
        "sender_tokenizer_id": args.sender_tokenizer_id if needs_sender else None,
        "receiver_tokenizer_id": args.receiver_tokenizer_id,
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_source_sha256": hashlib.sha256(runtime_source.read_bytes()).hexdigest(),
    }
    signature = _run_signature(run_config)
    expected_pairs = [
        (episode["gold"]["episode_id"], condition)
        for episode in episodes for condition in args.conditions
    ]
    current_preflight_sha256 = hashlib.sha256(preflight_path.read_bytes()).hexdigest() if preflight_path else None
    if args.resume:
        try:
            checkpoint = _read_checkpoint(checkpoint_path)
            if checkpoint.get("run_signature") != signature:
                raise ValueError("checkpoint configuration does not match this batch; no rows were reused")
            rows = checkpoint["rows"]
            _validate_checkpoint_prefix(rows, expected_pairs)
            checkpoint["resource_preflight_sha256s"] = list(checkpoint["resource_preflight_sha256s"])
            if current_preflight_sha256 and current_preflight_sha256 not in checkpoint["resource_preflight_sha256s"]:
                checkpoint["resource_preflight_sha256s"].append(current_preflight_sha256)
            checkpoint["resumed"] = True
            _write_checkpoint(checkpoint_path, checkpoint)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    else:
        if current_preflight_sha256 is None:
            parser.error("execution requires a resource preflight report")
        rows = []
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "run_signature": signature,
            "run_config": run_config,
            "rows": rows,
            "resource_preflight_sha256s": [current_preflight_sha256],
            "resumed": False,
        }
        _write_checkpoint(checkpoint_path, checkpoint)

    sender_client = OpenAICompatibleClient(
        sender_endpoint, args.sender_model, timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_tokens=160, follow_redirects=False, temperature=0.0,
    ) if needs_sender else None
    receiver_client = OpenAICompatibleClient(
        receiver_endpoint, args.receiver_model, timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_tokens=48, follow_redirects=False, temperature=0.0,
    )
    candidate_graph = conflict_graph_summary(episodes)
    checkpoint_rows_reused = len(rows) if args.resume else 0
    completed = len(rows)
    for episode in episodes:
        for condition in args.conditions:
            if completed:
                completed -= 1
                continue
            row = run_condition(
                episode=episode, condition=condition, stage=args.stage,
                sender_model=sender_client, receiver_model=receiver_client,
                sender_tokenizer_id=args.sender_tokenizer_id or None,
                receiver_tokenizer_id=args.receiver_tokenizer_id,
                attributes=split["attributes"], values=split["values_by_attribute"],
                protocol_card=protocol_card if condition in {"shared_protocol_card", "usage_only_transfer"} else None,
                usage_examples=usage_examples if condition == "usage_only_transfer" else None,
                split_seed=args.split_seed, task_seed=bundle["manifest"]["task_seed"],
                model_population_id=args.model_population_id,
                wire_budget_bytes=args.wire_budget_bytes,
            )
            row["stratum"]["candidate_suite_conflict_graph"] = candidate_graph
            rows.append(row)
            checkpoint["rows"] = rows
            _write_checkpoint(checkpoint_path, checkpoint)
    if len(rows) != len(expected_pairs):
        parser.error("completed results do not cover the exact planned episode-condition batch")
    preflight_digests = list(checkpoint["resource_preflight_sha256s"])
    output_digest = _write_results(
        rows,
        output_path,
        force=args.force,
        manifest={
            "experiment_id": EXPERIMENT_ID,
            "stage": args.stage,
            "conditions": args.conditions,
            "protocol_card_sha256": protocol_card_digest,
            "usage_examples_sha256": usage_examples_digest,
            "usage_example_metadata": usage_metadata,
            "candidate_sets": args.sets,
            "candidate_set_offset": args.set_offset,
            "candidate_count": bundle["manifest"]["k"],
            "communication_budget_bytes": args.wire_budget_bytes,
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
            "resource_preflight_sha256": current_preflight_sha256,
            "resource_preflight_sha256s": preflight_digests,
            "run_signature": signature,
            "resumed_from_checkpoint": bool(args.resume),
            "checkpoint_rows_reused": checkpoint_rows_reused,
            "capability_split_seed": args.capability_split_seed if needs_capability else None,
            "capability_input_episode_manifest_sha256": capability_bundle_manifest_sha256,
            "capability_ledger_sha256": capability_ledger_sha256,
            "capability_ledger_manifest_sha256": capability_ledger_manifest_sha256,
            "input_episode_manifest_sha256": input_manifest_sha256,
        },
    )
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps({
        "mode": "executed", "episodes": len(episodes), "conditions": args.conditions,
        "model_calls": calls_planned, "output": str(output_path), "sha256": output_digest,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
