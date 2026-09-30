"""Tokenizer-only scaling audit of explicit format instructions and payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.emergent_ood_v0_4.compact_fields import encode_fields as encode_compact_fields  # noqa: E402
from experiments.emergent_ood_v0_4.runner import _Protocol  # noqa: E402
from research.audit_protocol_token_costs import (  # noqa: E402
    FINAL_ANSWER_INSTRUCTION,
    TASK,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
    _project_path,
    _summary,
    _user_json,
)


SCHEMA = "tlu.protocol-token-scaling-audit.v1"
DIMENSIONS = (3, 4, 6, 8, 10)
CARDINALITIES = (2, 4, 8, 10)
EPISODES_PER_SETTING = 16
CANDIDATES_PER_EPISODE = 4
CONDITIONS = ("json", "shared_protocol_card", "symbolic")
EXTENSION_DIMENSIONS = (20, 40, 80)


def _tuple_from_support_rank(rank: int, *, dimensions: int, values: int) -> tuple[int, ...]:
    """Map [0,V^(d-1)) bijectively into the identity-rank modular holdout."""
    support_size = values ** (dimensions - 1)
    if not 0 <= rank < support_size:
        raise ValueError("support rank is out of range")
    coordinates = [0] * (dimensions - 1)
    remainder = rank
    for index in range(dimensions - 2, -1, -1):
        coordinates[index] = remainder % values
        remainder //= values
    coordinates.append((-sum(coordinates)) % values)
    return tuple(coordinates)


def synthetic_episodes(
    *, dimensions: int, values: int, episode_count: int = EPISODES_PER_SETTING,
) -> list[dict[str, Any]]:
    """Build deterministic role-local prompt records on a modular held-out set."""
    if dimensions < 3 or values < 2 or values > 10:
        raise ValueError("require dimensions >= 3 and 2 <= values <= 10")
    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    support_size = values ** (dimensions - 1)
    if support_size < CANDIDATES_PER_EPISODE:
        raise ValueError("held-out support cannot supply four distinct candidates")

    attributes = [f"axis{axis:02d}" for axis in range(dimensions)]
    values_by_attribute = {
        attribute: [f"v{axis:02d}_{value:02d}" for value in range(values)]
        for axis, attribute in enumerate(attributes)
    }
    offsets = [
        (support_size * index) // CANDIDATES_PER_EPISODE
        for index in range(CANDIDATES_PER_EPISODE)
    ]
    if len(set(offsets)) != CANDIDATES_PER_EPISODE:
        raise ValueError("support is too small for four distinct candidate ranks")

    episodes = []
    for episode_index in range(episode_count):
        target_rank = (episode_index * support_size) // episode_count
        target_indices = _tuple_from_support_rank(
            target_rank, dimensions=dimensions, values=values
        )
        target = {
            attribute: values_by_attribute[attribute][target_indices[axis]]
            for axis, attribute in enumerate(attributes)
        }
        ranked = [
            _tuple_from_support_rank(
                (target_rank + offset) % support_size,
                dimensions=dimensions,
                values=values,
            )
            for offset in offsets
        ]
        if len(set(ranked)) != CANDIDATES_PER_EPISODE:
            raise ValueError("candidate generator produced duplicate meanings")
        gold_position = episode_index % CANDIDATES_PER_EPISODE
        ordered = ranked[-gold_position:] + ranked[:-gold_position] if gold_position else ranked
        candidates = [
            {
                "candidate_id": f"c{candidate_index:04d}",
                "attributes": {
                    attribute: values_by_attribute[attribute][candidate[axis]]
                    for axis, attribute in enumerate(attributes)
                },
            }
            for candidate_index, candidate in enumerate(ordered)
        ]
        if candidates[gold_position]["attributes"] != target:
            raise AssertionError("rotated gold candidate does not match the sender target")
        episodes.append({
            "episode_index": episode_index,
            "target_rank": target_rank,
            "sender_target": target,
            "receiver_candidates": candidates,
            "gold_position": gold_position,
        })
    return episodes


def audit_setting(
    *,
    dimensions: int,
    values: int,
    tokenizer: Any,
    protocol_card: dict[str, str],
) -> dict[str, Any]:
    attributes = [f"axis{axis:02d}" for axis in range(dimensions)]
    values_by_attribute = {
        attribute: [f"v{axis:02d}_{value:02d}" for value in range(values)]
        for axis, attribute in enumerate(attributes)
    }
    protocols = {
        condition: _Protocol(
            condition,
            attributes,
            values_by_attribute,
            protocol_card=protocol_card if condition == "shared_protocol_card" else None,
        )
        for condition in CONDITIONS
    }
    episodes = synthetic_episodes(dimensions=dimensions, values=values)

    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    per_condition: dict[str, list[dict[str, int]]] = {condition: [] for condition in CONDITIONS}
    for episode in episodes:
        target = episode["sender_target"]
        candidates = episode["receiver_candidates"]
        sender_private_context = json.dumps(
            {"private_meaning": target}, ensure_ascii=False, separators=(",", ":")
        )
        receiver_private_context = json.dumps(
            {"candidates": candidates}, ensure_ascii=False, separators=(",", ":")
        )
        canonical_target = {attribute: target[attribute] for attribute in attributes}
        messages = {
            "json": json.dumps(canonical_target, ensure_ascii=False, separators=(",", ":")),
            "shared_protocol_card": encode_compact_fields(target),
            "symbolic": "".join(
                str(values_by_attribute[attribute].index(target[attribute]))
                for attribute in attributes
            ),
        }
        for condition, protocol in protocols.items():
            sender_system = protocol.agent_instructions["sender"]
            receiver_system = (
                protocol.agent_instructions["receiver"]
                + "\n\n"
                + FINAL_ANSWER_INSTRUCTION
            )
            sender_user = _user_json({
                "task": TASK,
                "private_context": sender_private_context,
                "visible_transcript": [],
                "instruction": "Send exactly the next message for this turn.",
            })
            receiver_user = _user_json({
                "task": TASK,
                "private_context": receiver_private_context,
                "visible_transcript": [{"sender": "sender", "message": messages[condition]}],
                "instruction": "Return the final task answer for external scoring.",
            })
            sender_input_tokens = token_count(sender_system) + token_count(sender_user)
            receiver_input_tokens = token_count(receiver_system) + token_count(receiver_user)
            sender_output_tokens = token_count(messages[condition])
            per_condition[condition].append({
                "sender_instruction_tokens": token_count(sender_system),
                "receiver_instruction_tokens": token_count(receiver_system),
                "total_instruction_tokens": token_count(sender_system) + token_count(receiver_system),
                "sender_context_tokens": token_count(sender_user),
                "receiver_context_tokens": token_count(receiver_user),
                "total_input_content_tokens": sender_input_tokens + receiver_input_tokens,
                "ideal_sender_message_output_tokens": sender_output_tokens,
                "known_total_content_tokens": sender_input_tokens + receiver_input_tokens + sender_output_tokens,
                "sender_instruction_bytes": len(sender_system.encode("utf-8")),
                "receiver_instruction_bytes": len(receiver_system.encode("utf-8")),
                "ideal_sender_message_bytes": len(messages[condition].encode("utf-8")),
            })

    summaries = {
        condition: _summary(rows)
        for condition, rows in per_condition.items()
    }
    deltas = {}
    for condition in ("shared_protocol_card", "symbolic"):
        deltas[condition] = round(
            sum(
                alternative["known_total_content_tokens"] - baseline["known_total_content_tokens"]
                for baseline, alternative in zip(per_condition["json"], per_condition[condition])
            ) / len(episodes),
            3,
        )
    return {
        "dimensions": dimensions,
        "values_per_dimension": values,
        "heldout_support_size": values ** (dimensions - 1),
        "episode_count": len(episodes),
        "candidate_count": CANDIDATES_PER_EPISODE,
        "conditions": summaries,
        "known_total_token_delta_vs_json": deltas,
    }


def build_report(
    tokenizer_json: Path,
    *,
    dimensions: Sequence[int] = DIMENSIONS,
    cardinalities: Sequence[int] = CARDINALITIES,
    preregistration_path: Path | None = None,
    schema: str = SCHEMA,
    exploratory: bool = False,
) -> dict[str, Any]:
    tokenizer_json = _project_path(tokenizer_json)
    tokenizer_bytes = tokenizer_json.read_bytes()
    tokenizer_digest = hashlib.sha256(tokenizer_bytes).hexdigest()
    if tokenizer_digest != TOKENIZER_SHA256:
        raise ValueError("tokenizer hash does not match the pinned Qwen3-4B revision")
    try:
        from tokenizers import Tokenizer, __version__ as tokenizers_version
    except ImportError as exc:
        raise RuntimeError("install the optional `tokenizers` package to run this audit") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    card_path = ROOT / "examples" / "compact_labeled_fields_v3.json"
    card_bytes = card_path.read_bytes()
    card_digest = hashlib.sha256(card_bytes).hexdigest()
    if card_digest != "2646cd19c10aa5f4bec19d8e38fdb43eac6c15238630b86f79c794bff1a46aa2":
        raise ValueError("the compact-fields card is not the pinned v0.3 artifact")
    card = json.loads(card_bytes.decode("utf-8"))
    prereg_path = preregistration_path or ROOT / "research" / "PROTOCOL_TOKEN_SCALING_V0_1.md"
    runner_path = ROOT / "experiments" / "emergent_ood_v0_4" / "runner.py"
    codec_path = ROOT / "experiments" / "emergent_ood_v0_4" / "compact_fields.py"
    report = {
        "schema": schema,
        "tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "sha256": tokenizer_digest,
            "library": "tokenizers",
            "library_version": tokenizers_version,
            "special_tokens_added": False,
        },
        "python_version": platform.python_version(),
        "preregistration_sha256": hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
        "source_sha256": {
            "runner_py": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
            "compact_fields_py": hashlib.sha256(codec_path.read_bytes()).hexdigest(),
            "protocol_card_json": card_digest,
        },
        "measurement": (
            "Deterministic synthetic prompt-cost sweep using exact v0.4 system/user content serialization. "
            "Known total includes both agents' input content and ideal sender message output; receiver output, "
            "chat-template tokens, model outcomes, and inference compute are excluded."
        ),
        "sweep": [
            audit_setting(
                dimensions=dimension,
                values=values,
                tokenizer=tokenizer,
                protocol_card=card,
            )
            for dimension in dimensions
            for values in cardinalities
        ],
    }
    if exploratory:
        report["status"] = "post-sweep exploratory extension"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension", action="store_true", help="run the frozen post-sweep exploratory dimensions")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = _project_path(args.output)
    if output.exists() and not args.force:
        parser.error("output already exists; choose a new project-local path")
    if args.extension:
        preregistration_path = ROOT / "research" / "PROTOCOL_TOKEN_SCALING_EXTENSION_PREREG_V0_1.md"
        prior_report_path = ROOT / "research" / "data" / "PROTOCOL_TOKEN_SCALING_V0_1.json"
        if not prior_report_path.is_file():
            parser.error("the frozen v0.1 sweep JSON is required before the exploratory extension")
        report = build_report(
            args.tokenizer_json,
            dimensions=EXTENSION_DIMENSIONS,
            preregistration_path=preregistration_path,
            schema="tlu.protocol-token-scaling-extension.v1",
            exploratory=True,
        )
        report["prior_frozen_sweep_sha256"] = hashlib.sha256(prior_report_path.read_bytes()).hexdigest()
        report["exploratory_range"] = {
            "dimensions": list(EXTENSION_DIMENSIONS),
            "cardinalities": list(CARDINALITIES),
            "episodes_per_setting": EPISODES_PER_SETTING,
            "prediction": "P19c",
        }
    else:
        report = build_report(args.tokenizer_json)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
