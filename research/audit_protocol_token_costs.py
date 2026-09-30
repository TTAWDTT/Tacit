"""Count exact Qwen tokenizer content tokens for matched v0.4 JSON and v3 prompts.

This offline audit reconstructs the text content passed to each OpenAI-style
chat call. It does not load model weights, invoke an endpoint, or claim exact
provider usage (chat-template/special-token policy remains server-dependent).
"""
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
from experiments.emergent_ood_v0_4.runner import _Protocol, load_episode_bundle  # noqa: E402


TOKENIZER_REPO = "Qwen/Qwen3-4B"
TOKENIZER_REVISION = "eb971e9fb1f41c13b5e5a56e56886305c5ad94a0"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
PROTOCOL_CARD_SHA256 = "2646cd19c10aa5f4bec19d8e38fdb43eac6c15238630b86f79c794bff1a46aa2"
TOKENIZER_SCHEMA = "tlu.protocol-token-cost-audit.v1"
TASK = "Select the candidate ID whose full tuple is the sender's private meaning."
FINAL_ANSWER_INSTRUCTION = "Return only the exact candidate_id of your selected candidate for external scoring."
CONDITIONS = ("json", "shared_protocol_card")


def _project_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("all paths must stay inside the project") from exc
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _user_json(value: Mapping[str, Any]) -> str:
    # Match tacit.runtime.exchange_dialogue exactly: default JSON separators,
    # UTF-8 characters preserved, and outer keys sorted.
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _summary(rows: Sequence[Mapping[str, int]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty episode set")
    keys = tuple(rows[0])
    report: dict[str, Any] = {"episodes": len(rows)}
    for key in keys:
        values = [row[key] for row in rows]
        report[key] = {
            "mean": round(sum(values) / len(values), 3),
            "min": min(values),
            "max": max(values),
        }
    return report


def audit_bundle(
    *,
    bundle_path: Path,
    split_seed: int,
    ontology: str,
    tokenizer: Any,
    protocol_card: dict[str, str],
) -> dict[str, Any]:
    bundle_path = _project_path(bundle_path)
    bundle, split = load_episode_bundle(bundle_path, split_seed=split_seed)
    attrs = split["attributes"]
    values: dict[str, list[str]] = {attribute: [] for attribute in attrs}
    protocols = {
        condition: _Protocol(
            condition,
            attrs,
            values,
            protocol_card=protocol_card if condition == "shared_protocol_card" else None,
        )
        for condition in CONDITIONS
    }

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    stage = "test"
    senders = bundle["sender"][stage]
    receivers = bundle["receiver"][stage]
    gold_rows = bundle["gold"][stage]
    if not (len(senders) == len(receivers) == len(gold_rows)):
        raise ValueError("test role ledgers are not aligned")

    by_condition: dict[str, list[dict[str, int]]] = {name: [] for name in CONDITIONS}
    for sender, receiver, gold in zip(senders, receivers, gold_rows):
        target = sender["private_meaning"]
        candidate_ids = {row["candidate_id"] for row in receiver["candidates"]}
        if gold["candidate_id"] not in candidate_ids:
            raise ValueError("gold answer is absent from the receiver's private candidates")

        sender_private_context = json.dumps(
            {"private_meaning": target}, ensure_ascii=False, separators=(",", ":")
        )
        receiver_private_context = json.dumps(
            {"candidates": receiver["candidates"]}, ensure_ascii=False, separators=(",", ":")
        )
        canonical_target = {attribute: target[attribute] for attribute in attrs}
        messages = {
            "json": json.dumps(canonical_target, ensure_ascii=False, separators=(",", ":")),
            "shared_protocol_card": encode_compact_fields(target),
        }
        for condition, protocol in protocols.items():
            sender_user = _user_json({
                "task": TASK,
                "private_context": sender_private_context,
                "visible_transcript": [],
                "instruction": "Send exactly the next message for this turn.",
            })
            transcript = [{"sender": "sender", "message": messages[condition]}]
            receiver_user = _user_json({
                "task": TASK,
                "private_context": receiver_private_context,
                "visible_transcript": transcript,
                "instruction": "Return the final task answer for external scoring.",
            })
            sender_system = protocol.agent_instructions["sender"]
            receiver_system = (
                protocol.agent_instructions["receiver"]
                + "\n\n"
                + FINAL_ANSWER_INSTRUCTION
            )
            sender_input = count(sender_system) + count(sender_user)
            receiver_input = count(receiver_system) + count(receiver_user)
            sender_output = count(messages[condition])
            row = {
                "sender_input_content_tokens": sender_input,
                "receiver_input_content_tokens": receiver_input,
                "known_content_input_tokens": sender_input + receiver_input,
            }
            row["ideal_sender_message_output_tokens"] = sender_output
            row["known_content_input_plus_message_output_tokens"] = (
                sender_input + receiver_input + sender_output
            )
            by_condition[condition].append(row)

    manifest_path = bundle_path / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    summaries = {condition: _summary(rows) for condition, rows in by_condition.items()}
    deltas = []
    for json_row, card_row in zip(by_condition["json"], by_condition["shared_protocol_card"]):
        deltas.append(
            card_row["known_content_input_plus_message_output_tokens"]
            - json_row["known_content_input_plus_message_output_tokens"]
        )
    return {
        "ontology": ontology,
        "split_seed": split_seed,
        "task_seed": bundle["manifest"].get("task_seed"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "episode_count": len(senders),
        "conditions": summaries,
        "shared_card_minus_json_known_content_tokens_per_episode": {
            "mean": round(sum(deltas) / len(deltas), 3),
            "min": min(deltas),
            "max": max(deltas),
        },
    }


def build_report(tokenizer_json: Path, bundles: Sequence[tuple[str, Path, int]]) -> dict[str, Any]:
    tokenizer_json = _project_path(tokenizer_json)
    raw_tokenizer = tokenizer_json.read_bytes()
    digest = hashlib.sha256(raw_tokenizer).hexdigest()
    if digest != TOKENIZER_SHA256:
        raise ValueError("tokenizer JSON SHA-256 does not match the pinned Qwen3-4B revision")
    try:
        from tokenizers import Tokenizer, __version__ as tokenizers_version
    except ImportError as exc:
        raise RuntimeError("install the optional `tokenizers` package to run this audit") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    card_path = ROOT / "examples" / "compact_labeled_fields_v3.json"
    card_bytes = card_path.read_bytes()
    if hashlib.sha256(card_bytes).hexdigest() != PROTOCOL_CARD_SHA256:
        raise ValueError("protocol card SHA-256 does not match the pinned v0.3 card")
    card = _read_json(card_path)
    source_files = {
        "runner_py": ROOT / "experiments" / "emergent_ood_v0_4" / "runner.py",
        "compact_fields_py": ROOT / "experiments" / "emergent_ood_v0_4" / "compact_fields.py",
        "protocol_card_json": card_path,
    }

    results = [
        audit_bundle(
            bundle_path=path,
            split_seed=seed,
            ontology=ontology,
            tokenizer=tokenizer,
            protocol_card=card,
        )
        for ontology, path, seed in bundles
    ]
    return {
        "schema": TOKENIZER_SCHEMA,
        "tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "sha256": digest,
            "library": "tokenizers",
            "library_version": tokenizers_version,
            "special_tokens_added": False,
        },
        "python_version": platform.python_version(),
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files.items()
        },
        "measurement": (
            "Token counts cover exactly serialized system/user message content from the runner, "
            "excluding model chat-template/special tokens. Message output is ideal canonical "
            "serialization; receiver final-answer output is excluded. These are not provider "
            "usage measurements and do not include inference compute or task outcomes."
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--default-bundle", type=Path, required=True)
    parser.add_argument("--robotics-bundle", type=Path, required=True)
    parser.add_argument("--music-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing project-local report")
    args = parser.parse_args()
    output = _project_path(args.output)
    if output.exists() and not args.force:
        parser.error("output already exists; choose a new project-local path")
    report = build_report(
        args.tokenizer_json,
        [
            ("default", args.default_bundle, 23),
            ("robotics", args.robotics_bundle, 31),
            ("music", args.music_bundle, 31),
        ],
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
