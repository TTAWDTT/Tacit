"""Generate auditable sender demonstrations from training meanings only.

Dry-run is the default. Execute mode makes at most 12 sequential requests to
an operator-started loopback endpoint after a fresh passing resource preflight.
It never starts a model server or downloads model files.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.emergent_ood_v0_3.runner import validate_resource_preflight
from experiments.emergent_ood_v0_4.episodes import _inside_project
from experiments.emergent_ood_v0_4.runner import (
    OpenAICompatibleClient,
    _endpoint_port,
    _loopback_url,
    load_episode_bundle,
    load_protocol_card,
)


MAX_CALLS_PER_BATCH = 12
MAX_EXAMPLES = 12
MAX_MESSAGE_BYTES = 4096
MAX_TOKENS_PER_MESSAGE = 160
REQUEST_TIMEOUT_SECONDS = 45.0
CHECKPOINT_SCHEMA = "tlu.usage-example-generation-checkpoint.v1"
ARTIFACT_SCHEMA = "tlu.usage_examples.v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_training_meanings(
    *, bundle: dict[str, Any], split: dict[str, Any], task_key: bytes, count: int,
) -> list[dict[str, Any]]:
    """Select unique training tuples in a reproducible keyed order."""
    if not isinstance(task_key, bytes) or len(task_key) != 32:
        raise ValueError("task key must contain exactly 32 bytes")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_EXAMPLES:
        raise ValueError(f"example count must be in 1..{MAX_EXAMPLES} per batch")
    train_ids = set(split["train_meaning_ids"])
    values_to_id = {tuple(row["values"]): row["meaning_id"] for row in split["meanings"]}
    unique: dict[str, dict[str, str]] = {}
    for row in bundle["sender"]["train"]:
        meaning = row["private_meaning"]
        value_tuple = tuple(meaning[axis] for axis in split["attributes"])
        meaning_id = values_to_id.get(value_tuple)
        if meaning_id not in train_ids:
            raise ValueError("sender training ledger contains a non-training meaning")
        unique[meaning_id] = {axis: meaning[axis] for axis in split["attributes"]}
    if not unique or not set(unique).issubset(train_ids):
        raise ValueError("sender training ledger has no valid training support")
    if count > len(unique):
        raise ValueError(f"requested {count} examples but the sender training ledger contains only {len(unique)} unique training meanings")
    ranked = sorted(
        unique,
        key=lambda meaning_id: hmac.new(
            task_key,
            f"tlu.usage-example-generation.v1/{split['split_sha256']}/{meaning_id}".encode("ascii"),
            hashlib.sha256,
        ).digest(),
    )
    return [
        {"meaning_id": meaning_id, "meaning": unique[meaning_id]}
        for meaning_id in ranked[:count]
    ]


def _request_messages(card: dict[str, str], meaning: dict[str, str]) -> list[dict[str, str]]:
    system = (
        "You are generating one public training demonstration for a protocol onboarding study. "
        "Apply the sender instruction exactly to the single private meaning below. Do not invent values, "
        "use a candidate table, or reveal any identifiers. Return only a JSON object with exactly one "
        "string field named message. The value must be only the protocol message payload.\n\n"
        "Sender instruction:\n" + card["sender_instruction"]
    )
    user = json.dumps({"private_meaning": meaning}, ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_message(text: str) -> str:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"sender demonstration repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("sender demonstration was not a JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"message"}:
        raise ValueError("sender demonstration must contain exactly one message field")
    message = value["message"]
    if not isinstance(message, str) or not message.strip():
        raise ValueError("sender demonstration message must be non-empty text")
    if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("sender demonstration exceeds the 4096-byte artifact limit")
    return message


def _config_signature(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["results_sha256"] = _config_signature({"results": checkpoint["results"]})
    _atomic_json(path, checkpoint)


def generate_usage_examples(
    *, input_dir: Path, split_seed: int, task_key_path: Path, protocol_card_path: Path,
    example_count: int, output_dir: Path, model: str, tokenizer_id: str,
    base_url: str, execute: bool, resource_preflight: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    input_dir = _inside_project(input_dir)
    key_path = _inside_project(task_key_path)
    card_path = _inside_project(protocol_card_path)
    output_dir = _inside_project(output_dir)
    bundle, split = load_episode_bundle(input_dir, split_seed=split_seed)
    task_key = key_path.read_bytes()
    card, card_sha256 = load_protocol_card(card_path)
    examples = select_training_meanings(
        bundle=bundle, split=split, task_key=task_key, count=example_count,
    )
    input_manifest_sha256 = _digest_file(input_dir / "manifest.json")
    config = {
        "input_episode_manifest_sha256": input_manifest_sha256,
        "split_sha256": split["split_sha256"],
        "protocol_card_sha256": card_sha256,
        "protocol_id": card["protocol_id"],
        "selected_meaning_ids": [row["meaning_id"] for row in examples],
        "model": model,
        "tokenizer_id": tokenizer_id,
        "base_url": base_url,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS_PER_MESSAGE,
    }
    signature = _config_signature(config)
    checkpoint_path = output_dir / "generation.checkpoint.json"
    artifact_path = output_dir / "usage-examples.json"
    manifest_path = output_dir / "generation-manifest.json"
    trace_path = output_dir / "generation-trace.jsonl"
    plan = {
        "schema": ARTIFACT_SCHEMA,
        "mode": "execute" if execute else "dry_run",
        "split_seed": split_seed,
        "split_sha256": split["split_sha256"],
        "input_episode_manifest_sha256": input_manifest_sha256,
        "protocol_id": card["protocol_id"],
        "protocol_card_sha256": card_sha256,
        "training_example_count": len(examples),
        "available_unique_training_meanings": len({
            tuple(row["private_meaning"][axis] for axis in split["attributes"])
            for row in bundle["sender"]["train"]
        }),
        "selected_meaning_ids_sha256": hashlib.sha256(
            "\n".join(row["meaning_id"] for row in examples).encode("ascii")
        ).hexdigest(),
        "planned_model_calls": len(examples),
        "maximum_model_calls_per_batch": MAX_CALLS_PER_BATCH,
        "model_loaded": False,
        "inference_started": False,
        "usage_artifact_path": artifact_path.relative_to(ROOT).as_posix(),
    }
    if len(examples) > MAX_CALLS_PER_BATCH:
        raise ValueError("planned requests exceed the per-batch model-call hard cap")
    if not execute:
        return plan
    if not model.strip() or not tokenizer_id.strip():
        raise ValueError("execution requires model and tokenizer identifiers")
    if resource_preflight is None:
        raise ValueError("execution requires a fresh passing resource preflight")
    preflight_path = _inside_project(resource_preflight)
    endpoint = _loopback_url(base_url)
    validate_resource_preflight(preflight_path, required_ports={_endpoint_port(endpoint)})
    if manifest_path.exists() or (
        not resume and (artifact_path.exists() or trace_path.exists())
    ):
        raise FileExistsError("refusing to overwrite a completed usage-example artifact")

    if resume:
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("no valid usage-example generation checkpoint exists") from exc
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("config_signature") != signature
            or checkpoint.get("config") != config
            or not isinstance(checkpoint.get("results"), list)
            or checkpoint.get("results_sha256") != _config_signature({"results": checkpoint.get("results")})
            or not isinstance(checkpoint.get("resource_preflight_sha256s"), list)
            or any(
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in checkpoint.get("resource_preflight_sha256s", [])
            )
            or len(checkpoint["results"]) > len(examples)
        ):
            raise ValueError("usage-example checkpoint does not match this exact generation batch")
        results = checkpoint["results"]
        for index, row in enumerate(results):
            expected_fields = {
                "meaning_id", "meaning", "message", "raw_completion", "model_reported",
                "input_tokens", "output_tokens", "service_seconds", "wall_seconds",
                "prompt_sha256", "completion_sha256",
            }
            if (
                not isinstance(row, dict) or set(row) != expected_fields
                or row.get("meaning_id") != examples[index]["meaning_id"]
                or row.get("meaning") != examples[index]["meaning"]
            ):
                raise ValueError("usage-example checkpoint is not an exact prefix of the selected train rows")
            expected_messages = _request_messages(card, examples[index]["meaning"])
            expected_prompt_sha256 = hashlib.sha256(
                json.dumps(expected_messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if (
                not isinstance(row["raw_completion"], str)
                or hashlib.sha256(row["raw_completion"].encode("utf-8")).hexdigest() != row["completion_sha256"]
                or row["prompt_sha256"] != expected_prompt_sha256
                or _parse_message(row["raw_completion"]) != row["message"]
            ):
                raise ValueError("usage-example checkpoint prompt/completion trace failed integrity checks")
        preflight_sha256 = _digest_file(preflight_path)
        preflight_hashes = list(checkpoint.get("resource_preflight_sha256s", []))
        if preflight_sha256 not in preflight_hashes:
            preflight_hashes.append(preflight_sha256)
    else:
        if checkpoint_path.exists():
            raise FileExistsError("a generation checkpoint exists; pass --resume or archive the run")
        results = []
        preflight_hashes = [_digest_file(preflight_path)]
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "config_signature": signature,
            "config": config,
            "results": results,
            "resource_preflight_sha256s": preflight_hashes,
        }
        _write_checkpoint(checkpoint_path, checkpoint)

    client = OpenAICompatibleClient(
        endpoint, model, timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_tokens=MAX_TOKENS_PER_MESSAGE, follow_redirects=False, temperature=0.0,
    )
    preflight_sha256 = _digest_file(preflight_path)
    all_started = time.perf_counter()
    for index in range(len(results), len(examples)):
        example = examples[index]
        messages = _request_messages(card, example["meaning"])
        prompt_bytes = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        started = time.perf_counter()
        completion = client.complete(messages)
        call_wall_seconds = time.perf_counter() - started
        try:
            message = _parse_message(completion.text)
        except ValueError as exc:
            failure_path = output_dir / f"failed-completion-{index + 1:02d}.txt"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(completion.text, encoding="utf-8")
            raise ValueError(f"training example {index + 1} returned an invalid message; raw completion saved to {failure_path}") from exc
        results.append({
            "meaning_id": example["meaning_id"],
            "meaning": example["meaning"],
            "message": message,
            "raw_completion": completion.text,
            "model_reported": completion.model or model,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "service_seconds": completion.service_seconds,
            "wall_seconds": call_wall_seconds,
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "completion_sha256": hashlib.sha256(completion.text.encode("utf-8")).hexdigest(),
        })
        checkpoint["results"] = results
        checkpoint["resource_preflight_sha256s"] = preflight_hashes
        _write_checkpoint(checkpoint_path, checkpoint)

    total_wall_seconds = time.perf_counter() - all_started
    input_token_values = [row["input_tokens"] for row in results]
    output_token_values = [row["output_tokens"] for row in results]
    service_values = [row["service_seconds"] for row in results]
    model_names = {row["model_reported"] for row in results}
    if len(model_names) != 1:
        raise ValueError("sender endpoint returned inconsistent model identifiers within one usage trace")
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "protocol_id": card["protocol_id"],
        "protocol_card_sha256": card_sha256,
        "training_split_sha256": split["split_sha256"],
        "training_episode_manifest_sha256": input_manifest_sha256,
        "acquisition": {
            "method": "model_generated",
            "model_id": next(iter(model_names)),
            "tokenizer_id": tokenizer_id,
            "generation_calls": len(results),
            "input_tokens": sum(input_token_values) if all(value is not None for value in input_token_values) else None,
            "output_tokens": sum(output_token_values) if all(value is not None for value in output_token_values) else None,
            "service_seconds": sum(service_values) if all(value is not None for value in service_values) else None,
            "wall_seconds": total_wall_seconds,
        },
        "examples": [
            {"meaning_id": row["meaning_id"], "meaning": row["meaning"], "message": row["message"]}
            for row in results
        ],
    }
    _atomic_json(artifact_path, artifact)
    trace_bytes = "".join(
        json.dumps({
            key: row[key] for key in (
                "meaning_id", "meaning", "raw_completion", "prompt_sha256",
                "completion_sha256", "input_tokens", "output_tokens",
                "service_seconds", "wall_seconds",
            )
        }, ensure_ascii=False, sort_keys=True) + "\n"
        for row in results
    ).encode("utf-8")
    trace_temp = trace_path.with_suffix(trace_path.suffix + ".tmp")
    trace_temp.write_bytes(trace_bytes)
    trace_temp.replace(trace_path)
    generation_manifest = {
        **plan,
        "mode": "executed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_signature": signature,
        "resource_preflight_sha256s": preflight_hashes,
        "model_loaded": True,
        "inference_started": True,
        "model_calls": len(results),
        "input_tokens": artifact["acquisition"]["input_tokens"],
        "output_tokens": artifact["acquisition"]["output_tokens"],
        "service_seconds": artifact["acquisition"]["service_seconds"],
        "wall_seconds": total_wall_seconds,
        "usage_artifact_sha256": _digest_file(artifact_path),
        "generation_trace_path": trace_path.relative_to(ROOT).as_posix(),
        "generation_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "example_traces": [
            {key: row[key] for key in (
                "meaning_id", "prompt_sha256", "completion_sha256", "input_tokens",
                "output_tokens", "service_seconds", "wall_seconds",
            )}
            for row in results
        ],
    }
    _atomic_json(manifest_path, generation_manifest)
    checkpoint_path.unlink(missing_ok=True)
    return generation_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--task-key", type=Path, required=True)
    parser.add_argument("--protocol-card", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("TLU_USAGE_SENDER_MODEL", ""))
    parser.add_argument("--tokenizer-id", default=os.environ.get("TLU_USAGE_SENDER_TOKENIZER_ID", ""))
    parser.add_argument("--base-url", default=os.environ.get("TLU_USAGE_SENDER_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--resource-preflight", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = generate_usage_examples(
            input_dir=args.input_dir, split_seed=args.split_seed, task_key_path=args.task_key,
            protocol_card_path=args.protocol_card, example_count=args.examples,
            output_dir=args.output_dir, model=args.model, tokenizer_id=args.tokenizer_id,
            base_url=args.base_url, execute=args.execute,
            resource_preflight=args.resource_preflight, resume=args.resume,
        )
    except (OSError, ValueError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
