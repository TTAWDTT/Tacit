"""Synthesize candidate protocol cards from a sampled training-only ontology view.

The default is a local dry-run. Execution makes exactly one request to an
operator-started loopback endpoint after the frozen resource preflight passes.
This produces LLM-proposed protocol instructions, not evidence of an emergent
language or of protocol utility.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.emergent_ood_v0_3.runner import validate_resource_preflight
from experiments.emergent_ood_v0_4.episodes import _inside_project
from experiments.emergent_ood_v0_4.runner import ROOT, OpenAICompatibleClient, _endpoint_port, _loopback_url
from experiments.emergent_ood_v0_4.split import build_split


CARD_SCHEMA = "tlu.shared_protocol_card.v1"
OUTPUT_SCHEMA = "tlu.emergent-ood-induced-card-set.v1"
REQUEST_TIMEOUT_SECONDS = 45.0
DEFAULT_EXAMPLES = 32
DEFAULT_CANDIDATES = 4
MAX_CANDIDATES = 8


def sample_training_examples(
    *, split: dict[str, Any], task_key: bytes, example_count: int = DEFAULT_EXAMPLES,
) -> list[dict[str, str]]:
    """Select a reproducible secret-keyed sample from training meanings only."""
    if not isinstance(task_key, bytes) or len(task_key) != 32:
        raise ValueError("task key must be exactly 32 bytes")
    train_ids = set(split.get("train_meaning_ids", []))
    if isinstance(example_count, bool) or not isinstance(example_count, int) or not 1 <= example_count <= len(train_ids):
        raise ValueError(f"example_count must be in 1..{len(train_ids)}")
    rows_by_id = {row["meaning_id"]: row for row in split["meanings"]}
    if not train_ids.issubset(rows_by_id):
        raise ValueError("split train IDs are not present in its meaning table")
    ranked = sorted(
        train_ids,
        key=lambda meaning_id: hmac.new(
            task_key,
            f"tlu.emergent-ood-protocol-induction.v1/{split['split_sha256']}/{meaning_id}".encode("ascii"),
            hashlib.sha256,
        ).digest(),
    )
    attributes = split["attributes"]
    return [
        {attribute: value for attribute, value in zip(attributes, rows_by_id[meaning_id]["values"])}
        for meaning_id in ranked[:example_count]
    ]


def build_induction_messages(
    *, split: dict[str, Any], examples: list[dict[str, str]], candidate_count: int = DEFAULT_CANDIDATES,
) -> list[dict[str, str]]:
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or not 2 <= candidate_count <= MAX_CANDIDATES:
        raise ValueError(f"candidate_count must be in 2..{MAX_CANDIDATES}")
    allowed_ids = set(split["train_meaning_ids"])
    if not examples or len(examples) > len(allowed_ids):
        raise ValueError("training example set is empty or too large")
    vocabulary = {
        attribute: list(split["values_by_attribute"][attribute])
        for attribute in split["attributes"]
    }
    for example in examples:
        if set(example) != set(split["attributes"]):
            raise ValueError("induction examples must contain only complete training meanings")
        meaning_id = "m-" + hashlib.sha256(
            json.dumps([example[name] for name in split["attributes"]], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        if meaning_id not in allowed_ids:
            raise ValueError("induction examples may contain training meanings only")

    system = (
        "You are proposing candidate message protocols for two stateless LLM agents. "
        "This is offline protocol synthesis; do not claim any candidate has been tested. "
        "Return only one valid JSON object matching the requested output schema, with no markdown or commentary."
    )
    user_payload = {
        "task": (
            "A sender sees one private four-attribute meaning. A receiver sees a candidate table of meanings and one message "
            "from the sender, then must return the exact candidate_id whose complete tuple matches. The sender never sees the table. "
            "Future evaluation includes held-out compositions, so design productive rules rather than a lookup table of complete tuples."
        ),
        "shared_attribute_vocabulary": vocabulary,
        "sampled_training_meanings_only": examples,
        "design_constraints": [
            "Propose distinct, compact, unambiguous, compositional protocols; do not just restate ordinary English or valid JSON.",
            "Define stable meaning for every emitted primitive and a deterministic composition/decoding rule in both role instructions.",
            "Represent every attribute and exact value; use explicit boundaries/order so unrelated messages cannot merge ambiguously.",
            "The sender may use only its private meaning. The receiver may use only its candidate table and the received message.",
            "Do not encode candidate IDs, episode IDs, split/task seeds, or a table mapping complete four-value tuples to labels.",
            "Do not assume shared conversation history, hidden state, tools, extra turns, or a response from the sender.",
            "Each card must be independently usable by two separately prompted agents. Return only the message payload at runtime.",
        ],
        "required_output": {
            "schema": OUTPUT_SCHEMA,
            "protocols": [
                {"sender_instruction": "...", "receiver_instruction": "..."}
                for _ in range(candidate_count)
            ],
        },
    }
    user = (
        "Create exactly " + str(candidate_count) + " genuinely different candidate protocol cards. "
        "Do not reproduce a full-tuple lookup from the examples. Prefer reusable attribute/value primitives, clear composition, "
        "and a fixed decoding procedure; each receiver instruction must explain how to choose one candidate_id.\n\n"
        + json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_candidate_cards(response: str, *, candidate_count: int) -> list[dict[str, str]]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("inducer output is not valid JSON; raw completion is retained for audit") from exc
    if not isinstance(value, dict) or set(value) != {"schema", "protocols"} or value.get("schema") != OUTPUT_SCHEMA:
        raise ValueError("inducer output has an unexpected schema")
    protocols = value.get("protocols")
    if not isinstance(protocols, list) or len(protocols) != candidate_count:
        raise ValueError(f"inducer must return exactly {candidate_count} candidate protocols")
    cards: list[dict[str, str]] = []
    fingerprints: set[str] = set()
    for item in protocols:
        if not isinstance(item, dict) or set(item) != {"sender_instruction", "receiver_instruction"}:
            raise ValueError("each induced protocol must contain exactly two role instructions")
        sender, receiver = item["sender_instruction"], item["receiver_instruction"]
        if not isinstance(sender, str) or not sender.strip() or not isinstance(receiver, str) or not receiver.strip():
            raise ValueError("induced sender and receiver instructions must be non-empty strings")
        if any(len(text.encode("utf-8")) > 32768 for text in (sender, receiver)):
            raise ValueError("induced protocol instructions exceed the card size limit")
        canonical = json.dumps([sender, receiver], ensure_ascii=False, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if fingerprint in fingerprints:
            raise ValueError("inducer returned duplicate protocol cards")
        fingerprints.add(fingerprint)
        cards.append({
            "schema": CARD_SCHEMA,
            "protocol_id": f"induced-{fingerprint[:16]}",
            "sender_instruction": sender,
            "receiver_instruction": receiver,
        })
    return cards


def _read_inside(path: Path) -> bytes:
    return _inside_project(path).read_bytes()


def run_induction(
    *, split_seed: int, task_key_path: Path, example_count: int, candidate_count: int,
    output_dir: Path, execute: bool, model: str = "", tokenizer_id: str = "",
    base_url: str = "http://127.0.0.1:8002/v1", resource_preflight: Path | None = None,
    temperature: float = 0.7, max_tokens: int = 4096,
) -> dict[str, Any]:
    split = build_split(seed=split_seed)
    task_key = _read_inside(task_key_path)
    examples = sample_training_examples(split=split, task_key=task_key, example_count=example_count)
    messages = build_induction_messages(split=split, examples=examples, candidate_count=candidate_count)
    prompt_bytes = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    plan = {
        "schema": OUTPUT_SCHEMA,
        "mode": "execute" if execute else "dry_run",
        "split_seed": split_seed,
        "split_sha256": split["split_sha256"],
        "task_key_id": hashlib.sha256(task_key).hexdigest()[:16],
        "training_examples": len(examples),
        "training_example_ids_sha256": hashlib.sha256("\n".join(sorted(
            "m-" + hashlib.sha256(json.dumps([row[a] for a in split["attributes"]], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
            for row in examples
        )).encode("utf-8")).hexdigest(),
        "candidate_count": candidate_count,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt_utf8_bytes": len(prompt_bytes),
        "test_examples_in_prompt": 0,
        "model_loaded": False,
        "inference_started": False,
        "protocol_induction_cost": "unmeasured until execute mode completes",
    }
    if not execute:
        return plan
    if not model or not tokenizer_id:
        raise ValueError("execution requires model and tokenizer identifiers")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be in [0, 2]")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    endpoint = _loopback_url(base_url)
    if resource_preflight is None:
        raise ValueError("execution requires a recent passing resource preflight")
    preflight = _inside_project(resource_preflight)
    validate_resource_preflight(preflight, required_ports={_endpoint_port(endpoint)})
    target_dir = _inside_project(output_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        raise ValueError(f"refusing to overwrite a non-empty induction directory: {target_dir}")

    client = OpenAICompatibleClient(
        endpoint, model, api_key=os.environ.get("TLU_PROTOCOL_INDUCER_API_KEY"),
        timeout_seconds=REQUEST_TIMEOUT_SECONDS, max_tokens=max_tokens,
        follow_redirects=False, temperature=temperature,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = target_dir / "induction-prompt.json"
    prompt_path.write_bytes(prompt_bytes)
    started = time.perf_counter()
    try:
        completion = client.complete(messages)
    except Exception as exc:
        plan.update({
            "request_attempted": True,
            "request_error_type": type(exc).__name__,
            "wall_seconds": time.perf_counter() - started,
            "preflight_sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(),
        })
        (target_dir / "induction-failure-manifest.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    wall_seconds = time.perf_counter() - started
    completion_bytes = completion.text.encode("utf-8")
    completion_path = target_dir / "induction-completion.txt"
    completion_path.write_bytes(completion_bytes)
    try:
        cards = parse_candidate_cards(completion.text, candidate_count=candidate_count)
    except ValueError as exc:
        plan.update({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_attempted": True,
            "model_calls": 1,
            "inducer_model": completion.model or model,
            "tokenizer_id": tokenizer_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "service_seconds": completion.service_seconds,
            "wall_seconds": wall_seconds,
            "completion_sha256": hashlib.sha256(completion_bytes).hexdigest(),
            "prompt_path": prompt_path.relative_to(ROOT).as_posix(),
            "completion_path": completion_path.relative_to(ROOT).as_posix(),
            "prompt_utf8_bytes": len(prompt_bytes),
            "completion_utf8_bytes": len(completion_bytes),
            "preflight_sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(),
            "parse_error": str(exc),
            "protocol_induction_cost": "one measured generator model call produced an invalid card response",
        })
        (target_dir / "induction-failure-manifest.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    card_entries = []
    for index, card in enumerate(cards, start=1):
        card_path = target_dir / f"candidate-{index:02d}-{card['protocol_id']}.json"
        payload = (json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        card_path.write_bytes(payload)
        card_entries.append({
            "protocol_id": card["protocol_id"],
            "path": card_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    plan.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_calls": 1,
        "model_loaded": True,
        "inference_started": True,
        "inducer_model": completion.model or model,
        "tokenizer_id": tokenizer_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "service_seconds": completion.service_seconds,
        "wall_seconds": wall_seconds,
        "completion_sha256": hashlib.sha256(completion_bytes).hexdigest(),
        "prompt_path": prompt_path.relative_to(ROOT).as_posix(),
        "completion_path": completion_path.relative_to(ROOT).as_posix(),
        "prompt_utf8_bytes": len(prompt_bytes),
        "completion_utf8_bytes": len(completion_bytes),
        "preflight_sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(),
        "candidate_cards": card_entries,
        "protocol_induction_cost": "one measured generator model call; integrate these costs into the later selector manifest",
    })
    manifest_path = target_dir / "induction-manifest.json"
    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--task-key", type=Path, default=Path(".cache/emergent_ood_v0_4/evaluator.key"))
    parser.add_argument("--training-examples", type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/emergent_ood_v0_4/induced-cards"))
    parser.add_argument("--base-url", default=os.environ.get("TLU_PROTOCOL_INDUCER_URL", "http://127.0.0.1:8002/v1"))
    parser.add_argument("--model", default=os.environ.get("TLU_PROTOCOL_INDUCER_MODEL", ""))
    parser.add_argument("--tokenizer-id", default=os.environ.get("TLU_PROTOCOL_INDUCER_TOKENIZER_ID", ""))
    parser.add_argument("--resource-preflight", type=Path)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--execute", action="store_true", help="make one call to the configured loopback model endpoint")
    args = parser.parse_args()
    try:
        result = run_induction(
            split_seed=args.split_seed, task_key_path=args.task_key,
            example_count=args.training_examples, candidate_count=args.candidates,
            output_dir=args.output_dir, execute=args.execute, model=args.model,
            tokenizer_id=args.tokenizer_id, base_url=args.base_url,
            resource_preflight=args.resource_preflight, temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
