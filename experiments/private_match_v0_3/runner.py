"""Dry-run-first, gated runner for Private Match v0.3.

No endpoint is contacted unless --execute is supplied and the local resource
preflight plus the disjoint full-information capability ledger pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

from experiments.emergent_ood_v0_3.runner import _endpoint_port, validate_resource_preflight
from experiments.private_match_v0_3.generate_tasks import generate_episode, score_answer
from experiments.private_match_v0_3.protocols import PROTOCOL_IDS, Protocol, parse_coordinate_message, protocol_by_id
from tacit import ChatCompletion, OpenAICompatibleClient, exchange_dialogue


EXPERIMENT_ID = "private-match-v0.3"
SCORER_ID = "private-match-exact-candidate-id-v1"
MAX_MODEL_CALLS_PER_BATCH = 12
REQUEST_TIMEOUT_SECONDS = 30.0
CONDITIONS = ("full_information", "no_message", "sender_x_only", "sender_y_only", "both_sources")


class _DialogueProtocol:
    def __init__(self, protocol: Protocol, names: set[str]) -> None:
        self.protocol_id = protocol.protocol_id
        all_instructions = {
            "sender_x": protocol.sender_instruction,
            "sender_y": protocol.sender_instruction,
            "receiver": protocol.receiver_instruction,
        }
        self.agent_instructions = {name: all_instructions[name] for name in names}


class _NamedClient:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.model_name = client.model

    def complete(self, messages: Any) -> ChatCompletion:
        return self.client.complete(messages)


def episode_id_for_seed(seed: int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return f"pmt3-{seed:012d}"


def planned_model_calls(episodes: int, condition: str) -> int:
    if episodes < 1 or condition not in CONDITIONS:
        raise ValueError("episodes must be positive and condition must be registered")
    return episodes * {"full_information": 1, "no_message": 1,
                       "sender_x_only": 2, "sender_y_only": 2, "both_sources": 3}[condition]


def _private_contexts(episode: tuple[dict[str, Any], ...], condition: str) -> dict[str, str]:
    sender_x, sender_y, receiver, _gold = episode
    receiver_context = dict(receiver)
    if condition == "full_information":
        receiver_context["calibration_coordinates"] = {
            "x": sender_x["private_value"], "y": sender_y["private_value"]
        }
    return {
        "sender_x": json.dumps(sender_x, sort_keys=True),
        "sender_y": json.dumps(sender_y, sort_keys=True),
        "receiver": json.dumps(receiver_context, sort_keys=True),
    }


def run_condition(*, seed: int, q: int, condition: str, protocol: Protocol,
                  sender_model: Any | None, receiver_model: Any,
                  sender_tokenizer_id: str | None, receiver_tokenizer_id: str,
                  model_population_id: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    episode_id = episode_id_for_seed(seed)
    episode = generate_episode(episode_id=episode_id, seed=seed, q=q)
    sender_x, sender_y, receiver, gold = episode
    contexts = _private_contexts(episode, condition)
    agents = {"sender_x": sender_model, "sender_y": sender_model, "receiver": receiver_model}
    if sender_model is None:
        agents.pop("sender_x")
        agents.pop("sender_y")
    contexts = {name: contexts[name] for name in agents}
    if condition in {"full_information", "no_message"}:
        schedule: tuple[tuple[str, str], ...] = ()
    elif condition == "sender_x_only":
        schedule = (("sender_x", "receiver"),)
    elif condition == "sender_y_only":
        schedule = (("sender_y", "receiver"),)
    else:
        schedule = (("sender_x", "receiver"), ("sender_y", "receiver"))
    if condition not in {"full_information", "no_message"} and sender_model is None:
        raise ValueError("sender model required for message conditions")
    final_instruction = (
        "For full-information calibration, use calibration_coordinates to identify the exact candidate. "
        "For all other conditions, use only your candidate table and delivered messages. Return only candidate_id."
    )
    started = time.perf_counter()
    dialogue = exchange_dialogue(
        agents, protocol=_DialogueProtocol(protocol, set(agents)), private_contexts=contexts,
        schedule=schedule, task="Identify the candidate row matching the hidden target.",
        max_turns=len(schedule), final_answer_agent="receiver",
        final_answer_instruction=final_instruction,
    )
    answer_raw = dialogue.final_submission.text if dialogue.final_submission else ""
    answer = answer_raw.strip()
    success = score_answer(receiver, gold, answer)
    formats: list[dict[str, Any]] = []
    for turn in dialogue.turns:
        formats.append({
            "sender": turn.speaker,
            "message": turn.completion.text,
            "format_valid": parse_coordinate_message(
                protocol.protocol_id.split(":", 1)[0], turn.completion.text,
                q=q, sender=turn.speaker,
            )[0],
            "semantic_fidelity": (
                None if protocol.protocol_id.startswith("concise_nl:") else parse_coordinate_message(
                    protocol.protocol_id.split(":", 1)[0], turn.completion.text,
                    q=q, sender=turn.speaker,
                )[2] == (sender_x if turn.speaker == "sender_x" else sender_y)["private_value"]
            ),
        })
    tokenizers = {"receiver": receiver_tokenizer_id}
    if sender_tokenizer_id:
        for speaker in {turn.speaker for turn in dialogue.turns}:
            tokenizers[speaker] = sender_tokenizer_id
    calls = dialogue.model_call_records(tokenizers=tokenizers)
    # Stage names preserve role identity for cost aggregation.
    for call in calls:
        if call["stage"] == "dialogue_turn":
            call["stage"] = call["agent"]
    return {
        "schema_version": "tlu.costs.v3",
        "episode_id": episode_id,
        "stratum": {
            "experiment_id": EXPERIMENT_ID,
            "task_id": "triadic-complementary-coordinate-match-v1",
            "split": "calibration" if condition == "full_information" else "evaluation",
            "task_parameters": {"q": q, "candidate_count": q * q, "agent_count": 3},
            "model_population_id": model_population_id,
            "agent_models": {
                "sender_x": getattr(sender_model, "model_name", "absent"),
                "sender_y": getattr(sender_model, "model_name", "absent"),
                "receiver": getattr(receiver_model, "model_name", "unknown"),
            },
            "scorer_id": SCORER_ID,
        },
        "protocol": {
            "policy_id": (condition if condition in {"no_message", "full_information"}
                          else f"{condition}:{protocol.protocol_id.split(':', 1)[0]}"),
            "code_id": "none" if condition in {"no_message", "full_information"} else protocol.code_id,
            "decoder_id": protocol.decoder_id,
        },
        "outcome": {"joint_success": success, "answer_score": float(success)},
        "diagnostics": {
            "condition": condition, "generation_seed": seed,
            "answer_text": answer, "answer_text_verbatim": answer_raw,
            "answer_candidate_id": answer if answer in {r["candidate_id"] for r in receiver["candidates"]} else None,
            "answer_format_valid": answer_raw == answer and answer in {r["candidate_id"] for r in receiver["candidates"]},
            "candidate_ids_in_receiver_order": [r["candidate_id"] for r in receiver["candidates"]],
            "message_diagnostics": formats,
            "message_texts": [turn.completion.text for turn in dialogue.turns],
            "stop_reason": dialogue.stop_reason,
            "final_answer_stage": "sealed_final_answer",
        },
        "transmissions": dialogue.transmission_records(),
        "model_calls": calls,
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "setup": [],
    }


def validate_capability_ledger(path: Path | None, *, episodes: int, seed: int, q: int,
                               receiver_model: str, receiver_tokenizer_id: str,
                               model_population_id: str) -> None:
    if path is None:
        raise ValueError("evaluation requires --capability-ledger from a perfect disjoint full_information batch")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read capability ledger: {path}") from exc
    if len(rows) != episodes:
        raise ValueError("capability ledger must contain exactly the requested calibration episode count")
    calibration = {row.get("episode_id"): row for row in rows if isinstance(row, dict)}
    if len(calibration) != episodes:
        raise ValueError("capability ledger episode IDs must be unique")
    eval_ids = {episode_id_for_seed(s) for s in range(seed, seed + episodes)}
    if eval_ids.intersection(calibration):
        raise ValueError("capability calibration episodes must be disjoint from evaluation episodes")
    for row in rows:
        diag = row.get("diagnostics", {})
        stratum = row.get("stratum", {})
        calls = row.get("model_calls", [])
        episode_seed = diag.get("generation_seed")
        if row.get("episode_id") != episode_id_for_seed(episode_seed):
            raise ValueError("calibration episode ID does not match generation seed")
        if (row.get("protocol", {}).get("policy_id") != "full_information"
                or diag.get("condition") != "full_information"
                or row.get("outcome", {}).get("joint_success") is not True
                or row.get("transmissions") != [] or len(calls) != 1):
            raise ValueError("every calibration row must be a successful full_information receiver-only call")
        if stratum.get("experiment_id") != EXPERIMENT_ID or stratum.get("scorer_id") != SCORER_ID:
            raise ValueError("capability ledger experiment/scorer differs")
        if stratum.get("task_parameters") != {"q": q, "candidate_count": q * q, "agent_count": 3}:
            raise ValueError("capability ledger task parameters differ")
        if stratum.get("model_population_id") != model_population_id:
            raise ValueError("capability ledger model population differs")
        if stratum.get("agent_models", {}).get("receiver") != receiver_model:
            raise ValueError("capability ledger receiver model differs")
        if calls[0].get("model") != receiver_model or calls[0].get("tokenizer") != receiver_tokenizer_id:
            raise ValueError("capability ledger receiver model/tokenizer differs")
        if calls[0].get("stage") != "final_answer" or calls[0].get("truncated") is not False:
            raise ValueError("calibration answer missing, truncated, or at the wrong stage")
        episode = generate_episode(episode_id=row["episode_id"], seed=episode_seed, q=q)
        if diag.get("answer_text_verbatim") != episode[3]["target_candidate_id"]:
            raise ValueError("calibration answer does not match generated gold")


def _loopback_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("only a loopback endpoint URL is allowed")
    return value.rstrip("/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="contact the configured loopback model endpoint")
    parser.add_argument("--condition", choices=CONDITIONS, default="full_information")
    parser.add_argument("--protocol", choices=PROTOCOL_IDS, default="compact_kv")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=303000)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(".cache/private_match_v0_3/full_information.jsonl"))
    parser.add_argument("--capability-ledger", type=Path)
    parser.add_argument("--resource-preflight", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sender-model", default=os.environ.get("TLU_SENDER_MODEL", ""))
    parser.add_argument("--receiver-model", default=os.environ.get("TLU_RECEIVER_MODEL", ""))
    parser.add_argument("--sender-tokenizer-id", default=os.environ.get("TLU_SENDER_TOKENIZER_ID", ""))
    parser.add_argument("--receiver-tokenizer-id", default=os.environ.get("TLU_RECEIVER_TOKENIZER_ID", ""))
    parser.add_argument("--model-population-id", default=os.environ.get("TLU_MODEL_POPULATION_ID", "local-unspecified"))
    parser.add_argument("--base-url", default=os.environ.get("TLU_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--sender-base-url", default=os.environ.get("TLU_SENDER_BASE_URL", ""))
    parser.add_argument("--receiver-base-url", default=os.environ.get("TLU_RECEIVER_BASE_URL", ""))
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed < 0 or args.q < 2 or args.q & (args.q - 1):
        parser.error("episodes and seed must be valid; q must be a power of two >= 2")
    calls = planned_model_calls(args.episodes, args.condition)
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "condition": args.condition,
                          "protocol": args.protocol, "planned_model_calls": calls,
                          "maximum_model_calls_per_batch": MAX_MODEL_CALLS_PER_BATCH,
                          "fits_call_cap": calls <= MAX_MODEL_CALLS_PER_BATCH,
                          "inference_started": False, "resource_preflight_required": True}, indent=2))
        return 0
    if calls > MAX_MODEL_CALLS_PER_BATCH:
        parser.error(f"batch plans {calls} calls, above cap {MAX_MODEL_CALLS_PER_BATCH}")
    if args.condition == "full_information" and args.capability_ledger:
        parser.error("full_information calibration is its own first-stage batch")
    if args.condition != "full_information":
        try:
            validate_capability_ledger(args.capability_ledger, episodes=args.episodes,
                seed=args.seed, q=args.q, receiver_model=args.receiver_model,
                receiver_tokenizer_id=args.receiver_tokenizer_id,
                model_population_id=args.model_population_id)
        except ValueError as exc:
            parser.error(str(exc))
    needs_sender = args.condition in {"sender_x_only", "sender_y_only", "both_sources"}
    if not args.receiver_model or not args.receiver_tokenizer_id or (needs_sender and not args.sender_model):
        parser.error("--execute requires receiver model/tokenizer and sender model for message conditions")
    if needs_sender and not args.sender_tokenizer_id:
        parser.error("message conditions require --sender-tokenizer-id")
    try:
        receiver_url = _loopback_url(args.receiver_base_url or args.base_url)
        sender_url = _loopback_url(args.sender_base_url or args.base_url) if needs_sender else None
        required_ports = {_endpoint_port(receiver_url)}
        if sender_url:
            required_ports.add(_endpoint_port(sender_url))
        validate_resource_preflight(args.resource_preflight, required_ports=required_ports)
    except ValueError as exc:
        parser.error(str(exc))
    target = args.output.resolve()
    if target.exists() and not args.force:
        parser.error(f"refusing to overwrite {target}; pass --force explicitly")
    target.parent.mkdir(parents=True, exist_ok=True)
    receiver = _NamedClient(OpenAICompatibleClient(receiver_url, args.receiver_model,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS, max_tokens=32, follow_redirects=False,
        temperature=0.0))
    sender = _NamedClient(OpenAICompatibleClient(sender_url, args.sender_model,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS, max_tokens=48, follow_redirects=False,
        temperature=0.0)) if needs_sender else None
    protocol = protocol_by_id(args.protocol, args.q)
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for episode_seed in range(args.seed, args.seed + args.episodes):
            row = run_condition(seed=episode_seed, q=args.q, condition=args.condition,
                protocol=protocol, sender_model=sender, receiver_model=receiver,
                sender_tokenizer_id=args.sender_tokenizer_id or None,
                receiver_tokenizer_id=args.receiver_tokenizer_id,
                model_population_id=args.model_population_id)
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
    print(json.dumps({"mode": "executed", "condition": args.condition,
                      "episodes": args.episodes, "model_calls": calls,
                      "output": str(target)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
