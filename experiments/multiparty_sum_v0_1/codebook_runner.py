"""Run a call-capped frozen bundle with an exhaustive codebook card."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.multiparty_private_sum import MAX_REQUESTS_PER_EPISODE, MAX_WIRE_BUDGET_BYTES, _client
from experiments.emergent_ood_v0_3.runner import validate_resource_preflight
from experiments.multiparty_sum_v0_1.codebook_protocol import build_protocol_card, run_codebook_episode
from experiments.multiparty_sum_v0_1.generate_tasks import ROOT, load_episodes


def run_codebook_bundle_batch(
    bundle: Path,
    *,
    agent_count: int,
    episode_indices: tuple[int, ...],
    payload_budget_bits: int,
    sender_clients: dict[str, Any],
    receiver_client: Any,
    wire_budget_bytes: int = MAX_WIRE_BUDGET_BYTES,
    request_cap: int = MAX_REQUESTS_PER_EPISODE,
) -> dict[str, Any]:
    """Validate the whole selected batch and total call budget before dispatch."""
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("episode_indices must be a non-empty sequence without duplicates")
    card = build_protocol_card(agent_count, payload_budget_bits)
    if set(sender_clients) != set(card["active_senders"]):
        raise ValueError("sender_clients must contain exactly one client for every active codebook sender")
    if isinstance(request_cap, bool) or not isinstance(request_cap, int) or not 1 <= request_cap <= MAX_REQUESTS_PER_EPISODE:
        raise ValueError(f"request_cap must be between 1 and {MAX_REQUESTS_PER_EPISODE}")
    if isinstance(wire_budget_bytes, bool) or not isinstance(wire_budget_bytes, int) or not 0 <= wire_budget_bytes <= MAX_WIRE_BUDGET_BYTES:
        raise ValueError(f"wire_budget_bytes must be between zero and {MAX_WIRE_BUDGET_BYTES}")
    calls_per_episode = len(card["active_senders"]) + 1
    planned_calls = len(episode_indices) * calls_per_episode
    if planned_calls > request_cap:
        raise ValueError(f"batch requires {planned_calls} model calls, exceeding request cap {request_cap}")

    tasks = load_episodes(bundle, agent_count, episode_indices)
    manifest_path = bundle / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    for task in tasks:
        expected_view = {
            "schema_version": manifest["schema_version"],
            "sender_count": agent_count,
            "role": "receiver",
            "prior": "independent_uniform_integer_0_to_3",
        }
        if task["receiver_view"] != expected_view:
            raise ValueError("receiver view differs from frozen public task schema")
        if sum(task["sender_values"]) != task["expected_sum"]:
            raise ValueError("runtime and scorer gold sums disagree")

    results = []
    for task in tasks:
        result = run_codebook_episode(
            task["sender_values"], card_spec=card,
            sender_clients=sender_clients, receiver_client=receiver_client,
            wire_budget_bytes=wire_budget_bytes, request_cap=request_cap,
        )
        if result["expected_sum"] != task["expected_sum"]:
            raise RuntimeError("runtime changed a prevalidated gold sum")
        results.append({"episode_id": task["episode_id"], **result})
    return {
        "schema_version": "tlu.multiparty-sum-codebook-run.v0.1.0",
        "bundle_schema_version": manifest["schema_version"],
        "task_seed": manifest["task_seed"],
        "task_key_id": manifest["task_key_id"],
        "bundle_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "codebook_card": card,
        "agent_count": agent_count,
        "payload_budget_bits_at_most": payload_budget_bits,
        "episode_indices": list(episode_indices),
        "planned_model_calls": planned_calls,
        "actual_model_calls": sum(int(row["model_calls"]) for row in results),
        "request_cap": request_cap,
        "wire_budget_bytes_per_episode": wire_budget_bytes,
        "episodes": results,
    }


def _loopback_port(endpoint: str, name: str, parser: argparse.ArgumentParser) -> int:
    try:
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            parser.error(f"{name} must be an HTTP loopback URL")
        return parsed.port or 80
    except ValueError as exc:
        parser.error(f"invalid {name}: {exc}")
    raise AssertionError("argparse.error exits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--agent-count", type=int, required=True)
    parser.add_argument("--episode-indices", type=int, nargs="+", required=True)
    parser.add_argument("--payload-budget-bits", type=int, required=True)
    parser.add_argument("--wire-budget-bytes", type=int, default=MAX_WIRE_BUDGET_BYTES)
    parser.add_argument("--request-cap", type=int, default=MAX_REQUESTS_PER_EPISODE)
    parser.add_argument("--resource-preflight", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    try:
        bundle.relative_to(ROOT)
    except ValueError:
        parser.error("bundle path must stay inside the project directory")
    sender_url = os.environ.get("TLU_SUM_SENDER_URL", "http://127.0.0.1:8000/v1")
    receiver_url = os.environ.get("TLU_SUM_RECEIVER_URL", "http://127.0.0.1:8001/v1")
    ports = {
        _loopback_port(sender_url, "sender endpoint", parser),
        _loopback_port(receiver_url, "receiver endpoint", parser),
        8000, 8001, 8002,
    }
    try:
        validate_resource_preflight(args.resource_preflight, required_ports=ports)
        card = build_protocol_card(args.agent_count, args.payload_budget_bits)
        planned_calls = len(args.episode_indices) * (len(card["active_senders"]) + 1)
        if planned_calls > args.request_cap:
            raise ValueError(f"batch requires {planned_calls} model calls, exceeding request cap {args.request_cap}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    sender_model = os.environ.get("TLU_SUM_SENDER_MODEL", "local-sender")
    receiver_model = os.environ.get("TLU_SUM_RECEIVER_MODEL", "local-receiver")
    sender_clients = {
        role: _client(sender_url, sender_model, os.environ.get("TLU_SUM_SENDER_API_KEY"))
        for role in card["active_senders"]
    }
    receiver_client = _client(receiver_url, receiver_model, os.environ.get("TLU_SUM_RECEIVER_API_KEY"))
    started_at_utc = datetime.now(timezone.utc).isoformat()
    try:
        report = run_codebook_bundle_batch(
            bundle, agent_count=args.agent_count,
            episode_indices=tuple(args.episode_indices),
            payload_budget_bits=args.payload_budget_bits,
            sender_clients=sender_clients, receiver_client=receiver_client,
            wire_budget_bytes=args.wire_budget_bytes, request_cap=args.request_cap,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report["execution"] = {
        "started_at_utc": started_at_utc,
        "python_version": platform.python_version(),
        "sender_model": sender_model,
        "receiver_model": receiver_model,
        "sender_endpoint": sender_url,
        "receiver_endpoint": receiver_url,
        "resource_preflight_sha256": hashlib.sha256(args.resource_preflight.read_bytes()).hexdigest(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
