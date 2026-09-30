"""Run a call-capped batch from a frozen IID bundle through the Tacit SDK."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
from urllib.parse import urlsplit
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.multiparty_private_sum import (
    MAX_REQUESTS_PER_EPISODE,
    MAX_WIRE_BUDGET_BYTES,
    MESSAGE_FORMATS,
    _client,
    run_sum_episode,
)
from experiments.emergent_ood_v0_3.runner import validate_resource_preflight
from experiments.multiparty_sum_v0_1.generate_tasks import ROOT, load_episodes


CONDITIONS = ("communicate", "no_message", "full_information")


def run_bundle_batch(
    bundle: Path,
    *,
    agent_count: int,
    episode_indices: tuple[int, ...],
    sender_clients: list[Any],
    receiver_client: Any,
    condition: str = "communicate",
    message_format: str = "decimal",
    wire_budget_bytes: int = MAX_WIRE_BUDGET_BYTES,
    request_cap: int = MAX_REQUESTS_PER_EPISODE,
) -> dict[str, Any]:
    """Run identical frozen episodes and enforce the batch request cap pre-dispatch."""
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("episode_indices must be a non-empty sequence without duplicates")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {', '.join(CONDITIONS)}")
    if message_format not in MESSAGE_FORMATS:
        raise ValueError(f"message_format must be one of {', '.join(MESSAGE_FORMATS)}")
    if not isinstance(sender_clients, list) or len(sender_clients) != agent_count:
        raise ValueError("sender_clients must contain one client per sender")
    if isinstance(request_cap, bool) or not isinstance(request_cap, int) or not 1 <= request_cap <= MAX_REQUESTS_PER_EPISODE:
        raise ValueError(f"request_cap must be between 1 and {MAX_REQUESTS_PER_EPISODE}")
    if isinstance(wire_budget_bytes, bool) or not isinstance(wire_budget_bytes, int) or not 0 <= wire_budget_bytes <= MAX_WIRE_BUDGET_BYTES:
        raise ValueError(f"wire_budget_bytes must be between zero and {MAX_WIRE_BUDGET_BYTES}")

    per_episode = agent_count + 1 if condition == "communicate" else 1
    planned_calls = len(episode_indices) * per_episode
    if planned_calls > request_cap:
        raise ValueError(f"batch requires {planned_calls} model calls, exceeding request cap {request_cap}")
    # Load and validate every row before the first potentially external request.
    tasks = load_episodes(bundle, agent_count, episode_indices)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for task in tasks:
        receiver = task["receiver_view"]
        if receiver != {
            "schema_version": manifest["schema_version"],
            "sender_count": agent_count,
            "role": "receiver",
            "prior": "independent_uniform_integer_0_to_3",
        }:
            raise ValueError("receiver view differs from the frozen public task schema")
        result = run_sum_episode(
            task["sender_values"], sender_clients=sender_clients,
            receiver_client=receiver_client, condition=condition,
            message_format=message_format, wire_budget_bytes=wire_budget_bytes,
            request_cap=request_cap,
        )
        if result["expected_sum"] != task["expected_sum"]:
            raise ValueError("runtime and scorer gold sums disagree")
        results.append({"episode_id": task["episode_id"], **result})
    return {
        "schema_version": "tlu.multiparty-sum-run.v0.1.0",
        "bundle_schema_version": manifest["schema_version"],
        "task_seed": manifest["task_seed"],
        "task_key_id": manifest["task_key_id"],
        "bundle_manifest_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
        "agent_count": agent_count,
        "condition": condition,
        "message_format": message_format if condition == "communicate" else None,
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
    parser.add_argument("--condition", choices=CONDITIONS, default="communicate")
    parser.add_argument("--message-format", choices=MESSAGE_FORMATS, default="decimal")
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
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    sender_model = os.environ.get("TLU_SUM_SENDER_MODEL", "local-sender")
    receiver_model = os.environ.get("TLU_SUM_RECEIVER_MODEL", "local-receiver")
    senders = [
        _client(sender_url, sender_model, os.environ.get("TLU_SUM_SENDER_API_KEY"))
        for _ in range(args.agent_count)
    ]
    receiver = _client(receiver_url, receiver_model, os.environ.get("TLU_SUM_RECEIVER_API_KEY"))
    started_at_utc = datetime.now(timezone.utc).isoformat()
    try:
        report = run_bundle_batch(
            bundle, agent_count=args.agent_count, episode_indices=tuple(args.episode_indices),
            sender_clients=senders, receiver_client=receiver, condition=args.condition,
            message_format=args.message_format, wire_budget_bytes=args.wire_budget_bytes,
            request_cap=args.request_cap,
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
