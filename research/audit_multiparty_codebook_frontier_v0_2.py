"""Audit the exact-runner tokenizer, call, and loopback-byte frontier."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.multiparty_private_sum import (  # noqa: E402
    RECEIVER, SUM_TASK, _role_instructions,
)
from experiments.multiparty_sum_v0_1.codebook_protocol import build_protocol_card  # noqa: E402
from research.multiparty_sum_lossy_frontier import exact_frontier  # noqa: E402
from tacit import LocalTCPMessageChannel  # noqa: E402


TOKENIZER_REPO = "Qwen/Qwen3-4B"
TOKENIZER_REVISION = "eb971e9fb1f41c13b5e5a56e56886305c5ad94a0"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_RELATIVE_PATH = Path(
    ".cache/tokenizers/hfhub/models--Qwen--Qwen3-4B/snapshots"
) / TOKENIZER_REVISION / "tokenizer.json"
FINAL_RUNNER_INSTRUCTION = "Return only one non-negative base-10 integer."
CODEBOOK_FINAL_INSTRUCTION = "Return only the exact non-negative integer sum as base-10 digits."
SCHEMA = "tlu.multiparty-sum-codebook-frontier-audit.v0.2.0"
CONDITIONS = ("no_message", "decimal", "json", "binary", "exhaustive_codebook")


def _count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def _user_tokens(tokenizer: Any, context: str, transcript: list[dict[str, str]]) -> int:
    payload = {
        "task": SUM_TASK,
        "private_context": context,
        "visible_transcript": transcript,
        "instruction": "Return the final task answer for external scoring.",
    }
    # exchange_dialogue uses ensure_ascii=False and json.dumps(sort_keys=True).
    return _count(tokenizer, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _width(partition: list[list[int]]) -> int:
    return (len(partition) - 1).bit_length()


def _codebook_message_ids(values: tuple[int, ...], partitions: list[list[list[int]]]) -> tuple[int, ...]:
    return tuple(
        next(index for index, block in enumerate(partition) if value in block)
        for value, partition in zip(values, partitions)
    )


def _cost_row(
    tokenizer: Any,
    *,
    system: dict[str, str],
    sender_values: tuple[int, ...],
    sender_messages: dict[str, str],
    receiver_context: str,
    answer: int,
    protocol_id: str,
) -> dict[str, int]:
    senders = list(sender_messages)
    sender_instruction_tokens = sum(_count(tokenizer, system[sender]) for sender in senders)
    sender_user_tokens = 0
    sender_output_tokens = 0
    app_bytes = 0
    channel = LocalTCPMessageChannel(lambda _envelope: None)
    for round_number, sender in enumerate(senders, start=1):
        index = int(sender[1:]) - 1
        context = (
            f"Public metadata: sender_count={len(sender_values)}. "
            f"Your private integer is {sender_values[index]}."
        )
        sender_user_tokens += _count(tokenizer, json.dumps({
            "task": SUM_TASK,
            "private_context": context,
            "visible_transcript": [],
            "instruction": "Send exactly the next message for this turn.",
        }, ensure_ascii=False, sort_keys=True))
        sender_output_tokens += _count(tokenizer, sender_messages[sender])
        app_bytes += channel.measure(
            sender_messages[sender], protocol_id=protocol_id,
            round_number=round_number, sender=sender, recipient=RECEIVER,
        ).total_application_bytes

    receiver_user_tokens = _user_tokens(
        tokenizer, receiver_context,
        [{"sender": sender, "message": message} for sender, message in sender_messages.items()],
    )
    receiver_system_tokens = _count(tokenizer, system[RECEIVER])
    receiver_output_tokens = _count(tokenizer, str(answer))
    input_tokens = sender_instruction_tokens + sender_user_tokens + receiver_system_tokens + receiver_user_tokens
    output_tokens = sender_output_tokens + receiver_output_tokens
    return {
        "sender_role_instruction_tokens": sender_instruction_tokens,
        "sender_user_content_tokens": sender_user_tokens,
        "receiver_role_instruction_tokens": receiver_system_tokens,
        "receiver_user_content_tokens": receiver_user_tokens,
        "sender_completion_tokens": sender_output_tokens,
        "receiver_completion_tokens": receiver_output_tokens,
        "model_input_content_tokens": input_tokens,
        "completion_content_tokens": output_tokens,
        "total_known_content_tokens": input_tokens + output_tokens,
        "delivered_application_bytes": app_bytes,
        "planned_model_calls": len(senders) + 1,
        "exact_sum_success": int(answer == sum(sender_values)),
    }


def _summary(rows: list[dict[str, int]]) -> dict[str, Any]:
    report: dict[str, Any] = {"episodes": len(rows)}
    for metric in rows[0]:
        values = [row[metric] for row in rows]
        report[metric] = {
            "mean": round(sum(values) / len(values), 6),
            "min": min(values),
            "max": max(values),
        }
    return report


def build_report(tokenizer_path: Path) -> dict[str, Any]:
    try:
        tokenizer_path = tokenizer_path.resolve()
        tokenizer_path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("tokenizer path must stay inside the project") from exc
    tokenizer_bytes = tokenizer_path.read_bytes()
    digest = hashlib.sha256(tokenizer_bytes).hexdigest()
    if digest != TOKENIZER_SHA256:
        raise ValueError("tokenizer JSON SHA-256 does not match the pinned Qwen3-4B revision")
    try:
        from tokenizers import Tokenizer, __version__ as tokenizers_version
    except ImportError as exc:
        raise RuntimeError("install the optional `tokenizers` package to run this audit") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    records = []
    for agent_count in range(2, 5):
        frontier = exact_frontier(agent_count)
        for budget, frontier_row in enumerate(frontier):
            try:
                card = build_protocol_card(agent_count, budget)
            except ValueError as exc:
                if str(exc) == "the zero-bit point has no codebook senders; use the no_message condition":
                    continue
                raise

            per_condition: dict[str, list[dict[str, int]]] = {condition: [] for condition in CONDITIONS}
            codebook_ids = {
                tuple(row["message_ids"]): row["decoded_sum"]
                for row in frontier_row["optimal_receiver_decoder"]
            }
            baseline_instructions: dict[str, dict[str, str]] = {}
            for condition in ("decimal", "json", "binary"):
                baseline_instructions[condition] = _role_instructions(
                    condition, [f"S{i + 1}" for i in range(agent_count)],
                )

            for values in itertools.product(range(4), repeat=agent_count):
                no_message_guess = (3 * agent_count) // 2
                per_condition["no_message"].append(_cost_row(
                    tokenizer, system={
                        RECEIVER: baseline_instructions["decimal"][RECEIVER]
                        + "\n\n" + FINAL_RUNNER_INSTRUCTION,
                    },
                    sender_values=values, sender_messages={},
                    receiver_context=f"Public metadata: sender_count={agent_count}.",
                    answer=no_message_guess, protocol_id="private-sum-decimal-v0",
                ))

                for condition in ("decimal", "json", "binary"):
                    if condition == "decimal":
                        messages = {f"S{i + 1}": str(value) for i, value in enumerate(values)}
                    elif condition == "json":
                        messages = {
                            f"S{i + 1}": json.dumps({"value": value}, separators=(",", ":"))
                            for i, value in enumerate(values)
                        }
                    else:
                        messages = {f"S{i + 1}": format(value, "02b") for i, value in enumerate(values)}
                    instructions = dict(baseline_instructions[condition])
                    instructions[RECEIVER] += "\n\n" + FINAL_RUNNER_INSTRUCTION
                    per_condition[condition].append(_cost_row(
                        tokenizer, system=instructions, sender_values=values,
                        sender_messages=messages,
                        receiver_context=f"Public metadata: sender_count={agent_count}.",
                        answer=sum(values), protocol_id=f"private-sum-{condition}-v0",
                    ))

                partitions = card["sender_partitions"]
                ids = _codebook_message_ids(values, partitions)
                messages = {
                    f"S{i + 1}": format(ids[i], f"0{width}b")
                    for i, width in enumerate(card["sender_widths_bits"]) if width
                }
                codebook_system = dict(card["agent_instructions"])
                codebook_system[RECEIVER] += "\n\n" + CODEBOOK_FINAL_INSTRUCTION
                per_condition["exhaustive_codebook"].append(_cost_row(
                    tokenizer, system=codebook_system, sender_values=values,
                    sender_messages=messages,
                    receiver_context=(
                        f"Public metadata: sender_count={agent_count}. Assumed input prior: "
                        "independent_uniform_integer_0_to_3."
                    ),
                    answer=codebook_ids[ids], protocol_id=card["protocol_id"],
                ))

            summaries = {condition: _summary(rows) for condition, rows in per_condition.items()}
            records.append({
                "agent_count": agent_count,
                "budget_bits_at_most": budget,
                "actual_fixed_width_payload_bits": card["actual_fixed_width_payload_bits"],
                "codebook_exact_sum_success": card["optimal_exact_sum_success"],
                "no_message_exact_sum_success": summaries["no_message"]["exact_sum_success"]["mean"],
                "active_codebook_senders": card["active_senders"],
                "codebook_protocol_id": card["protocol_id"],
                "codebook_card_sha256": card["card_sha256"],
                "codebook_card_tokens_if_distributed_as_json": _count(
                    tokenizer,
                    json.dumps(card["dialogue_protocol_card"], ensure_ascii=False, separators=(",", ":")),
                ),
                "conditions": summaries,
                "pareto_efficient_within_agent_count": {},
            })

    # A point is dominated only when another condition has at least its exact success
    # and no greater tokenizer-content cost, application-byte cost, or call count.
    for m in range(2, 5):
        cohort = [row for row in records if row["agent_count"] == m]
        points = []
        for row in cohort:
            for condition, summary in row["conditions"].items():
                points.append((
                    row["budget_bits_at_most"], condition,
                    summary["exact_sum_success"]["mean"],
                    summary["total_known_content_tokens"]["mean"],
                    summary["delivered_application_bytes"]["mean"],
                    summary["planned_model_calls"]["mean"],
                ))
        for row in cohort:
            row["pareto_efficient_within_agent_count"] = {
                condition: not any(
                    other[2] >= summary["exact_sum_success"]["mean"]
                    and other[3] <= summary["total_known_content_tokens"]["mean"]
                    and other[4] <= summary["delivered_application_bytes"]["mean"]
                    and other[5] <= summary["planned_model_calls"]["mean"]
                    and (
                        other[2] > summary["exact_sum_success"]["mean"]
                        or other[3] < summary["total_known_content_tokens"]["mean"]
                        or other[4] < summary["delivered_application_bytes"]["mean"]
                        or other[5] < summary["planned_model_calls"]["mean"]
                    )
                    for other in points
                    if not (other[0] == row["budget_bits_at_most"] and other[1] == condition)
                )
                for condition, summary in row["conditions"].items()
            }

    sources = {
        "preregistration_md": ROOT / "research/MULTIPARTY_SUM_CODEBOOK_FRONTIER_AUDIT_PREREG_V0_2.md",
        "audit_script_py": ROOT / "research/audit_multiparty_codebook_frontier_v0_2.py",
        "sum_example_py": ROOT / "examples/multiparty_private_sum.py",
        "codebook_protocol_py": ROOT / "experiments/multiparty_sum_v0_1/codebook_protocol.py",
        "frontier_calculator_py": ROOT / "research/multiparty_sum_lossy_frontier.py",
        "frontier_artifact_json": ROOT / "research/data/MULTIPARTY_SUM_LOSSY_FRONTIER_V0_1.json",
        "sdk_channel_py": ROOT / "tacit/channel.py",
        "sdk_runtime_py": ROOT / "tacit/runtime.py",
        "previous_audit_json": ROOT / "research/data/MULTIPARTY_SUM_CODEBOOK_TOKEN_AUDIT_V0_1.json",
    }
    return {
        "schema": SCHEMA,
        "status": "preregistered exact-runner tokenizer and application-byte accounting audit",
        "supersedes": "cross-protocol comparison boundary in tlu.multiparty-sum-codebook-token-audit.v0.1.0",
        "preregistration_sha256": hashlib.sha256(sources["preregistration_md"].read_bytes()).hexdigest(),
        "tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "sha256": digest,
            "file_size_bytes": len(tokenizer_bytes),
            "library": "tokenizers",
            "library_version": tokenizers_version,
            "chat_template_included": False,
            "provider_special_tokens_included": False,
        },
        "python_version": platform.python_version(),
        "accounting_boundary": (
            "Actual SDK system/user content strings and ideal completion content are tokenizer-counted. "
            "Delivered application bytes use LocalTCPMessageChannel.measure (JSON envelope, length prefix, ACK; "
            "network headers excluded). No-message has zero transmissions and one receiver call. Card JSON tokens "
            "are separate and excluded from episode costs. No actual model behavior is measured."
        ),
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "conditions": list(CONDITIONS),
        "design": {
            "agent_counts": [2, 3, 4],
            "uniform_vectors_per_agent_count": {str(m): 4 ** m for m in range(2, 5)},
            "baselines_use_actual_run_sum_episode_receiver_scaffold": True,
            "task_values": [0, 1, 2, 3],
            "no_message_bayes_guess": "floor(3*m/2)",
            "codebook_receiver": "frozen exhaustive MAP decoder",
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, default=ROOT / TOKENIZER_RELATIVE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        parser.error("output path must stay inside the project")
    if output.exists() and not args.force:
        parser.error("output already exists; choose a project-local path or pass --force")
    try:
        report = build_report(args.tokenizer_json)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "output": str(output), "schema": report["schema"],
        "settings": len(report["records"]), "tokenizer_sha256": report["tokenizer"]["sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
