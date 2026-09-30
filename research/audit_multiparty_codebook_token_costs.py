"""Exhaustive tokenizer-only accounting for frozen finite-sum protocol cards."""
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
    RECEIVER,
    SUM_TASK,
    _format_instructions,
)
from experiments.multiparty_sum_v0_1.codebook_protocol import build_protocol_card  # noqa: E402
from research.multiparty_sum_lossy_frontier import exact_frontier  # noqa: E402


TOKENIZER_REPO = "Qwen/Qwen3-4B"
TOKENIZER_REVISION = "eb971e9fb1f41c13b5e5a56e56886305c5ad94a0"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_RELATIVE_PATH = Path(
    ".cache/tokenizers/hfhub/models--Qwen--Qwen3-4B/snapshots"
) / TOKENIZER_REVISION / "tokenizer.json"
FINAL_ANSWER_INSTRUCTION = "Return only the exact non-negative integer sum as base-10 digits."
SCHEMA = "tlu.multiparty-sum-codebook-token-audit.v0.1.0"
CONDITIONS = ("decimal", "json", "binary", "exhaustive_codebook")


def _count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def _user_content(tokenizer: Any, *, task: str, context: str, transcript: list[dict[str, str]], instruction: str) -> int:
    payload = {
        "task": task,
        "private_context": context,
        "visible_transcript": transcript,
        "instruction": instruction,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return _count(tokenizer, serialized)


def _message_ids(values: tuple[int, ...], partitions: list[list[list[int]]]) -> tuple[int, ...]:
    result = []
    for value, partition in zip(values, partitions):
        result.append(next(i for i, block in enumerate(partition) if value in block))
    return tuple(result)


def _summary(rows: list[dict[str, int]]) -> dict[str, Any]:
    names = tuple(rows[0])
    report: dict[str, Any] = {"episodes": len(rows)}
    for name in names:
        values = [row[name] for row in rows]
        report[name] = {
            "mean": round(sum(values) / len(values), 4),
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

    records: list[dict[str, Any]] = []
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
            for values in itertools.product(range(4), repeat=agent_count):
                for condition in CONDITIONS:
                    if condition == "exhaustive_codebook":
                        role_instructions = card["agent_instructions"]
                        active_senders = card["active_senders"]
                        partitions = card["sender_partitions"]
                        ids = _message_ids(values, partitions)
                        codewords: dict[str, str] = {}
                        for index, width in enumerate(card["sender_widths_bits"]):
                            if width:
                                codewords[f"S{index + 1}"] = format(ids[index], f"0{width}b")
                        decoder_lookup = {
                            tuple(row["message_ids"]): row["decoded_sum"]
                            for row in frontier_row["optimal_receiver_decoder"]
                        }
                        predicted_sum = decoder_lookup[ids]
                        final_output = str(predicted_sum)
                        receiver_context = (
                            f"Public metadata: sender_count={agent_count}. Assumed input prior: "
                            "independent_uniform_integer_0_to_3."
                        )
                        receiver_instruction = role_instructions[RECEIVER] + "\n\n" + FINAL_ANSWER_INSTRUCTION
                    else:
                        sender_instruction, receiver_instruction_base = _format_instructions(condition)
                        role_instructions = {f"S{i + 1}": sender_instruction for i in range(agent_count)}
                        role_instructions[RECEIVER] = receiver_instruction_base
                        active_senders = [f"S{i + 1}" for i in range(agent_count)]
                        if condition == "decimal":
                            codewords = {f"S{i + 1}": str(value) for i, value in enumerate(values)}
                        elif condition == "json":
                            codewords = {
                                f"S{i + 1}": json.dumps({"value": value}, separators=(",", ":"))
                                for i, value in enumerate(values)
                            }
                        else:
                            codewords = {f"S{i + 1}": format(value, "02b") for i, value in enumerate(values)}
                        predicted_sum = sum(values)
                        final_output = str(predicted_sum)
                        receiver_context = f"Public metadata: sender_count={agent_count}."
                        receiver_instruction = receiver_instruction_base + "\n\n" + FINAL_ANSWER_INSTRUCTION

                    sender_system = sum(
                        _count(tokenizer, role_instructions[sender]) for sender in active_senders
                    )
                    sender_user = 0
                    sender_outputs = 0
                    for sender in active_senders:
                        index = int(sender[1:]) - 1
                        context = (
                            f"Public metadata: sender_count={agent_count}. Your private integer is {values[index]}."
                        )
                        sender_user += _user_content(
                            tokenizer, task=SUM_TASK, context=context, transcript=[],
                            instruction="Send exactly the next message for this turn.",
                        )
                        sender_outputs += _count(tokenizer, codewords[sender])
                    receiver_transcript = [
                        {"sender": sender, "message": codewords[sender]}
                        for sender in active_senders
                    ]
                    receiver_user = _user_content(
                        tokenizer, task=SUM_TASK, context=receiver_context,
                        transcript=receiver_transcript,
                        instruction="Return the final task answer for external scoring.",
                    )
                    receiver_system = _count(tokenizer, receiver_instruction)
                    receiver_output = _count(tokenizer, final_output)
                    per_condition[condition].append({
                        "sender_role_instruction_tokens": sender_system,
                        "receiver_role_instruction_tokens": receiver_system,
                        "role_instruction_tokens": sender_system + receiver_system,
                        "sender_user_content_tokens": sender_user,
                        "receiver_user_content_tokens": receiver_user,
                        "sender_completion_tokens": sender_outputs,
                        "receiver_completion_tokens": receiver_output,
                        "total_model_input_content_tokens": sender_system + receiver_system + sender_user + receiver_user,
                        "total_completion_content_tokens": sender_outputs + receiver_output,
                        "total_known_content_tokens": sender_system + receiver_system + sender_user + receiver_user + sender_outputs + receiver_output,
                        "exact_sum_success": int(predicted_sum == sum(values)),
                    })

            summarized = {condition: _summary(rows) for condition, rows in per_condition.items()}
            serialized_card = card["dialogue_protocol_card"]
            card_json = json.dumps(serialized_card, ensure_ascii=False, separators=(",", ":"))
            records.append({
                "agent_count": agent_count,
                "budget_bits_at_most": budget,
                "actual_fixed_width_payload_bits": card["actual_fixed_width_payload_bits"],
                "frontier_exact_sum_success": card["optimal_exact_sum_success"],
                "active_senders": card["active_senders"],
                "omitted_sender_indices": card["omitted_sender_indices"],
                "protocol_id": card["protocol_id"],
                "codebook_card_sha256": card["card_sha256"],
                "codebook_card_utf8_bytes": card["card_json_utf8_bytes"],
                "codebook_card_content_tokens_if_sent_as_plain_text": _count(tokenizer, card_json),
                "conditions": summarized,
                "planned_model_calls_per_episode": {
                    "decimal": agent_count + 1,
                    "json": agent_count + 1,
                    "binary": agent_count + 1,
                    "exhaustive_codebook": len(card["active_senders"]) + 1,
                },
                "condition_minus_decimal_total_known_content_tokens_per_episode": {
                    condition: round(
                        summarized[condition]["total_known_content_tokens"]["mean"]
                        - summarized["decimal"]["total_known_content_tokens"]["mean"], 4
                    )
                    for condition in ("json", "binary", "exhaustive_codebook")
                },
            })

    source_files = {
        "codebook_protocol_py": ROOT / "experiments/multiparty_sum_v0_1/codebook_protocol.py",
        "codebook_runner_py": ROOT / "experiments/multiparty_sum_v0_1/codebook_runner.py",
        "sum_example_py": ROOT / "examples/multiparty_private_sum.py",
        "frontier_calculator_py": ROOT / "research/multiparty_sum_lossy_frontier.py",
        "frontier_json": ROOT / "research/data/MULTIPARTY_SUM_LOSSY_FRONTIER_V0_1.json",
        "preregistration_md": ROOT / "research/MULTIPARTY_SUM_CODEBOOK_TOKEN_AUDIT_PREREG_V0_1.md",
        "audit_script_py": ROOT / "research/audit_multiparty_codebook_token_costs.py",
    }
    return {
        "schema": SCHEMA,
        "status": "deterministic tokenizer-only exhaustive content-cost audit",
        "preregistration_sha256": hashlib.sha256(source_files["preregistration_md"].read_bytes()).hexdigest(),
        "tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "sha256": digest,
            "file_size_bytes": len(tokenizer_bytes),
            "library": "tokenizers",
            "library_version": tokenizers_version,
            "special_tokens_added": False,
            "chat_template_included": False,
        },
        "python_version": platform.python_version(),
        "measurement_boundary": (
            "Counts runtime system and user content strings plus ideal sender and final-answer outputs; excludes chat "
            "templates, provider-added special tokens, hidden reasoning, actual model behavior, billing, latency, and compute. "
            "Card JSON token count is a separate distribution diagnostic and is not included in episode totals."
        ),
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files.items()
        },
        "conditions": list(CONDITIONS),
        "design": {
            "agent_counts": [2, 3, 4],
            "private_values": [0, 1, 2, 3],
            "uniform_vectors_enumerated_per_agent_count": {str(m): 4 ** m for m in range(2, 5)},
            "baselines": ["decimal", "json", "binary"],
            "codebook": "each active budget row from frozen exhaustive MAP frontier",
        },
        "frontier_records": records,
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
        parser.error("output already exists; choose a new project-local path or pass --force")
    try:
        report = build_report(args.tokenizer_json)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "schema": report["schema"],
        "settings": len(report["frontier_records"]),
        "tokenizer_sha256": report["tokenizer"]["sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
