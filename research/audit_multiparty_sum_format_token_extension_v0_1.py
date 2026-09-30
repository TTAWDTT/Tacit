"""Token-count shipped labeled and fixed-sentence sum baselines."""
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
    RECEIVER, _role_instructions,
)
from research.audit_multiparty_codebook_frontier_v0_2 import (  # noqa: E402
    FINAL_RUNNER_INSTRUCTION,
    TOKENIZER_REPO,
    TOKENIZER_RELATIVE_PATH,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
    _cost_row,
    _summary,
)


CONDITIONS = ("labeled", "sentence")
SCHEMA = "tlu.multiparty-sum-format-token-extension.v0.1.0"
P24_PATH = ROOT / "research/data/MULTIPARTY_SUM_CODEBOOK_FRONTIER_AUDIT_V0_2.json"


def build_report(tokenizer_path: Path) -> dict[str, Any]:
    tokenizer_path = tokenizer_path.resolve()
    try:
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
    p24_bytes = P24_PATH.read_bytes()
    p24 = json.loads(p24_bytes)
    p24_by_m = {
        m: [row for row in p24["records"] if row["agent_count"] == m]
        for m in range(2, 5)
    }
    records = []
    for agent_count in range(2, 5):
        by_condition: dict[str, list[dict[str, int]]] = {name: [] for name in CONDITIONS}
        names = [f"S{i + 1}" for i in range(agent_count)]
        for values in itertools.product(range(4), repeat=agent_count):
            for condition in CONDITIONS:
                messages = {}
                for index, value in enumerate(values, start=1):
                    if condition == "labeled":
                        messages[f"S{index}"] = f"v={value}"
                    else:
                        messages[f"S{index}"] = f"My private integer is {value}."
                instructions = _role_instructions(condition, names)
                instructions[RECEIVER] += "\n\n" + FINAL_RUNNER_INSTRUCTION
                by_condition[condition].append(_cost_row(
                    tokenizer,
                    system=instructions,
                    sender_values=values,
                    sender_messages=messages,
                    receiver_context=f"Public metadata: sender_count={agent_count}.",
                    answer=sum(values),
                    protocol_id=f"private-sum-{condition}-v0",
                ))
        extended = {name: _summary(rows) for name, rows in by_condition.items()}
        p24_records = p24_by_m[agent_count]
        p24_no_message_reference = p24_records[0]["conditions"]
        p24_zero_error_row = next(row for row in p24_records if row["codebook_exact_sum_success"] == 1.0)
        references = {
            **{
                name: p24_no_message_reference[name]
                for name in ("no_message", "decimal", "json", "binary")
            },
            "zero_error_codebook": p24_zero_error_row["conditions"]["exhaustive_codebook"],
        }
        records.append({
            "agent_count": agent_count,
            "uniform_source_vectors": 4 ** agent_count,
            "extension_conditions": extended,
            "zero_error_codebook_reference_budget_bits": p24_zero_error_row["budget_bits_at_most"],
            "existing_p24_reference_conditions": {
                name: references[name]
                for name in ("no_message", "decimal", "json", "binary", "zero_error_codebook")
            },
            "extension_minus_existing_total_known_content_tokens": {
                ext_name: {
                    ref_name: round(
                        extended[ext_name]["total_known_content_tokens"]["mean"]
                        - references[ref_name]["total_known_content_tokens"]["mean"],
                        6,
                    )
                    for ref_name in ("decimal", "json", "binary", "zero_error_codebook")
                }
                for ext_name in CONDITIONS
            },
        })

    sources = {
        "preregistration_md": ROOT / "research/MULTIPARTY_SUM_FORMAT_TOKEN_EXTENSION_PREREG_V0_1.md",
        "audit_script_py": ROOT / "research/audit_multiparty_sum_format_token_extension_v0_1.py",
        "sum_example_py": ROOT / "examples/multiparty_private_sum.py",
        "p24_audit_script_py": ROOT / "research/audit_multiparty_codebook_frontier_v0_2.py",
        "p24_data_json": P24_PATH,
    }
    return {
        "schema": SCHEMA,
        "status": "preregistered deterministic tokenizer and application-byte extension",
        "tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "sha256": digest,
            "file_size_bytes": len(tokenizer_bytes),
            "library": "tokenizers",
            "library_version": tokenizers_version,
            "chat_template_included": False,
        },
        "p24_data_sha256": hashlib.sha256(p24_bytes).hexdigest(),
        "preregistration_sha256": hashlib.sha256(sources["preregistration_md"].read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "accounting_boundary": (
            "Same operational role-instruction constructor and SDK cost estimator as P24. Content-token totals include "
            "sender/receiver system+user inputs and ideal completion outputs. Application bytes use the loopback SDK "
            "JSON envelope/length-prefix/ACK measurement; TCP/IP headers, chat templates, provider tokens, real model "
            "behavior, latency, inference compute, and billing are excluded."
        ),
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "conditions": list(CONDITIONS),
        "design": {
            "agent_counts": [2, 3, 4],
            "vectors_per_count": {str(m): 4 ** m for m in range(2, 5)},
            "message_serialization": {
                "labeled": "v=N",
                "sentence": "My private integer is N.",
            },
            "ideal_sender_fidelity": True,
            "ideal_receiver_exact_sum": True,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "schema": report["schema"],
        "source_vectors": sum(4 ** m for m in range(2, 5)),
        "tokenizer_sha256": report["tokenizer"]["sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
