"""Audit raw v0.4 results from the frozen compact labeled-fields card."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.emergent_ood_v0_4.compact_fields import (  # noqa: E402
    PROTOCOL_ID,
    audit_result_rows,
)
from tacit.protocol import ProtocolCard  # noqa: E402


SCHEMA = "tlu.compact-labeled-fields-audit.v3"
DEFAULT_CARD = ROOT / "examples" / "compact_labeled_fields_v3.json"
EXPECTED_CARD_SHA256 = "2646cd19c10aa5f4bec19d8e38fdb43eac6c15238630b86f79c794bff1a46aa2"


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("all input and output paths must stay inside the project") from exc
    return resolved


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = _inside_project(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL record at line {line_number}")
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL record at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("input JSONL is empty")
    return rows, digest


def verify_run_manifest(
    manifest_raw: bytes, *, input_path: Path, input_sha256: str,
    card_sha256: str, row_count: int,
) -> str:
    """Verify the runner sidecar binds the source and exact frozen condition."""
    try:
        manifest = json.loads(manifest_raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("runner manifest is invalid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "tlu.emergent-ood-run-manifest.v0.4"
        or manifest.get("results_file") != input_path.name
        or manifest.get("results_sha256") != input_sha256
        or manifest.get("result_records") != row_count
        or manifest.get("conditions") != ["shared_protocol_card"]
        or manifest.get("protocol_card_sha256") != card_sha256
    ):
        raise ValueError("runner manifest does not bind this input to the frozen card and condition")
    return hashlib.sha256(manifest_raw).hexdigest()


def build_report(
    rows: list[dict[str, Any]], *, input_sha256: str, card_sha256: str,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    audited = audit_result_rows(rows)
    count = len(audited)
    syntax_count = sum(row["exact_format_valid"] for row in audited)
    parse_syntax_count = sum(row["syntactic_parse_valid"] for row in audited)
    parse_count = sum(row["semantic_parse_valid"] for row in audited)
    fidelity_count = sum(row["canonical_label_fidelity"] for row in audited)
    task_count = sum(row["exact_selection"] for row in audited)
    joint_count = sum(row["canonical_label_fidelity"] and row["exact_selection"] for row in audited)
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "input_sha256": input_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "protocol_card_sha256": card_sha256,
        "row_count": count,
        "summary": {
            "syntactic_parse_valid_count": parse_syntax_count,
            "exact_format_valid_count": syntax_count,
            "semantic_parse_valid_count": parse_count,
            "canonical_label_fidelity_count": fidelity_count,
            "exact_selection_count": task_count,
            "fidelity_and_exact_selection_count": joint_count,
        },
        "rows": audited,
        "interpretation_limit": (
            "This offline audit scores the frozen message grammar and sender-target fidelity; "
            "it does not alter the source cost ledger or establish protocol superiority."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="raw v0.4 runner JSONL")
    parser.add_argument("--output", required=True, type=Path, help="project-local audit JSON")
    parser.add_argument("--protocol-card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args()
    try:
        input_path = _inside_project(args.input)
        card_path = _inside_project(args.protocol_card)
        output_path = _inside_project(args.output)
        if output_path in {input_path, card_path}:
            raise ValueError("audit output must not overwrite the raw ledger or protocol card")
        card, card_raw = ProtocolCard.read(str(card_path))
        if card.protocol_id != PROTOCOL_ID:
            raise ValueError("protocol card is not the frozen compact labeled-fields card")
        card_sha256 = hashlib.sha256(card_raw).hexdigest()
        if card_sha256 != EXPECTED_CARD_SHA256:
            raise ValueError("protocol card bytes differ from the frozen baseline artifact")
        rows, input_sha256 = load_jsonl(input_path)
        run_manifest_path = _inside_project(
            input_path.with_suffix(input_path.suffix + ".manifest.json")
        )
        if output_path == run_manifest_path:
            raise ValueError("audit output must not overwrite the runner manifest")
        manifest_raw = run_manifest_path.read_bytes()
        run_manifest_sha256 = verify_run_manifest(
            manifest_raw,
            input_path=input_path,
            input_sha256=input_sha256,
            card_sha256=card_sha256,
            row_count=len(rows),
        )
        report = build_report(
            rows,
            input_sha256=input_sha256,
            card_sha256=card_sha256,
            run_manifest_sha256=run_manifest_sha256,
        )
        if output_path.exists() and not args.force:
            raise ValueError(f"refusing to overwrite {output_path}; pass --force explicitly")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = _inside_project(output_path.with_name(output_path.name + ".tmp"))
        if temporary_path in {input_path, card_path, run_manifest_path}:
            raise ValueError("temporary output path conflicts with an input or manifest")
        if temporary_path.exists():
            raise ValueError(f"refusing to overwrite temporary file {temporary_path}")
        with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary_path, output_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": str(output_path),
        "input_sha256": input_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "protocol_card_sha256": card_sha256,
        "row_count": report["row_count"],
        "summary": report["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
