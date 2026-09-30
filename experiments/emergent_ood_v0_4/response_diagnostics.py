"""Offline, metadata-only response diagnostics; never a capability ledger.

Inspect raw API envelopes before adapter validation. Content is passed unchanged
into the frozen runner parser. No wrapper removal or reasoning fallback occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence

from experiments.emergent_ood_v0_4.runner import _parse_choice


@dataclass(frozen=True)
class ResponseConfiguration:
    """Operator-supplied immutable identifiers, not auto-detected provenance.

Use digests/revisions for model artifact, template and service configuration;
never put credentials, raw templates or reasoning text in these identifiers.
"""

    model_id: str
    model_artifact_id: str
    template_id: str
    service_build_id: str
    service_config_id: str
    reasoning_mode: str
    reasoning_format: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("configuration identifiers must be nonempty strings")


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "non_json"


def _field_metadata(container: dict, name: str) -> dict:
    return {"present": name in container,
            "type": _json_type(container[name]) if name in container else "missing"}


def diagnose_response(payload: Any, *, candidate_ids: Sequence[str],
                      configuration: ResponseConfiguration) -> dict:
    """Return orthogonal observations without answers, gold labels or raw text.

Truncation is provider-declared (finish_reason=length), not inferred from text
or token count. A valid ID can still be reported as truncated. Format validity
says nothing about correctness and cannot authorize communication arms.
"""
    if (isinstance(candidate_ids, (str, bytes)) or not candidate_ids
            or any(not isinstance(v, str) or not v for v in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)):
        raise ValueError("candidate_ids must be distinct nonempty strings")
    root = payload if isinstance(payload, dict) else {}
    choices = root.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    envelope_valid = isinstance(message, dict)
    message = message if envelope_valid else {}
    content = message.get("content")
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    # Do not echo arbitrary provider strings: some endpoints put text in errors.
    known_reasons = ("stop", "length", "content_filter", "tool_calls", "function_call")
    finish_status = (finish_reason if isinstance(finish_reason, str) and finish_reason in known_reasons
                     else "missing_or_null" if finish_reason is None else "other")
    valid = isinstance(content, str) and _parse_choice(content, candidate_ids)[1]
    think = isinstance(content, str) and bool(re.search(r"</?think(?:\s[^>]*)?>", content, re.IGNORECASE))
    if not envelope_valid:
        shape = "invalid_envelope"
    elif "content" not in message:
        shape = "missing_content"
    elif not isinstance(content, str):
        shape = "non_text_content"
    elif not content.strip():
        shape = "empty_content"
    elif think:
        shape = "think_wrapper"
    elif valid:
        shape = "candidate_id"
    elif any(re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", content) for v in candidate_ids):
        shape = "extra_text"
    else:
        shape = "unrecognized_text"
    return {
        "schema": "tlu.response-diagnostic.v0.1",
        "evidence_kind": "interface_diagnostic_only",
        "configuration": asdict(configuration),
        "configuration_provenance": "operator_supplied_unverified",
        "response_model": _field_metadata(root, "model"),
        "response_model_matches_requested": root.get("model") == configuration.model_id,
        "envelope_valid": envelope_valid,
        "content": _field_metadata(message, "content"),
        "content_shape": shape,
        "think_marker_present": think,
        "strict_format_valid": valid,
        "finish_reason": finish_status,
        "truncated": finish_reason == "length",
        # Presence includes explicit null and empty string. Never inspect values.
        "reasoning_fields": {name: _field_metadata(message, name)
                             for name in ("reasoning_content", "reasoning")},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="local JSON object: configuration, candidate_ids, response")
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.input.read_text(encoding="utf-8"))
        result = diagnose_response(record["response"], candidate_ids=record["candidate_ids"],
                                   configuration=ResponseConfiguration(**record["configuration"]))
    except (OSError, ValueError, TypeError, KeyError):
        parser.exit(2, "invalid diagnostic input (details suppressed to avoid exposing response text)\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
