"""Audit frozen default/robotics/music protocol prompts with the Mistral tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.audit_protocol_token_costs import (  # noqa: E402
    PROTOCOL_CARD_SHA256,
    audit_bundle,
    _project_path,
    _read_json,
)


MISTRAL_REPO = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_REVISION = "adadfb3fbae87ecc77cd5bf2c3318434d5da04cf"
MISTRAL_TOKENIZER_SHA256 = "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49"
ONTOLOGIES = ("default", "robotics", "music")
SCHEMA = "tlu.mistral-ontology-token-cost-audit.v1"


def build_report(
    *,
    tokenizer_json: Path,
    bundles: Sequence[tuple[str, Path, int]],
) -> dict[str, Any]:
    if tuple(name for name, _, _ in bundles) != ONTOLOGIES:
        raise ValueError("bundle order must be exactly default, robotics, music")
    tokenizer_json = _project_path(tokenizer_json)
    raw_tokenizer = tokenizer_json.read_bytes()
    tokenizer_digest = hashlib.sha256(raw_tokenizer).hexdigest()
    if tokenizer_digest != MISTRAL_TOKENIZER_SHA256:
        raise ValueError("tokenizer hash does not match the pinned Mistral repository artifact")
    try:
        from tokenizers import Tokenizer, __version__ as tokenizers_version
    except ImportError as exc:
        raise RuntimeError("install the optional `tokenizers` package to run this audit") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_json))

    card_path = ROOT / "examples" / "compact_labeled_fields_v3.json"
    card_bytes = card_path.read_bytes()
    card_digest = hashlib.sha256(card_bytes).hexdigest()
    if card_digest != PROTOCOL_CARD_SHA256:
        raise ValueError("protocol card hash does not match the frozen v0.3 artifact")
    card = _read_json(card_path)

    qwen_path = ROOT / "research" / "data" / "COMPACT_FIELDS_QWEN3_4B_TOKEN_AUDIT_V0_1.json"
    if not qwen_path.is_file():
        raise ValueError("the frozen Qwen ontology tokenizer report is required")
    qwen_report = _read_json(qwen_path)
    expected_bundles = {entry["ontology"]: entry for entry in qwen_report["results"]}
    qwen_hash = hashlib.sha256(qwen_path.read_bytes()).hexdigest()

    bundle_results = []
    for ontology, bundle_path, split_seed in bundles:
        result = audit_bundle(
            bundle_path=bundle_path,
            split_seed=split_seed,
            ontology=ontology,
            tokenizer=tokenizer,
            protocol_card=card,
        )
        prior = expected_bundles.get(ontology)
        if prior is None:
            raise ValueError(f"the frozen Qwen report has no {ontology} bundle")
        if (
            prior["split_seed"] != split_seed
            or prior["manifest_sha256"] != result["manifest_sha256"]
            or prior["episode_count"] != result["episode_count"]
        ):
            raise ValueError(f"{ontology} bundle does not exactly match the Qwen audit fixture")
        qwen_delta = prior["condition_minus_json_known_content_tokens_per_episode"]
        mistral_delta = result["condition_minus_json_known_content_tokens_per_episode"]
        result["mistral_minus_qwen_delta_shift"] = {
            condition: round(
                mistral_delta[condition]["mean"] - qwen_delta[condition]["mean"], 3
            )
            for condition in ("shared_protocol_card", "symbolic")
        }
        bundle_results.append(result)

    runner_path = ROOT / "experiments" / "emergent_ood_v0_4" / "runner.py"
    codec_path = ROOT / "experiments" / "emergent_ood_v0_4" / "compact_fields.py"
    qwen_auditor_path = ROOT / "research" / "audit_protocol_token_costs.py"
    cross_tokenizer_path = ROOT / "research" / "audit_protocol_tokenizer_transfer.py"
    prereg_path = ROOT / "research" / "PROTOCOL_TOKENIZER_ONTOLOGY_TRANSFER_PREREG_V0_1.md"
    return {
        "schema": SCHEMA,
        "status": "preregistered deterministic ontology cost replication",
        "tokenizer": {
            "repo": MISTRAL_REPO,
            "revision": MISTRAL_REVISION,
            "sha256": tokenizer_digest,
            "file_size_bytes": len(raw_tokenizer),
            "vocabulary_size_with_added_tokens": tokenizer.get_vocab_size(with_added_tokens=True),
            "library": "tokenizers",
            "library_version": tokenizers_version,
            "special_tokens_added": False,
        },
        "python_version": platform.python_version(),
        "preregistration_sha256": hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
        "qwen_report_sha256": qwen_hash,
        "source_sha256": {
            "runner_py": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
            "compact_fields_py": hashlib.sha256(codec_path.read_bytes()).hexdigest(),
            "protocol_card_json": card_digest,
            "qwen_audit_py": hashlib.sha256(qwen_auditor_path.read_bytes()).hexdigest(),
            "mistral_synthetic_audit_py": hashlib.sha256(cross_tokenizer_path.read_bytes()).hexdigest(),
            "this_audit_py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "measurement": (
            "Exact runner content prompt reconstruction over the frozen 192 default/robotics/music test episodes. "
            "Known total includes sender and receiver input content, the echoed message, and ideal sender output; "
            "it excludes chat templates, receiver output, provider usage, caching, inference compute, and task outcomes."
        ),
        "results": bundle_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--default-bundle", type=Path, required=True)
    parser.add_argument("--robotics-bundle", type=Path, required=True)
    parser.add_argument("--music-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = _project_path(args.output)
    if output.exists() and not args.force:
        parser.error("output already exists; choose a new project-local path")
    report = build_report(
        tokenizer_json=args.tokenizer_json,
        bundles=[
            ("default", args.default_bundle, 23),
            ("robotics", args.robotics_bundle, 31),
            ("music", args.music_bundle, 31),
        ],
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
