"""Compare the frozen protocol-cost fixtures under a second model tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.emergent_ood_v0_4.runner import _Protocol  # noqa: E402
from research.audit_protocol_token_costs import TOKENIZER_SHA256, _project_path  # noqa: E402
from research.audit_protocol_token_scaling import (  # noqa: E402
    CARDINALITIES,
    audit_setting,
)


MISTRAL_REPO = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_REVISION = "adadfb3fbae87ecc77cd5bf2c3318434d5da04cf"
MISTRAL_TOKENIZER_SHA256 = "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49"
DIMENSIONS = (4, 10, *range(20, 41))
SCHEMA = "tlu.protocol-tokenizer-transfer.v1"


def build_report(tokenizer_json: Path) -> dict[str, Any]:
    tokenizer_json = _project_path(tokenizer_json)
    tokenizer_bytes = tokenizer_json.read_bytes()
    tokenizer_digest = hashlib.sha256(tokenizer_bytes).hexdigest()
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
    if card_digest != "2646cd19c10aa5f4bec19d8e38fdb43eac6c15238630b86f79c794bff1a46aa2":
        raise ValueError("the compact-fields card is not the pinned v0.3 artifact")
    card = json.loads(card_bytes.decode("utf-8"))

    preregistration_path = ROOT / "research" / "PROTOCOL_TOKENIZER_TRANSFER_PREREG_V0_1.md"
    qwen_report_path = ROOT / "research" / "data" / "PROTOCOL_TOKEN_SCALING_CONFIRMATION_V0_1.json"
    if not qwen_report_path.is_file():
        raise ValueError("the preregistered Qwen confirmation artifact is required")
    runner_path = ROOT / "experiments" / "emergent_ood_v0_4" / "runner.py"
    codec_path = ROOT / "experiments" / "emergent_ood_v0_4" / "compact_fields.py"
    return {
        "schema": SCHEMA,
        "status": "preregistered deterministic cross-tokenizer cost comparison",
        "tokenizer": {
            "repo": MISTRAL_REPO,
            "revision": MISTRAL_REVISION,
            "sha256": tokenizer_digest,
            "file_size_bytes": len(tokenizer_bytes),
            "vocabulary_size_with_added_tokens": tokenizer.get_vocab_size(with_added_tokens=True),
            "library": "tokenizers",
            "library_version": tokenizers_version,
            "special_tokens_added": False,
            "license": "Apache-2.0 (model repository metadata)",
        },
        "python_version": platform.python_version(),
        "preregistration_sha256": hashlib.sha256(preregistration_path.read_bytes()).hexdigest(),
        "qwen_confirmation_sha256": hashlib.sha256(qwen_report_path.read_bytes()).hexdigest(),
        "source_sha256": {
            "runner_py": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
            "compact_fields_py": hashlib.sha256(codec_path.read_bytes()).hexdigest(),
            "protocol_card_json": card_digest,
            "audit_protocol_token_scaling_py": hashlib.sha256(
                (ROOT / "research" / "audit_protocol_token_scaling.py").read_bytes()
            ).hexdigest(),
            "audit_protocol_tokenizer_transfer_py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "measurement": (
            "Deterministic synthetic prompt-cost comparison using the same v0.4 content construction and task fixtures "
            "as the Qwen sweep. Known total includes both agent input content, receiver-side message echo, and ideal "
            "sender output. Model task outcomes, chat templates, receiver output, provider usage, caching, and inference "
            "compute are not measured."
        ),
        "design": {
            "dimensions": list(DIMENSIONS),
            "cardinalities": list(CARDINALITIES),
            "settings": len(DIMENSIONS) * len(CARDINALITIES),
            "episodes_per_setting": 16,
            "candidate_count": 4,
            "conditions": ["json", "shared_protocol_card", "symbolic"],
        },
        "sweep": [
            audit_setting(
                dimensions=dimension,
                values=values,
                tokenizer=tokenizer,
                protocol_card=card,
            )
            for dimension in DIMENSIONS
            for values in CARDINALITIES
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = _project_path(args.output)
    if output.exists() and not args.force:
        parser.error("output already exists; choose a new project-local path")
    report = build_report(args.tokenizer_json)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
