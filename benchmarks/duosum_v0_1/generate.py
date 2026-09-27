"""Generate deterministic two-agent hidden-sum episodes.

The task deliberately uses a very small, exact operation so communication
necessity can be checked independently from long-chain reasoning. Agent inputs
are private in the simulator context; the benchmark file contains gold data for
evaluation and reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "tasks"
AGENTS = 2


def build_case(bits: int, replicate: int, seed: int) -> dict[str, Any]:
    if bits < 2:
        raise ValueError("bit width must be at least 2")
    rng = random.Random(seed)
    maximum = (1 << bits) - 1
    private_values = [rng.randint(1, maximum) for _ in range(AGENTS)]
    total = sum(private_values)
    case_id = f"DUOSUM-B{bits:02d}-S{replicate:02d}"
    split = "calibration" if replicate == 0 else "heldout"
    description = (
        "Two agents each hold one private positive integer. Determine the exact "
        "sum of both values. Every agent must submit the same integer answer. "
        "An agent knows its own value, but the other agent's value is private."
    )
    agent_configs = []
    for agent_id, value in enumerate(private_values):
        user_prompt = (
            f"Task: compute the exact sum of the two agents' private values.\n"
            f"You are Agent {agent_id} of {AGENTS}.\n"
            f"Your private value is {value}.\n"
            "The other agent has its own private value; you do not know it. "
            "Communicate as needed, then every agent must "
            "submit the exact same total using submit_result(integer)."
        )
        agent_configs.append(
            {
                "agent_id": agent_id,
                "input_shard": value,
                "user_prompt": user_prompt,
            }
        )

    return {
        "case_id": case_id,
        "case_name": "Two-Agent Hidden Sum",
        "paradigm": "Synthetic exact hidden-information aggregation",
        "task_description": description,
        "metadata": {
            "num_agents": AGENTS,
            "is_segmented": False,
            "input_bits": bits,
            "input_domain": [1, maximum],
            "replicate": replicate,
            "split": split,
            "generation_seed": seed,
            "communication_required": True,
        },
        "agent_configs": agent_configs,
        "expected_output": {
            "type": "distributed",
            "per_agent_values": [total] * AGENTS,
            "is_segmented": False,
            "verification_logic": (
                "len(agent_outputs) == num_agents and "
                "all(agent_outputs[i] == expected_output[i] "
                "for i in range(num_agents))"
            ),
            "verification_logic_hint": "each agent submits the sum of both private integers",
        },
    }


def write_suite(
    output: Path, bit_widths: list[int], repeats: int, seed_base: int
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for bits in bit_widths:
        for replicate in range(repeats):
            seed = seed_base + bits * 100 + replicate
            case = build_case(bits, replicate, seed)
            total = case["expected_output"]["per_agent_values"][0]
            local_values = [agent["input_shard"] for agent in case["agent_configs"]]
            if any(value == total for value in local_values):
                raise AssertionError("positive other input must make local answer insufficient")
            path = output / f"{case['case_id']}.json"
            encoded = json.dumps(case, ensure_ascii=False, indent=2) + "\n"
            path.write_text(encoded, encoding="utf-8", newline="\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(
                {
                    "file": path.name,
                    "sha256": digest,
                    "bits": bits,
                    "replicate": replicate,
                    "split": case["metadata"]["split"],
                    "seed": seed,
                    "input_values": local_values,
                    "gold_sum": total,
                }
            )

    manifest = {
        "schema_version": "duosum-v0.1",
        "agents": AGENTS,
        "seed_base": seed_base,
        "bit_widths": bit_widths,
        "repeats_per_width": repeats,
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=2026092700)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    manifest = write_suite(args.output, args.bits, args.repeats, args.seed_base)
    print(f"Wrote {len(manifest['files'])} cases and manifest to {args.output}")
    print(
        "Communication-necessity check passed for every case: each private "
        "value is strictly less than the gold sum."
    )


if __name__ == "__main__":
    main()
