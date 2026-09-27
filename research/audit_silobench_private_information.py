"""Audit simple Silo-Bench Paradigm-I cases for local-answer sufficiency.

This is an instance-level sanity check, not a claim that all tasks in a family
are or are not communication-dependent. It compares each agent's exact local
answer under the task's stated operation with the benchmark's global answer.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import operator
import os
import statistics
from pathlib import Path
from typing import Any, Callable


def _local_solver(case_id: str) -> Callable[[list[Any]], Any]:
    solvers: dict[str, Callable[[list[Any]], Any]] = {
        "I-01": max,
        "I-02": lambda values: values.count("apple"),
        "I-03": lambda values: collections.Counter(values).most_common(1)[0][0],
        "I-04": lambda values: any("ERROR" in value for value in values),
        "I-05": lambda values: sum(100 <= value <= 500 for value in values),
        "I-06": _xor,
        "I-07": lambda values: sum(values) / len(values),
        "I-08": lambda values: len(set(values)),
        "I-09": lambda values: sorted(values, reverse=True)[:10],
        "I-10": lambda values: statistics.pstdev(values),
    }
    try:
        return solvers[case_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported case id: {case_id}") from exc


def _xor(values: list[int]) -> int:
    result = 0
    for value in values:
        result = operator.xor(result, value)
    return result


def _same_answer(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.0051)
    return actual == expected


def _global_answer(data: dict[str, Any]) -> Any:
    answer = data["expected_output"]
    if isinstance(answer, dict):
        answer = answer.get("per_agent_values", [None])[0]
    return answer


def audit(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    case_id = data["case_id"]
    solver = _local_solver(case_id)
    expected = _global_answer(data)
    agents = []
    for config in data["agent_configs"]:
        local_answer = solver(config["input_shard"])
        agents.append(
            {
                "agent_id": config["agent_id"],
                "local_answer": local_answer,
                "locally_sufficient": _same_answer(local_answer, expected),
            }
        )
    return {
        "file": path.name,
        "case_id": case_id,
        "case_name": data["case_name"],
        "agent_count": len(agents),
        "global_answer": expected,
        "agents": agents,
        "all_agents_locally_sufficient": all(
            agent["locally_sufficient"] for agent in agents
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="I-*_n2.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show-local-answers", action="store_true")
    args = parser.parse_args()

    rows = [audit(Path(filename)) for filename in sorted(glob.glob(str(args.suite_dir / args.pattern)))]
    if not rows:
        raise SystemExit(f"No benchmark instances matched {args.suite_dir / args.pattern}")

    lines = [
        "# Silo-Bench local-answer sufficiency audit",
        "",
        "This instance audit evaluates each agent's exact local result under the task's stated operation. It does not model prior-based guessing or claim that a task family is always solvable without communication.",
        "",
        "| Instance | Agents | Task | Global answer | Locally sufficient agents | All locally sufficient? |",
        "|---|---:|---|---:|---|:---:|",
    ]
    for row in rows:
        locally_sufficient = [
            str(agent["agent_id"])
            for agent in row["agents"]
            if agent["locally_sufficient"]
        ]
        local = (
            ", ".join(
                f"agent {agent['agent_id']}: `{json.dumps(agent['local_answer'])}`"
                for agent in row["agents"]
            )
            if args.show_local_answers
            else f"{len(locally_sufficient)}/{row['agent_count']} agents"
            + (f" (IDs: {', '.join(locally_sufficient)})" if locally_sufficient else "")
        )
        lines.append(
            f"| `{row['file']}` | {row['agent_count']} | {row['case_name']} | `{json.dumps(row['global_answer'])}` | {local} | {row['all_agents_locally_sufficient']} |"
        )
    sufficient = [row["file"] for row in rows if row["all_agents_locally_sufficient"]]
    lines.extend(
        [
            "",
            f"Instances where every agent's exact local answer equals the global answer: {', '.join(f'`{name}`' for name in sufficient) if sufficient else 'none'}.",
            "",
            f"This audit matched {len(rows)} fixed Paradigm-I instances using pattern `{args.pattern}`. It checks exact local-task operations only; other paradigms and prior-based guessing are outside this audit.",
        ]
    )
    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
