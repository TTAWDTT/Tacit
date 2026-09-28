"""Export sanitized DuoSum v0.5/v0.6 outcomes and paired episode bootstrap.

The source run CSVs and message traces stay under ignored ``.cache``. The
published JSONL contains only task IDs, condition IDs, and aggregate outcomes
and costs; it never contains prompts, messages, submissions, or private shards.
"""

from __future__ import annotations

import importlib.util
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT / "research" / "data" / "DUOSUM_PAIRED_OUTCOMES_V0_5_V0_6.jsonl"
REPORT_OUT = ROOT / "research" / "DUOSUM_PAIRED_ANALYSIS_V0_1.md"
CONDITIONS = (
    "scaffold_only",
    "concise_nl",
    "compact_kv",
    "json_schema",
    "binary",
    "autoform",
    "no_communication",
)
METRICS = (
    "strict_agent_rate",
    "joint_strict_success",
    "semantic_agent_rate",
    "joint_semantic_success",
    "payload_bytes",
    "serialized_message_file_bytes",
    "model_tokens",
    "message_count",
    "elapsed_seconds",
)
PRIMARY_CONDITION = "compact_kv"
COMPARATORS = tuple(name for name in CONDITIONS if name != PRIMARY_CONDITION)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 1729


def _load_analyzer(version: str) -> list[dict[str, Any]]:
    analyzer_path = ROOT / "experiments" / version / "analyze_duosum_heldout.py"
    # v0.5/v0.6 use the same analyzer function names and differ only in the
    # frozen source CSV and policy paths defined by their own module.
    spec = importlib.util.spec_from_file_location(f"_{version}_duosum_analyzer", analyzer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load analyzer: {analyzer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_rows()


def _sanitize(rows_by_version: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    expected_task_sets: dict[str, set[str]] = {}
    expected_manifests: dict[str, set[str]] = {}
    expected_engine_commits: dict[str, set[str]] = {}
    sanitized: list[dict[str, Any]] = []
    for version, rows in rows_by_version.items():
        keyed: dict[tuple[str, str], dict[str, Any]] = {}
        tasks: set[str] = set()
        seen_conditions: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            task = Path(str(row["task"])).name
            condition = str(row["condition"])
            key = (task, condition)
            if key in keyed:
                raise ValueError(f"duplicate run cell: {version} {task} {condition}")
            keyed[key] = row
            tasks.add(task)
            seen_conditions[task].add(condition)
        expected_task_sets[version] = tasks
        model_revisions = {str(row["model_revision"]) for row in rows}
        expected_manifests[version] = {str(row["task_manifest_sha256"]) for row in rows}
        expected_engine_commits[version] = {str(row["upstream_engine_commit"]) for row in rows}
        if (
            len(tasks) != 8
            or len(model_revisions) != 1
            or len(expected_manifests[version]) != 1
            or len(expected_engine_commits[version]) != 1
        ):
            raise ValueError(f"expected eight tasks and single pinned model/benchmark/engine revisions in {version}")
        for task in sorted(tasks):
            if seen_conditions[task] != set(CONDITIONS):
                raise ValueError(f"incomplete conditions for {version} {task}")
            for condition in CONDITIONS:
                row = keyed[(task, condition)]
                strict = int(row["strict_correct"])
                semantic = int(row["semantic_correct"])
                if not (0 <= strict <= 2 and 0 <= semantic <= 2):
                    raise ValueError(f"expected exactly two agent outcomes in {version} {task}")
                sanitized.append({
                    "run_id": version,
                    "task_id": task,
                    "condition_id": condition,
                    "model_revision": str(row["model_revision"]),
                    "task_manifest_sha256": str(row["task_manifest_sha256"]),
                    "engine_commit": str(row["upstream_engine_commit"]),
                    "strict_correct_agents": strict,
                    "joint_strict_success": strict == 2,
                    "semantic_correct_agents": semantic,
                    "joint_semantic_success": semantic == 2,
                    "payload_bytes": int(row["message_payload_bytes"]),
                    "serialized_message_file_bytes": int(row["simulator_message_file_bytes"]),
                    "model_tokens": int(row["total_tokens"]),
                    "message_count": int(row["message_count"]),
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                })
    if (
        len(expected_task_sets) != 2
        or expected_task_sets["pilot_v0_5"] != expected_task_sets["pilot_v0_6"]
        or expected_manifests["pilot_v0_5"] != expected_manifests["pilot_v0_6"]
        or expected_engine_commits["pilot_v0_5"] != expected_engine_commits["pilot_v0_6"]
    ):
        raise ValueError("v0.5 and v0.6 must contain the same task IDs, task manifest, and engine commit")
    return sanitized


def _metric(row: dict[str, Any], name: str) -> float:
    if name == "strict_agent_rate":
        return row["strict_correct_agents"] / 2
    if name == "semantic_agent_rate":
        return row["semantic_correct_agents"] / 2
    return float(row[name])


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _paired_delta(
    left: list[dict[str, Any]], right: list[dict[str, Any]], metric: str,
    rng: random.Random,
) -> tuple[float, float, float]:
    right_by_task = {row["task_id"]: row for row in right}
    if len(right_by_task) != len(right) or {row["task_id"] for row in left} != set(right_by_task):
        raise ValueError("paired conditions require unique identical task coverage")
    differences = [
        _metric(row, metric) - _metric(right_by_task[row["task_id"]], metric)
        for row in left
    ]
    point = statistics.mean(differences)
    draws = [
        statistics.mean(rng.choice(differences) for _ in differences)
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    draws.sort()
    return point, _quantile(draws, 0.025), _quantile(draws, 0.975)


def _report(sanitized: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sanitized:
        groups[(row["run_id"], row["condition_id"])].append(row)
    lines = [
        "# DuoSum v0.5/v0.6 paired reanalysis",
        "",
        "This analysis uses the eight shared task IDs per run and resamples at the task level, keeping the two submissions in an episode together. The public ledger records the task-manifest SHA-256, upstream engine commit, and pinned model revision for lineage; the original condition policies are [v0.5](../experiments/pilot_v0_5/policies.json) and [v0.6](../experiments/pilot_v0_6/policies.json). It is descriptive: eight tasks provide very low precision, and the two model runs differ in size, quantization, backend, and chat template. No format-superiority claim follows.",
        "",
        "Rebuild the sanitized episode ledger and this report from the local ignored run traces with `python research/analyze_duosum_paired.py`; the script makes no model requests.",
        "",
        "Differences are `compact_kv − comparator`; cost differences below zero favor compact-KV on that measure. `payload_bytes` are message-content UTF-8 bytes as recorded by the existing pilot; `serialized_message_file_bytes` include the simulator's per-message JSON files. Neither includes the full repeated prompt/context, so these are channel-content diagnostics, not total wire-cost frontiers.",
        "",
        f"Paired percentile bootstrap: {BOOTSTRAP_REPLICATES:,} episode-cluster resamples, seed {BOOTSTRAP_SEED}; intervals are exploratory with n=8.",
        "",
    ]
    for version in ("pilot_v0_5", "pilot_v0_6"):
        lines.extend([
            f"## {version}",
            "",
            "| Paired metric | Comparator | Mean difference | 95% percentile interval |",
            "|---|---|---:|---:|",
        ])
        rng = random.Random(BOOTSTRAP_SEED)
        for comparator in COMPARATORS:
            left = groups[(version, PRIMARY_CONDITION)]
            right = groups[(version, comparator)]
            for metric in METRICS:
                point, low, high = _paired_delta(left, right, metric, rng)
                lines.append(
                    f"| `{metric}` | `{comparator}` | {point:.4f} | [{low:.4f}, {high:.4f}] |"
                )
        lines.append("")
    lines.extend([
        "## Interpretation boundary",
        "",
        "This reanalysis can quantify paired exploratory differences in the existing runs. It cannot repair instruction non-adherence (the v0.6 JSON arm emitted invalid JSON), separate model/backend changes across v0.5/v0.6, infer population-level superiority from eight tasks, or create a matched-budget frontier. Message format was not the only varying causal factor across conditions. The public JSONL contains sanitized per-task outcomes and cost summaries only; source traces remain ignored under `.cache/`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    rows_by_version = {
        "pilot_v0_5": _load_analyzer("pilot_v0_5"),
        "pilot_v0_6": _load_analyzer("pilot_v0_6"),
    }
    sanitized = _sanitize(rows_by_version)
    DATA_OUT.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in sanitized),
        encoding="utf-8",
        newline="\n",
    )
    REPORT_OUT.write_text(_report(sanitized), encoding="utf-8", newline="\n")
    print(f"Wrote {len(sanitized)} sanitized episode-condition records to {DATA_OUT}")
    print(f"Wrote paired analysis to {REPORT_OUT}")


if __name__ == "__main__":
    main()
