"""Optional GEPA adapter for train-only prompt optimization in v0.4.

GEPA is deliberately not a core dependency. Importing this module performs no
model calls; evaluation requires injected chat clients and a finite request
budget. Each GEPA example is one complete candidate-set cluster, preserving
the experimental unit used by the v0.4 runner.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.emergent_ood_v0_4.nl_feedback import _load_train_bundle
from experiments.emergent_ood_v0_4.runner import run_condition
from tacit.runtime import ChatCompletion, ChatModel


@dataclass(frozen=True)
class CandidateSetCluster:
    """One train-only cluster of aligned sender/receiver/gold episodes."""

    candidate_set_id: str
    episodes: tuple[dict[str, Any], ...]
    task_seed: int


@dataclass(frozen=True)
class CallRecord:
    sequence: int
    role: str
    outcome: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    service_seconds: float | None = None


class RequestBudgetExceeded(RuntimeError):
    """Raised before dispatch when the shared task/reflection request cap is spent."""


class RequestLedger:
    """Thread-safe hard cap counting actual client dispatch attempts."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("request budget must be a positive integer")
        self.limit = limit
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()

    @property
    def records(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def used(self) -> int:
        with self._lock:
            return len(self._records)

    def dispatch(self, role: str, client: ChatModel, messages: Sequence[Mapping[str, str]]) -> ChatCompletion:
        with self._lock:
            if len(self._records) >= self.limit:
                raise RequestBudgetExceeded(f"request budget of {self.limit} has been exhausted")
            sequence = len(self._records) + 1
            self._records.append(CallRecord(sequence, role, "dispatched"))
        try:
            result = client.complete(messages)
        except BaseException:
            with self._lock:
                self._records[sequence - 1] = CallRecord(sequence, role, "failed")
            raise
        with self._lock:
            self._records[sequence - 1] = CallRecord(
                sequence, role, "completed", result.input_tokens,
                result.output_tokens, result.service_seconds,
            )
        return result


@dataclass
class BudgetedChatModel:
    """Count all dispatched task-model requests, including failed attempts."""

    client: ChatModel
    ledger: RequestLedger
    role: str
    model: str = field(init=False)

    def __post_init__(self) -> None:
        if self.role not in {"sender", "receiver", "reflection"}:
            raise ValueError("role must be sender, receiver, or reflection")
        self.model = str(getattr(self.client, "model", getattr(self.client, "model_name", "injected-client")))

    def complete(self, messages: Sequence[Mapping[str, str]]) -> ChatCompletion:
        return self.ledger.dispatch(self.role, self.client, messages)


@dataclass(frozen=True)
class _Trajectory:
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Rollout:
    score: float
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _EvaluationBatch:
    outputs: list[_Rollout]
    scores: list[float]
    trajectories: list[_Trajectory] | None


def load_train_clusters(episode_dir: Path, *, split_seed: int) -> tuple[list[CandidateSetCluster], dict[str, Any]]:
    """Read and validate train ledgers only; validation/test files are never opened."""
    bundle, split, _manifest_hash = _load_train_bundle(episode_dir, split_seed=split_seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sender, receiver, gold in zip(bundle["sender"], bundle["receiver"], bundle["gold"]):
        cluster_id = gold["candidate_set_id"]
        grouped.setdefault(cluster_id, []).append({"sender": sender, "receiver": receiver, "gold": gold})
    task_seed = bundle["manifest"].get("task_seed")
    if isinstance(task_seed, bool) or not isinstance(task_seed, int) or task_seed < 0:
        raise ValueError("training manifest has an invalid task seed")
    clusters = [CandidateSetCluster(key, tuple(rows), task_seed) for key, rows in grouped.items()]
    expected_k = bundle["manifest"].get("k")
    if not clusters or any(len(cluster.episodes) != expected_k for cluster in clusters):
        raise ValueError("training candidate-set clusters must be non-empty and complete")
    return clusters, split


class TacitGEPAAdapter:
    """GEPAAdapter-compatible train evaluator for v0.4 plain-English cards.

    The candidate has exactly two mutable string components. GEPA's evaluator
    score is the mean exact candidate-selection rate within a complete set.
    Reflection models must be wrapped with ``BudgetedChatModel`` using the
    same ledger to share the endpoint cap.
    """

    def __init__(
        self, *, episode_dir: Path, split_seed: int, sender_model: ChatModel,
        receiver_model: ChatModel, request_budget: int,
        model_population_id: str = "local-unspecified",
    ) -> None:
        self.clusters, self.split = load_train_clusters(episode_dir, split_seed=split_seed)
        self.split_seed = split_seed
        self.ledger = RequestLedger(request_budget)
        self.sender_model = BudgetedChatModel(sender_model, self.ledger, "sender")
        self.receiver_model = BudgetedChatModel(receiver_model, self.ledger, "receiver")
        self.model_population_id = model_population_id
        self._by_id = {cluster.candidate_set_id: cluster for cluster in self.clusters}

    def wrap_reflection_model(self, client: ChatModel) -> BudgetedChatModel:
        """Wrap the GEPA proposer LM so its requests share the same hard cap."""
        return BudgetedChatModel(client, self.ledger, "reflection")

    def get_adapter_state(self) -> dict[str, Any]:
        """Checkpoint actual dispatches so optimizer resume cannot reset its cap."""
        return {
            "schema": "tlu.gepa-request-ledger.v1",
            "limit": self.ledger.limit,
            "records": [record.__dict__.copy() for record in self.ledger.records],
        }

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("schema") != "tlu.gepa-request-ledger.v1":
            raise ValueError("unsupported GEPA adapter checkpoint")
        if state.get("limit") != self.ledger.limit:
            raise ValueError("cannot resume GEPA with a different request budget")
        raw_records = state.get("records")
        if not isinstance(raw_records, list) or len(raw_records) > self.ledger.limit:
            raise ValueError("GEPA checkpoint request ledger is malformed")
        restored: list[CallRecord] = []
        for index, row in enumerate(raw_records, start=1):
            if (
                not isinstance(row, dict) or row.get("sequence") != index
                or row.get("role") not in {"sender", "receiver", "reflection"}
                or row.get("outcome") not in {"dispatched", "failed", "completed"}
            ):
                raise ValueError("GEPA checkpoint request ledger is malformed")
            restored.append(CallRecord(
                index, row["role"], row["outcome"], row.get("input_tokens"),
                row.get("output_tokens"), row.get("service_seconds"),
            ))
        with self.ledger._lock:
            self.ledger._records = restored

    @staticmethod
    def _candidate(candidate: Mapping[str, Any]) -> tuple[str, str]:
        if not isinstance(candidate, Mapping) or set(candidate) != {"sender_instruction", "receiver_instruction"}:
            raise ValueError("candidate must contain only sender_instruction and receiver_instruction")
        sender, receiver = candidate["sender_instruction"], candidate["receiver_instruction"]
        if not isinstance(sender, str) or not sender.strip() or not isinstance(receiver, str) or not receiver.strip():
            raise ValueError("candidate instructions must be non-empty strings")
        return sender, receiver

    def evaluate(self, batch: Sequence[CandidateSetCluster], candidate: Mapping[str, Any], capture_traces: bool = False) -> _EvaluationBatch:
        sender_instruction, receiver_instruction = self._candidate(candidate)
        scores: list[float] = []
        outputs: list[_Rollout] = []
        trajectories: list[_Trajectory] = []
        card = {
            "protocol_id": "gepa-train-candidate",
            "sender_instruction": (
                "Communicate the private meaning in grammatical ordinary English. Include each attribute name "
                "and its exact value. Do not use JSON, code, symbolic shorthand, lookup tables, or candidate IDs. "
                "The receiver does not know a private code.\nCandidate style guidance:\n" + sender_instruction
            ),
            "receiver_instruction": (
                "Interpret one ordinary-English message as attribute names and exact values. Do not assume a private "
                "code or consult information outside the message and candidate table.\nCandidate decoding guidance:\n"
                + receiver_instruction
            ),
        }
        for supplied in batch:
            if not isinstance(supplied, CandidateSetCluster) or supplied.candidate_set_id not in self._by_id:
                raise ValueError("GEPA may evaluate only clusters loaded from this adapter's train split")
            expected = self._by_id[supplied.candidate_set_id]
            if supplied != expected:
                raise ValueError("candidate-set cluster does not match the adapter's validated train data")
            results: list[dict[str, Any]] = []
            for episode in supplied.episodes:
                try:
                    result = run_condition(
                        episode=episode, condition="shared_protocol_card", stage="train",
                        sender_model=self.sender_model, receiver_model=self.receiver_model,
                        attributes=self.split["attributes"], values=self.split["values_by_attribute"],
                        protocol_card=card, split_seed=self.split_seed,
                        task_seed=supplied.task_seed,
                        model_population_id=self.model_population_id,
                    )
                except RequestBudgetExceeded:
                    raise
                except Exception as exc:
                    result = {"outcome": {"exact_selection": False}, "trace": {},
                              "error": f"{type(exc).__name__}: {exc}"}
                results.append(result)
            score = sum(float(row["outcome"].get("exact_selection") is True) for row in results) / len(results)
            frozen_rows = tuple(results)
            scores.append(score)
            outputs.append(_Rollout(score, frozen_rows if capture_traces else ()))
            if capture_traces:
                trajectories.append(_Trajectory(frozen_rows))
        # GEPA's class is optional; the local dataclass has its documented attributes
        # and keeps dry-run/fake-client use possible without installing GEPA.
        try:
            from gepa.core.adapter import EvaluationBatch as GepaEvaluationBatch
        except ImportError:
            return _EvaluationBatch(outputs, scores, trajectories if capture_traces else None)
        return GepaEvaluationBatch(
            outputs=outputs, scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self, candidate: Mapping[str, Any], eval_batch: _EvaluationBatch,
        components_to_update: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        self._candidate(candidate)
        allowed = {"sender_instruction", "receiver_instruction"}
        if any(component not in allowed for component in components_to_update):
            raise ValueError("unknown GEPA component requested for reflection")
        if eval_batch.trajectories is None:
            raise ValueError("reflective dataset requires an evaluation with capture_traces=True")
        dataset: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            examples: list[dict[str, Any]] = []
            for trajectory in eval_batch.trajectories:
                for row in trajectory.rows:
                    examples.append({
                        "stage": "train",
                        "private_input": row["trace"].get("target_tuple_for_evaluator"),
                        "message": row["trace"].get("message"),
                        "receiver_input_candidate_ids": row["trace"].get("candidate_ids_in_receiver_order"),
                        "target_candidate_id": row["outcome"].get("target_candidate_id"),
                        "predicted_candidate_id": row["outcome"].get("answer_candidate_id"),
                        "exact_selection": row["outcome"].get("exact_selection"),
                        "component": component,
                        "Feedback": (
                            row.get("error") or (
                                "Correct exact selection." if row["outcome"].get("exact_selection") is True
                                else f"Expected {row['outcome'].get('target_candidate_id')}; "
                                     f"received {row['outcome'].get('answer_candidate_id')}."
                            )
                        ),
                    })
            dataset[component] = examples
        return dataset


def dry_run(episode_dir: Path, split_seed: int) -> dict[str, Any]:
    clusters, _split = load_train_clusters(episode_dir, split_seed=split_seed)
    return {
        "mode": "offline-dry-run",
        "training_clusters": len(clusters),
        "episodes_per_cluster": len(clusters[0].episodes),
        "requests_per_candidate_per_cluster": 2 * len(clusters[0].episodes),
        "validation_or_test_rows_loaded": 0,
        "model_requests": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GEPA train-only v0.4 inputs without model calls")
    parser.add_argument("--episodes-dir", type=Path, default=Path(".cache/emergent_ood_v0_4/episodes"))
    parser.add_argument("--split-seed", type=int, default=17)
    args = parser.parse_args()
    import json
    print(json.dumps(dry_run(args.episodes_dir, args.split_seed), sort_keys=True))


if __name__ == "__main__":
    main()
