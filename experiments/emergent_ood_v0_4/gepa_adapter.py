"""Optional GEPA adapter for train-only prompt optimization in v0.4.

GEPA is deliberately not a core dependency. Importing this module performs no
model calls; evaluation requires injected chat clients and a finite request
budget. Each GEPA example is one complete candidate-set cluster, preserving
the experimental unit used by the v0.4 runner.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.emergent_ood_v0_4.nl_feedback import _load_train_bundle
from experiments.emergent_ood_v0_4.runner import (
    CAPABILITY_CALIBRATION_SETS,
    _endpoint_port,
    _loopback_url,
    load_episode_bundle,
    run_condition,
    select_candidate_sets,
    validate_capability_ledger,
)
from experiments.emergent_ood_v0_4.episodes import PROJECT_ROOT
from experiments.emergent_ood_v0_4.split import split_task_id
from experiments.emergent_ood_v0_3.runner import validate_resource_preflight
from tacit.protocol import ProtocolCard
from tacit.runtime import ChatCompletion, ChatModel, OpenAICompatibleClient


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
    input_message_bytes: int = 0
    completion_bytes: int | None = None


class RequestBudgetExceeded(RuntimeError):
    """Raised before dispatch when the shared task/reflection request cap is spent."""


class RequestLedger:
    """Thread-safe hard cap counting actual client dispatch attempts."""

    def __init__(self, limit: int, *, path: Path | None = None) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("request budget must be a positive integer")
        self.limit = limit
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()
        self.path: Path | None = None
        if path is not None:
            resolved = path.resolve()
            try:
                resolved.relative_to(PROJECT_ROOT)
            except ValueError as exc:
                raise ValueError("request ledger path must stay inside the project") from exc
            self.path = resolved
            if self.path.exists():
                self._records = self._read_records(self.path.read_bytes(), expected_limit=limit)

    @property
    def records(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def used(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.limit - len(self._records)

    def require_capacity(self, calls: int) -> None:
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError("required call capacity must be a non-negative integer")
        with self._lock:
            if calls > self.limit - len(self._records):
                raise RequestBudgetExceeded(
                    f"next complete task batch requires {calls} requests; "
                    f"only {self.limit - len(self._records)} remain"
                )

    @staticmethod
    def _validate_record(row: Any, *, sequence: int) -> CallRecord:
        if (
            not isinstance(row, dict)
            or isinstance(row.get("sequence"), bool)
            or row.get("sequence") != sequence
            or row.get("role") not in {"sender", "receiver", "reflection"}
            or row.get("outcome") not in {"dispatched", "failed", "completed"}
        ):
            raise ValueError("GEPA request ledger has a malformed entry")
        input_bytes = row.get("input_message_bytes", 0)
        completion_bytes = row.get("completion_bytes")
        if isinstance(input_bytes, bool) or not isinstance(input_bytes, int) or input_bytes < 0:
            raise ValueError("GEPA request ledger has an invalid input byte count")
        if completion_bytes is not None and (
            isinstance(completion_bytes, bool) or not isinstance(completion_bytes, int) or completion_bytes < 0
        ):
            raise ValueError("GEPA request ledger has an invalid completion byte count")
        for field_name in ("input_tokens", "output_tokens"):
            token_count = row.get(field_name)
            if token_count is not None and (
                isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0
            ):
                raise ValueError(f"GEPA request ledger has invalid {field_name}")
        service_seconds = row.get("service_seconds")
        if service_seconds is not None and (
            isinstance(service_seconds, bool)
            or not isinstance(service_seconds, (int, float))
            or not math.isfinite(service_seconds)
            or service_seconds < 0
        ):
            raise ValueError("GEPA request ledger has invalid service_seconds")
        return CallRecord(
            sequence, row["role"], row["outcome"], row.get("input_tokens"),
            row.get("output_tokens"), service_seconds, input_bytes, completion_bytes,
        )

    @staticmethod
    def _read_records(payload: bytes, *, expected_limit: int) -> list[CallRecord]:
        try:
            state = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("durable GEPA request ledger is invalid JSON") from exc
        if (
            not isinstance(state, dict)
            or state.get("schema") != "tlu.gepa-request-ledger.v1"
            or state.get("limit") != expected_limit
            or not isinstance(state.get("records"), list)
            or len(state["records"]) > expected_limit
        ):
            raise ValueError("durable GEPA request ledger is malformed or has a different budget")
        records: list[CallRecord] = []
        for index, row in enumerate(state["records"], start=1):
            try:
                records.append(RequestLedger._validate_record(row, sequence=index))
            except ValueError as exc:
                raise ValueError("durable GEPA request ledger has a malformed entry") from exc
        return records

    def _persist_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        payload = {
            "schema": "tlu.gepa-request-ledger.v1",
            "limit": self.limit,
            "records": [record.__dict__.copy() for record in self._records],
        }
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def restore(self, records: list[CallRecord]) -> None:
        with self._lock:
            if self.path is not None and self.path.exists():
                durable = self._read_records(self.path.read_bytes(), expected_limit=self.limit)
                # A crash can occur after a provider request was reserved but
                # before GEPA's less frequent optimizer checkpoint. The durable
                # ledger is authoritative and may be ahead of that checkpoint.
                if len(durable) > len(records):
                    records = durable
                elif len(durable) == len(records) and durable != records:
                    raise ValueError("GEPA checkpoint disagrees with its durable request ledger")
            if len(records) > self.limit:
                raise ValueError("restored GEPA request ledger exceeds its budget")
            self._records = list(records)
            self._persist_locked()

    def persist(self) -> None:
        with self._lock:
            self._persist_locked()

    def dispatch(self, role: str, client: ChatModel, messages: Sequence[Mapping[str, str]]) -> ChatCompletion:
        input_message_bytes = len(json.dumps(
            list(messages), ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))
        with self._lock:
            if len(self._records) >= self.limit:
                raise RequestBudgetExceeded(f"request budget of {self.limit} has been exhausted")
            sequence = len(self._records) + 1
            self._records.append(CallRecord(
                sequence, role, "dispatched", input_message_bytes=input_message_bytes,
            ))
            try:
                # Reserve the call on disk before crossing the model boundary.
                self._persist_locked()
            except BaseException:
                self._records.pop()
                raise
        try:
            result = client.complete(messages)
            if not isinstance(result, ChatCompletion):
                raise TypeError("ChatModel.complete must return ChatCompletion")
            completed = CallRecord(
                sequence, role, "completed", result.input_tokens,
                result.output_tokens, result.service_seconds,
                input_message_bytes, len(result.text.encode("utf-8")),
            )
        except BaseException:
            with self._lock:
                self._records[sequence - 1] = CallRecord(
                    sequence, role, "failed", input_message_bytes=input_message_bytes,
                )
                self._persist_locked()
            raise
        with self._lock:
            self._records[sequence - 1] = completed
            self._persist_locked()
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

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """Expose GEPA's LanguageModel callable contract for reflection calls."""
        if isinstance(prompt, str):
            messages: Sequence[Mapping[str, str]] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(message, dict) and isinstance(message.get("role"), str)
            and isinstance(message.get("content"), str) for message in prompt
        ):
            messages = prompt
        else:
            raise TypeError("reflection prompt must be text or a list of role/content messages")
        return self.complete(messages).text

    def batch_complete(self, messages_list: list[list[dict[str, Any]]]) -> list[str]:
        """Count each reflection prompt separately; do not use hidden batch retries."""
        return [self(messages) for messages in messages_list]


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


def load_train_clusters(
    episode_dir: Path, *, split_seed: int,
) -> tuple[list[CandidateSetCluster], dict[str, Any], dict[str, Any]]:
    """Read and validate train ledgers only; validation/test files are never opened."""
    bundle, split, manifest_hash = _load_train_bundle(episode_dir, split_seed=split_seed)
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
    if 2 * expected_k > 12:
        raise ValueError("each candidate-set evaluation must remain within the frozen 12-request task-batch limit")
    manifest = bundle["manifest"]
    metadata = {
        "manifest_sha256": manifest_hash,
        "split_sha256": manifest["split_sha256"],
        "task_seed": task_seed,
        "task_key_id": manifest["task_key_id"],
        "candidate_count": manifest["k"],
        "ontology_id": manifest.get("ontology_id"),
    }
    return clusters, split, metadata


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
        request_ledger_path: Path | None = None,
        model_population_id: str = "local-unspecified",
    ) -> None:
        self.clusters, self.split, self.train_metadata = load_train_clusters(
            episode_dir, split_seed=split_seed,
        )
        self.split_seed = split_seed
        self.ledger = RequestLedger(request_budget, path=request_ledger_path)
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
            try:
                restored.append(RequestLedger._validate_record(row, sequence=index))
            except ValueError as exc:
                raise ValueError("GEPA checkpoint request ledger is malformed") from exc
        self.ledger.restore(restored)

    @staticmethod
    def _candidate(candidate: Mapping[str, Any]) -> tuple[str, str]:
        if not isinstance(candidate, Mapping) or set(candidate) != {"sender_instruction", "receiver_instruction"}:
            raise ValueError("candidate must contain only sender_instruction and receiver_instruction")
        sender, receiver = candidate["sender_instruction"], candidate["receiver_instruction"]
        if not isinstance(sender, str) or not sender.strip() or not isinstance(receiver, str) or not receiver.strip():
            raise ValueError("candidate instructions must be non-empty strings")
        if len(sender.encode("utf-8")) > 32_768 or len(receiver.encode("utf-8")) > 32_768:
            raise ValueError("candidate instructions exceed the portable protocol-card limit")
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
            # Finish or skip a complete candidate-set task batch; never consume
            # part of a cluster and leave a misleading partial score.
            self.ledger.require_capacity(2 * len(supplied.episodes))
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
                        "Inputs": {
                            "private_meaning": row["trace"].get("target_tuple_for_evaluator"),
                            "candidate_ids": row["trace"].get("candidate_ids_in_receiver_order"),
                        },
                        "Generated Outputs": {
                            "sender_message": row["trace"].get("message"),
                            "receiver_candidate_id": row["outcome"].get("answer_candidate_id"),
                        },
                        "Feedback": (
                            row.get("error") or (
                                "Correct exact selection." if row["outcome"].get("exact_selection") is True
                                else f"Expected {row['outcome'].get('target_candidate_id')}; "
                                     f"received {row['outcome'].get('answer_candidate_id')} from the receiver."
                            )
                        ),
                        "stage": "train",
                        "private_input": row["trace"].get("target_tuple_for_evaluator"),
                        "message": row["trace"].get("message"),
                        "receiver_input_candidate_ids": row["trace"].get("candidate_ids_in_receiver_order"),
                        "target_candidate_id": row["outcome"].get("target_candidate_id"),
                        "predicted_candidate_id": row["outcome"].get("answer_candidate_id"),
                        "exact_selection": row["outcome"].get("exact_selection"),
                        "component": component,
                    })
            dataset[component] = examples
        return dataset


class _RequestBudgetStopper:
    def __init__(self, adapter: TacitGEPAAdapter) -> None:
        self.adapter = adapter

    def __call__(self, _gepa_state: Any) -> bool:
        return self.adapter.ledger.used >= self.adapter.ledger.limit


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("GEPA outputs and preflight inputs must stay inside the project") from exc
    return resolved


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _provider_endpoint(client: ChatModel) -> tuple[str, int]:
    if not isinstance(client, OpenAICompatibleClient):
        raise ValueError("GEPA execution currently requires OpenAI-compatible loopback clients")
    endpoint = _loopback_url(client.base_url)
    return endpoint, _endpoint_port(endpoint)


def minimum_search_budget(clusters: Sequence[CandidateSetCluster]) -> dict[str, int]:
    """Calculate calls for initial train scoring plus one 3-cluster mutation."""
    if len(clusters) < 3:
        raise ValueError("GEPA prompt search requires at least three train candidate-set clusters")
    bootstrap = sum(2 * len(cluster.episodes) for cluster in clusters)
    first_minibatch_episodes = sum(len(cluster.episodes) for cluster in clusters[:3])
    # One parent task minibatch, one child minibatch, and one reflected component.
    mutation_task_calls = 4 * first_minibatch_episodes
    reflection_calls = 1
    return {
        "initial_train_evaluation_requests": bootstrap,
        "first_mutation_task_requests": mutation_task_calls,
        "first_reflection_requests": reflection_calls,
        "minimum_requests_for_one_mutation": bootstrap + mutation_task_calls + reflection_calls,
        "initial_train_metric_calls": len(clusters),
        "first_mutation_metric_calls": 6,
        "minimum_metric_calls_for_one_mutation": len(clusters) + 6,
    }


def run_gepa_optimization(
    adapter: TacitGEPAAdapter,
    *,
    reflection_client: ChatModel,
    run_dir: Path,
    preflight_path: Path,
    capability_episode_dir: Path,
    capability_split_seed: int,
    capability_ledger_path: Path,
    receiver_tokenizer_id: str,
    sender_tokenizer_id: str,
    reflection_tokenizer_id: str,
    max_metric_calls: int,
    seed: int,
    seed_candidate: Mapping[str, str],
    resume: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Run pinned GEPA only after the independent receiver and host gates pass.

    All state and outputs stay under the project's ignored `.cache` directory.
    The primary optimization's valset is deliberately omitted so GEPA reuses
    train examples rather than reading held-out validation/test ledgers.
    """
    if isinstance(max_metric_calls, bool) or not isinstance(max_metric_calls, int) or max_metric_calls < 1:
        raise ValueError("max_metric_calls must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(resume, bool):
        raise ValueError("resume must be boolean")
    TacitGEPAAdapter._candidate(seed_candidate)
    budget_floor = minimum_search_budget(adapter.clusters)
    if adapter.ledger.limit < budget_floor["minimum_requests_for_one_mutation"]:
        raise ValueError(
            "request_budget is below the minimum for one complete three-cluster mutation: "
            f"need {budget_floor['minimum_requests_for_one_mutation']} requests"
        )
    if max_metric_calls < budget_floor["minimum_metric_calls_for_one_mutation"]:
        raise ValueError(
            "max_metric_calls is below the initial train score plus one complete mutation: "
            f"need {budget_floor['minimum_metric_calls_for_one_mutation']} metric calls"
        )
    if not isinstance(run_dir, Path):
        raise ValueError("run_dir must be a pathlib.Path")
    resolved_run_dir = _inside_project(run_dir)
    runs_root = (PROJECT_ROOT / ".cache" / "emergent_ood_v0_4" / "gepa_runs").resolve()
    try:
        resolved_run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("GEPA run_dir must be inside .cache/emergent_ood_v0_4/gepa_runs") from exc
    expected_ledger_path = resolved_run_dir / "request-ledger.json"
    if adapter.ledger.path != expected_ledger_path:
        raise ValueError("adapter request ledger must be run_dir/request-ledger.json for crash-safe resumption")
    if resume and (not resolved_run_dir.is_dir() or not expected_ledger_path.is_file()):
        raise ValueError("resume requires an existing GEPA run directory and durable request ledger")
    if not resume and resolved_run_dir.exists():
        raise FileExistsError(f"refusing to reuse existing GEPA run directory: {resolved_run_dir}")

    # The calibrated receiver must already pass on three independent, balanced
    # train-only candidate sets, matching the existing v0.4 execution gate.
    calibration_clusters, calibration_split, calibration_meta = load_train_clusters(
        capability_episode_dir, split_seed=capability_split_seed,
    )
    expected_calibration_episodes = [episode for cluster in calibration_clusters for episode in cluster.episodes]
    evaluation_episodes = [episode for cluster in adapter.clusters for episode in cluster.episodes]
    if len(calibration_clusters) != CAPABILITY_CALIBRATION_SETS:
        raise ValueError(f"receiver screen bundle must contain exactly {CAPABILITY_CALIBRATION_SETS} train sets")
    calibration_manifest = {
        "task_seed": calibration_clusters[0].task_seed,
        "task_key_id": calibration_meta["task_key_id"],
    }
    validate_capability_ledger(
        _inside_project(capability_ledger_path),
        expected_calibration_episodes=expected_calibration_episodes,
        evaluation_episodes=evaluation_episodes,
        input_manifest_sha256=calibration_meta["manifest_sha256"],
        split_seed=capability_split_seed,
        split_sha256=calibration_meta["split_sha256"],
        evaluation_split_seed=adapter.split_seed,
        task_seed=calibration_manifest["task_seed"],
        task_key_id=calibration_manifest["task_key_id"],
        receiver_model=adapter.receiver_model.model,
        receiver_tokenizer_id=receiver_tokenizer_id,
        model_population_id=adapter.model_population_id,
        task_id=split_task_id(adapter.split),
        train_target_support_size=len(adapter.split["train_meaning_ids"]),
        ontology_id=adapter.split.get("ontology_id"),
    )

    providers = {
        "sender": adapter.sender_model.client,
        "receiver": adapter.receiver_model.client,
        "reflection": reflection_client,
    }
    provider_endpoints = {name: _provider_endpoint(client) for name, client in providers.items()}
    preflight = _inside_project(preflight_path)
    validate_resource_preflight(
        preflight,
        required_ports={port for _endpoint, port in provider_endpoints.values()},
    )

    try:
        gepa_version = importlib.metadata.version("gepa")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('install the optional pinned dependency with `pip install "gepa==0.1.4"`') from exc
    if gepa_version != "0.1.4":
        raise RuntimeError(f"this adapter was audited for gepa==0.1.4, found {gepa_version}")
    try:
        from gepa import optimize
    except ImportError as exc:
        raise RuntimeError("could not import the pinned GEPA optimize API") from exc

    capability_path = _inside_project(capability_ledger_path)
    preflight_sha256 = hashlib.sha256(preflight.read_bytes()).hexdigest()
    capability_sha256 = hashlib.sha256(capability_path.read_bytes()).hexdigest()
    capability_manifest_path = capability_path.with_suffix(capability_path.suffix + ".manifest.json")
    capability_manifest_sha256 = hashlib.sha256(capability_manifest_path.read_bytes()).hexdigest()
    config: dict[str, Any] = {
        "schema": "tlu.gepa-run-config.v1",
        "gepa_version": gepa_version,
        "seed": seed,
        "max_metric_calls": max_metric_calls,
        "request_budget": adapter.ledger.limit,
        "split_seed": adapter.split_seed,
        "training_split_sha256": adapter.train_metadata["split_sha256"],
        "training_episode_manifest_sha256": adapter.train_metadata["manifest_sha256"],
        "candidate_sets": len(adapter.clusters),
        "episodes_per_candidate_set": len(adapter.clusters[0].episodes),
        "task_requests_per_candidate_set": 2 * len(adapter.clusters[0].episodes),
        "budget_floor": budget_floor,
        "sender_model": adapter.sender_model.model,
        "receiver_model": adapter.receiver_model.model,
        "reflection_model": str(getattr(reflection_client, "model", "unknown")),
        "sender_tokenizer_id": sender_tokenizer_id,
        "receiver_tokenizer_id": receiver_tokenizer_id,
        "reflection_tokenizer_id": reflection_tokenizer_id,
        "model_population_id": adapter.model_population_id,
        "provider_endpoints": {name: endpoint for name, (endpoint, _port) in provider_endpoints.items()},
        "provider_ports": {name: port for name, (_endpoint, port) in provider_endpoints.items()},
        "capability_ledger_sha256": capability_sha256,
        "capability_manifest_sha256": capability_manifest_sha256,
        "capability_split_seed": capability_split_seed,
        "capability_split_sha256": calibration_meta["split_sha256"],
        "capability_episode_manifest_sha256": calibration_meta["manifest_sha256"],
        "seed_candidate_sha256": hashlib.sha256(
            json.dumps(dict(seed_candidate), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    config_path = resolved_run_dir / "run-config.json"
    config_digest = hashlib.sha256(json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    gate_history_path = resolved_run_dir / "gate-history.json"
    if resume:
        try:
            prior_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("resume run has no valid run-config.json") from exc
        if prior_config != config:
            raise ValueError("resume settings or source hashes differ from the original GEPA run")
        try:
            gate_history = json.loads(gate_history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("resume run has no valid gate-history.json") from exc
        if (
            not isinstance(gate_history, dict)
            or gate_history.get("schema") != "tlu.gepa-gate-history.v1"
            or gate_history.get("run_config_sha256") != config_digest
            or not isinstance(gate_history.get("preflight_sha256s"), list)
            or any(not isinstance(value, str) or len(value) != 64 for value in gate_history["preflight_sha256s"])
        ):
            raise ValueError("resume gate history is malformed or belongs to different run settings")
    else:
        _write_json_atomic(config_path, config)
        gate_history = {
            "schema": "tlu.gepa-gate-history.v1",
            "run_config_sha256": config_digest,
            "preflight_sha256s": [],
        }
    if preflight_sha256 not in gate_history["preflight_sha256s"]:
        gate_history["preflight_sha256s"].append(preflight_sha256)
    _write_json_atomic(gate_history_path, gate_history)

    # Create an empty durable ledger before GEPA's first checkpoint or model
    # request. A crash after a request reservation therefore cannot reset usage.
    adapter.ledger.persist()

    result = optimize(
        seed_candidate=dict(seed_candidate),
        trainset=adapter.clusters,
        valset=None,
        adapter=adapter,
        reflection_lm=adapter.wrap_reflection_model(reflection_client),
        max_metric_calls=max_metric_calls,
        stop_callbacks=_RequestBudgetStopper(adapter),
        run_dir=str(resolved_run_dir),
        seed=seed,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        batch_sampler="epoch_shuffled",
        reflection_minibatch_size=3,
        use_merge=False,
        cache_evaluation=False,
        track_best_outputs=False,
        display_progress_bar=False,
        raise_on_exception=True,
    )
    best_candidate = result.best_candidate
    sender_instruction, receiver_instruction = TacitGEPAAdapter._candidate(best_candidate)
    best_card = ProtocolCard(
        protocol_id=f"gepa-v0.1.4-s{adapter.split_seed}-r{seed}",
        sender_instruction=sender_instruction,
        receiver_instruction=receiver_instruction,
    )
    card_path = resolved_run_dir / "best-training-card.json"
    card_bytes = best_card.to_json_bytes() + b"\n"
    card_tmp = card_path.with_name(card_path.name + ".tmp")
    try:
        card_tmp.write_bytes(card_bytes)
        os.replace(card_tmp, card_path)
    finally:
        card_tmp.unlink(missing_ok=True)
    ledger_bytes = expected_ledger_path.read_bytes()
    records = adapter.ledger.records
    def sum_known(name: str) -> int | float | None:
        values = [getattr(record, name) for record in records]
        return sum(values) if all(value is not None for value in values) else None

    result_manifest = {
        **config,
        "schema": "tlu.gepa-run-manifest.v1",
        "preflight_sha256": preflight_sha256,
        "preflight_sha256s": gate_history["preflight_sha256s"],
        "stage": "train_only_optimization",
        "validation_rows_opened": 0,
        "test_rows_opened": 0,
        "best_candidate_index": result.best_idx,
        "best_training_mean_exact_selection": result.val_aggregate_scores[result.best_idx],
        "gepa_candidates": result.num_candidates,
        "gepa_metric_calls": result.total_metric_calls,
        "actual_request_count": len(records),
        "actual_request_counts_by_role": {
            role: sum(record.role == role for record in records)
            for role in ("sender", "receiver", "reflection")
        },
        "failed_request_count": sum(record.outcome == "failed" for record in records),
        "input_tokens": sum_known("input_tokens"),
        "output_tokens": sum_known("output_tokens"),
        "service_seconds": sum_known("service_seconds"),
        "model_input_message_bytes": sum(record.input_message_bytes for record in records),
        "model_completion_bytes": (
            sum(record.completion_bytes for record in records)
            if all(record.completion_bytes is not None for record in records) else None
        ),
        "model_input_message_bytes_by_role": {
            role: sum(record.input_message_bytes for record in records if record.role == role)
            for role in ("sender", "receiver", "reflection")
        },
        "model_completion_bytes_by_role": {
            role: (
                sum(record.completion_bytes for record in records if record.role == role)
                if all(record.completion_bytes is not None for record in records if record.role == role)
                else None
            )
            for role in ("sender", "receiver", "reflection")
        },
        "request_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "best_card_file": card_path.name,
        "best_card_sha256": hashlib.sha256(card_bytes).hexdigest(),
        "best_card_training_score_only": True,
    }
    _write_json_atomic(resolved_run_dir / "result-manifest.json", result_manifest)
    return result, result_manifest


def dry_run(episode_dir: Path, split_seed: int) -> dict[str, Any]:
    clusters, _split, metadata = load_train_clusters(episode_dir, split_seed=split_seed)
    return {
        "mode": "offline-dry-run",
        "training_clusters": len(clusters),
        "episodes_per_cluster": len(clusters[0].episodes),
        "requests_per_candidate_per_cluster": 2 * len(clusters[0].episodes),
        "minimum_budgets_for_one_mutation": minimum_search_budget(clusters),
        "validation_or_test_rows_loaded": 0,
        "model_requests": 0,
        "training_split_sha256": metadata["split_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GEPA train-only v0.4 inputs without model calls")
    parser.add_argument("--episodes-dir", type=Path, default=Path(".cache/emergent_ood_v0_4/episodes"))
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--execute", action="store_true", help="run optional GEPA after both independent gates pass")
    parser.add_argument("--resource-preflight", type=Path)
    parser.add_argument("--capability-episode-dir", type=Path)
    parser.add_argument("--capability-split-seed", type=int)
    parser.add_argument("--capability-ledger", type=Path)
    parser.add_argument("--sender-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--receiver-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--reflection-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--sender-model")
    parser.add_argument("--receiver-model")
    parser.add_argument("--reflection-model")
    parser.add_argument("--sender-tokenizer-id")
    parser.add_argument("--receiver-tokenizer-id")
    parser.add_argument("--reflection-tokenizer-id")
    parser.add_argument("--model-population-id", default="local-unspecified")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--request-budget", type=int)
    parser.add_argument("--max-metric-calls", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sender-instruction", default="State each attribute name and its exact value in one short English sentence.")
    parser.add_argument("--receiver-instruction", default="Match every attribute and return the exact candidate_id.")
    parser.add_argument("--resume", action="store_true", help="resume the exact GEPA run directory and durable request ledger")
    args = parser.parse_args()
    try:
        estimate = dry_run(args.episodes_dir, args.split_seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not args.execute:
        print(json.dumps(estimate, sort_keys=True))
        return
    required = {
        "--resource-preflight": args.resource_preflight,
        "--capability-episode-dir": args.capability_episode_dir,
        "--capability-split-seed": args.capability_split_seed,
        "--capability-ledger": args.capability_ledger,
        "--sender-model": args.sender_model,
        "--receiver-model": args.receiver_model,
        "--reflection-model": args.reflection_model,
        "--sender-tokenizer-id": args.sender_tokenizer_id,
        "--receiver-tokenizer-id": args.receiver_tokenizer_id,
        "--reflection-tokenizer-id": args.reflection_tokenizer_id,
        "--run-dir": args.run_dir,
        "--request-budget": args.request_budget,
        "--max-metric-calls": args.max_metric_calls,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("--execute requires " + ", ".join(missing))
    if args.request_budget < 1 or args.max_metric_calls < 1:
        parser.error("request and metric budgets must be positive")
    try:
        sender_client = OpenAICompatibleClient(
            args.sender_base_url, args.sender_model, timeout_seconds=45.0,
            max_tokens=160, follow_redirects=False, temperature=0.0,
        )
        receiver_client = OpenAICompatibleClient(
            args.receiver_base_url, args.receiver_model, timeout_seconds=45.0,
            max_tokens=48, follow_redirects=False, temperature=0.0,
        )
        reflection_client = OpenAICompatibleClient(
            args.reflection_base_url, args.reflection_model, timeout_seconds=60.0,
            max_tokens=1024, follow_redirects=False, temperature=0.0,
        )
        adapter = TacitGEPAAdapter(
            episode_dir=args.episodes_dir,
            split_seed=args.split_seed,
            sender_model=sender_client,
            receiver_model=receiver_client,
            request_budget=args.request_budget,
            request_ledger_path=args.run_dir / "request-ledger.json",
            model_population_id=args.model_population_id,
        )
        _result, manifest = run_gepa_optimization(
            adapter,
            reflection_client=reflection_client,
            run_dir=args.run_dir,
            preflight_path=args.resource_preflight,
            capability_episode_dir=args.capability_episode_dir,
            capability_split_seed=args.capability_split_seed,
            capability_ledger_path=args.capability_ledger,
            receiver_tokenizer_id=args.receiver_tokenizer_id,
            sender_tokenizer_id=args.sender_tokenizer_id,
            reflection_tokenizer_id=args.reflection_tokenizer_id,
            max_metric_calls=args.max_metric_calls,
            seed=args.seed,
            seed_candidate={
                "sender_instruction": args.sender_instruction,
                "receiver_instruction": args.receiver_instruction,
            },
            resume=args.resume,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
