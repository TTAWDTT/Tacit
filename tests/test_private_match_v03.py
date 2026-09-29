from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from itertools import product

from tacit import ChatCompletion, exchange_dialogue
from experiments.private_match_v0_3.protocols import (
    PROTOCOL_IDS, parse_coordinate_message, protocol_by_id,
)
from experiments.private_match_v0_3.runner import (
    main as runner_main, planned_model_calls, run_condition, validate_capability_ledger,
)
from contextlib import redirect_stdout
from experiments.private_match_v0_3.bit_frontier import (
    frontier as triadic_bit_frontier, optimal_success_probability,
)


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "private_match_v0_3"
    / "generate_tasks.py"
)
SPEC = importlib.util.spec_from_file_location("private_match_v0_3", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class CoordinateOracle:
    def __init__(self, coordinate: str) -> None:
        self.coordinate = coordinate
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        request = json.loads(messages[1]["content"])
        view = json.loads(request["private_context"])
        self_value = view["private_value"]
        self.assert_role(view)
        return ChatCompletion(f"{self.coordinate}={self_value}", f"oracle-{self.coordinate}")

    def assert_role(self, view):
        if view["coordinate"] != self.coordinate:
            raise AssertionError("sender received the other role's private view")


class CandidateTableOracle:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        request = json.loads(messages[1]["content"])
        receiver = json.loads(request["private_context"])
        values = {}
        for entry in request["visible_transcript"]:
            coordinate, value = entry["message"].split("=", 1)
            values[coordinate] = value
        answer = module.oracle_answer(receiver, values["x"], values["y"])
        return ChatCompletion(answer, "oracle-receiver")


class FrozenFormatSender:
    model_name = "fake-sender-v1"

    def __init__(self, protocol_id):
        self.protocol_id = protocol_id.split(":", 1)[0]

    def complete(self, messages):
        request = json.loads(messages[1]["content"])
        view = json.loads(request["private_context"])
        coordinate = view["coordinate"]
        value = view["private_value"]
        if self.protocol_id == "compact_kv":
            text = f"{coordinate}={value}"
        elif self.protocol_id == "strict_json":
            text = json.dumps({coordinate: value}, separators=(",", ":"))
        elif self.protocol_id == "fixed_binary":
            text = f"{int(value[1:]):02b}"
        else:
            text = f"The {coordinate} coordinate is {value}."
        return ChatCompletion(text, self.model_name, input_tokens=10, output_tokens=4)


class FrozenFormatReceiver:
    model_name = "fake-receiver-v1"

    def complete(self, messages):
        request = json.loads(messages[1]["content"])
        view = json.loads(request["private_context"])
        if "calibration_coordinates" in view:
            coords = view["calibration_coordinates"]
            answer = module.oracle_answer(view, coords["x"], coords["y"])
        else:
            values = {}
            for entry in request["visible_transcript"]:
                coordinate, message = entry["sender"], entry["message"]
                role_coord = "x" if coordinate == "sender_x" else "y"
                if message.startswith("{"):
                    value = json.loads(message)[role_coord]
                elif "=" in message:
                    value = message.split("=", 1)[1]
                elif message and set(message) <= {"0", "1"}:
                    value = f"{role_coord}{int(message, 2):04d}"
                else:
                    value = message.rsplit(" ", 1)[-1].rstrip(".")
                values[role_coord] = value
            if len(values) == 2:
                answer = module.oracle_answer(view, values["x"], values["y"])
            else:
                answer = view["candidates"][0]["candidate_id"]
        return ChatCompletion(answer, self.model_name, input_tokens=20, output_tokens=1)


class PrivateMatchV03Tests(unittest.TestCase):
    def test_triadic_bit_frontier_matches_exhaustive_encoder_pairs(self):
        q = 4
        width = q.bit_length() - 1
        for budget in range(2 * width + 2):
            best = Fraction(0)
            for x_bits in range(width + 1):
                for y_bits in range(width + 1):
                    if x_bits + y_bits > budget:
                        continue
                    x_classes = max(
                        len(set(encoder))
                        for encoder in product(range(1 << x_bits), repeat=q)
                    )
                    y_classes = max(
                        len(set(encoder))
                        for encoder in product(range(1 << y_bits), repeat=q)
                    )
                    best = max(best, Fraction(x_classes * y_classes, q * q))
            self.assertEqual(optimal_success_probability(q=q, total_payload_bits=budget), best)
        self.assertEqual(
            [row["success_fraction"] for row in triadic_bit_frontier(q=4, max_total_payload_bits=4)],
            ["1/16", "1/8", "1/4", "1/2", "1/1"],
        )

    def test_frozen_protocols_have_strict_decoders_and_fail_closed(self):
        self.assertEqual(len(PROTOCOL_IDS), 4)
        expected = {
            "compact_kv": "x=x0002",
            "strict_json": '{"x":"x0002"}',
            "fixed_binary": "10",
        }
        for name, valid in expected.items():
            with self.subTest(name=name):
                self.assertEqual(parse_coordinate_message(name, valid, q=4, sender="sender_x"), (True, True, "x0002"))
                self.assertFalse(parse_coordinate_message(name, valid + " ", q=4, sender="sender_x")[0])
                protocol = protocol_by_id(name, 4)
                self.assertIn("pmt3-prompts-1", protocol.protocol_id)
        self.assertEqual(parse_coordinate_message("concise_nl", "The x coordinate is x0002.", q=4, sender="sender_x"), (None, None, None))

    def test_model_free_runner_rows_cover_calibration_no_message_and_both_sources(self):
        for condition, planned in (("full_information", 1), ("no_message", 1), ("both_sources", 3)):
            self.assertEqual(planned_model_calls(1, condition), planned)
            protocol = protocol_by_id("compact_kv", 4)
            sender = None if condition in {"full_information", "no_message"} else FrozenFormatSender(protocol.protocol_id)
            row = run_condition(seed=304001, q=4, condition=condition, protocol=protocol,
                sender_model=sender, receiver_model=FrozenFormatReceiver(),
                sender_tokenizer_id="fake-tokenizer", receiver_tokenizer_id="fake-tokenizer",
                model_population_id="fake-population-v1")
            self.assertEqual(len(row["model_calls"]), planned)
            self.assertEqual(len(row["transmissions"]), 2 if condition == "both_sources" else 0)
            self.assertEqual(row["outcome"]["joint_success"], condition in {"full_information", "both_sources"})
            self.assertEqual(row["diagnostics"]["generation_seed"], 304001)

    def test_runner_defaults_to_dry_run_and_capability_ledger_is_seed_disjoint(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(runner_main(["--condition", "both_sources", "--episodes", "4"]), 0)
        self.assertIn('"inference_started": false', output.getvalue())
        rows = [run_condition(seed=303000 + i, q=4, condition="full_information",
            protocol=protocol_by_id("compact_kv", 4), sender_model=None,
            receiver_model=FrozenFormatReceiver(), sender_tokenizer_id=None,
            receiver_tokenizer_id="fake-tokenizer", model_population_id="fake-population-v1")
            for i in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "calibration.jsonl"
            ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            validate_capability_ledger(ledger, episodes=2, seed=304000, q=4,
                receiver_model="fake-receiver-v1", receiver_tokenizer_id="fake-tokenizer",
                model_population_id="fake-population-v1")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                validate_capability_ledger(ledger, episodes=2, seed=303000, q=4,
                    receiver_model="fake-receiver-v1", receiver_tokenizer_id="fake-tokenizer",
                    model_population_id="fake-population-v1")

    def test_generation_is_deterministic_and_role_separated(self):
        first = module.generate_episode(episode_id="triad-1", seed=42, q=4)
        second = module.generate_episode(episode_id="triad-1", seed=42, q=4)
        self.assertEqual(first, second)
        sender_x, sender_y, receiver, gold = first
        self.assertTrue(sender_x["private_value"].startswith("x"))
        self.assertNotIn("private_value", receiver)
        self.assertNotIn("target_candidate_id", sender_x)
        self.assertNotIn("target_candidate_id", sender_y)
        self.assertNotIn("target_candidate_id", receiver)
        self.assertIn("target_candidate_id", gold)
        self.assertEqual(sender_x["episode_id"], sender_y["episode_id"])

    def test_each_source_alone_leaves_q_candidates_and_both_are_unique(self):
        for q in (2, 4, 8, 16):
            for seed in range(12):
                sender_x, sender_y, receiver, gold = module.generate_episode(
                    episode_id=f"q{q}-seed{seed}", seed=seed, q=q
                )
                x_matches = [
                    row["candidate_id"] for row in receiver["candidates"]
                    if row["record"]["x"] == sender_x["private_value"]
                ]
                y_matches = [
                    row["candidate_id"] for row in receiver["candidates"]
                    if row["record"]["y"] == sender_y["private_value"]
                ]
                self.assertEqual(len(x_matches), q)
                self.assertEqual(len(y_matches), q)
                self.assertEqual(
                    module.oracle_answer(receiver, sender_x["private_value"], sender_y["private_value"]),
                    gold["target_candidate_id"],
                )
                self.assertTrue(module.score_answer(receiver, gold, gold["target_candidate_id"]))
                self.assertFalse(module.score_answer(receiver, gold, "not-a-candidate"))

    def test_exact_bayes_references_and_zero_error_bit_bound(self):
        self.assertEqual(
            module.bayes_accuracy_references(4),
            {
                "no_message": Fraction(1, 16),
                "sender_x_only": Fraction(1, 4),
                "sender_y_only": Fraction(1, 4),
                "both_sources": Fraction(1, 1),
            },
        )
        for q in (2, 4, 8, 16):
            self.assertEqual(2 * (q.bit_length() - 1), (q * q).bit_length() - 1)
        for q in (0, 1, 3, 6, True):
            with self.subTest(q=q), self.assertRaises(ValueError):
                module.bayes_accuracy_references(q)

    def test_validator_rejects_cross_role_leaks_and_incomplete_candidate_table(self):
        sender_x, sender_y, receiver, gold = module.generate_episode(
            episode_id="validate-1", seed=7, q=4
        )
        leaked_x = dict(sender_x, target_candidate_id=gold["target_candidate_id"])
        with self.assertRaisesRegex(ValueError, "leaks"):
            module.validate_episode(leaked_x, sender_y, receiver, gold)
        missing_row_receiver = dict(receiver, candidates=receiver["candidates"][:-1])
        with self.assertRaisesRegex(ValueError, "exactly q squared"):
            module.validate_episode(sender_x, sender_y, missing_row_receiver, gold)

    def test_three_agent_oracle_uses_real_unicast_channel_and_sealed_submission(self):
        sender_x, sender_y, receiver, gold = module.generate_episode(
            episode_id="loopback-oracle-1", seed=777, q=4
        )
        agents = {
            "sender_x": CoordinateOracle("x"),
            "sender_y": CoordinateOracle("y"),
            "receiver": CandidateTableOracle(),
        }

        class Protocol:
            protocol_id = "triadic-oracle-integration-v1"
            agent_instructions = {
                "sender_x": "Send only your x coordinate.",
                "sender_y": "Send only your y coordinate.",
                "receiver": "Use only your table and routed messages.",
            }

        result = exchange_dialogue(
            agents,
            protocol=Protocol(),
            private_contexts={
                "sender_x": json.dumps(sender_x),
                "sender_y": json.dumps(sender_y),
                "receiver": json.dumps(receiver),
            },
            schedule=(("sender_x", "receiver"), ("sender_y", "receiver")),
            task="Find the candidate row matching both private coordinates.",
            max_turns=2,
            final_answer_agent="receiver",
            final_answer_instruction="Return only the exact candidate ID for scoring.",
        )

        self.assertEqual(result.model_calls, 3)
        self.assertEqual(result.stop_reason, "schedule_complete")
        self.assertEqual(len(result.transmission_records()), 2)
        self.assertEqual(
            [row["recipients"] for row in result.transmission_records()],
            [["receiver"], ["receiver"]],
        )
        self.assertTrue(module.score_answer(receiver, gold, result.final_submission.text))
        self.assertEqual(
            [row["stage"] for row in result.model_call_records()],
            ["dialogue_turn", "dialogue_turn", "final_answer"],
        )
        self.assertNotIn(result.final_submission.text, [
            turn.completion.text for turn in result.turns
        ])

    def test_dataset_manifest_hashes_roles_and_protects_existing_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "shard"
            manifest = module.generate_dataset(output, episodes=5, seed=3000, q=4)
            self.assertEqual(manifest["agent_count"], 3)
            self.assertEqual(manifest["candidate_count"], 16)
            for name in module.ROLE_FILES:
                path = output / name
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    manifest["files_sha256"][name],
                )
                self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 5)
            with self.assertRaises(FileExistsError):
                module.generate_dataset(output, episodes=5, seed=3000, q=4)
            replaced = module.generate_dataset(output, episodes=2, seed=4000, q=2, force=True)
            self.assertEqual(replaced["episodes"], 2)
            with (output / "receiver.jsonl").open(encoding="utf-8") as stream:
                receiver = json.loads(next(stream))
            self.assertEqual(len(receiver["candidates"]), 4)


if __name__ == "__main__":
    unittest.main()
