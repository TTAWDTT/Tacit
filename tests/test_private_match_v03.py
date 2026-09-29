from __future__ import annotations

import hashlib
import copy
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
    PROTOCOL_IDS, encode_coordinate_message, parse_coordinate_message, protocol_by_id,
)
from experiments.private_match_v0_3.runner import (
    episode_id_for_seed, main as runner_main, planned_model_calls, run_condition,
    validate_capability_ledger,
)
from experiments.private_match_v0_3.generate_tasks import (
    generate_episode_for_target, model_visible_view,
)
from experiments.private_match_v0_3.prior_shift import evaluate_prior_shift
from experiments.private_match_v0_3.select_nl_baseline import select_natural_language_baseline
from contextlib import redirect_stdout
from experiments.private_match_v0_3.bit_frontier import (
    frontier as triadic_bit_frontier, optimal_nonuniform_success_probability,
    optimal_nonuniform_codebook, optimal_success_probability,
)
from experiments.private_match_v0_3.report import main as report_main, private_match_report
from tools.cost_report import RecordError
from tools.frontier_report import frontier_report
from tools.paired_report import paired_report


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "private_match_v0_3"
    / "generate_tasks.py"
)
TEST_TASK_KEY = bytes(range(32))
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
        if "episode_id" in view:
            raise AssertionError("evaluator episode ID leaked into a sender prompt")
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
        if "episode_id" in receiver:
            raise AssertionError("evaluator episode ID leaked into a receiver prompt")
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
        if "episode_id" in view:
            raise AssertionError("evaluator episode ID leaked into a sender prompt")
        coordinate = view["coordinate"]
        value = view["private_value"]
        if self.protocol_id == "compact_kv":
            text = f"{coordinate}={value}"
        elif self.protocol_id == "decimal_index":
            text = str(int(value[1:]))
        elif self.protocol_id == "short_nl":
            text = f"{coordinate} is {value}."
        elif self.protocol_id == "strict_json":
            text = json.dumps({coordinate: value}, separators=(",", ":"))
        elif self.protocol_id == "fixed_binary":
            text = f"{int(value[1:]):02b}"
        else:
            text = f"The {coordinate} coordinate is {value}."
        return ChatCompletion(text, self.model_name, input_tokens=10, output_tokens=4)


class FrozenFormatReceiver:
    model_name = "fake-receiver-v1"

    def __init__(self, protocol_id="compact_kv"):
        self.protocol_id = protocol_id

    def complete(self, messages):
        request = json.loads(messages[1]["content"])
        view = json.loads(request["private_context"])
        if "episode_id" in view:
            raise AssertionError("evaluator episode ID leaked into a receiver prompt")
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
                elif self.protocol_id == "decimal_index":
                    value = f"{role_coord}{int(message):04d}"
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
    def test_hmac_bounded_sampling_rejects_out_of_range_values(self):
        rng = module._HMACRandom(TEST_TASK_KEY, domain="test", seed=9)
        drawn = iter((3, 2, 0))
        rng.getrandbits = lambda count: next(drawn)
        self.assertEqual(rng.randbelow(3), 2)
        self.assertEqual(rng.randbelow(1), 0)
        with self.assertRaises(ValueError):
            rng.randbelow(0)

    def test_nonuniform_frontier_matches_exhaustive_small_encoder_pairs(self):
        px = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))
        py = (Fraction(2, 3), Fraction(1, 3), Fraction(0, 1))

        def classes(code):
            grouped = {}
            for index, label in enumerate(code):
                grouped.setdefault(label, []).append(index)
            return list(grouped.values())

        for budget in range(5):
            exhaustive = Fraction(0)
            for x_bits in range(budget + 1):
                for y_bits in range(budget - x_bits + 1):
                    x_labels = min(len(px), 1 << x_bits)
                    y_labels = min(len(py), 1 << y_bits)
                    x_codes = list(product(range(x_labels), repeat=len(px)))
                    y_codes = list(product(range(y_labels), repeat=len(py)))
                    for x_code in x_codes:
                        x_classes = classes(x_code)
                        for y_code in y_codes:
                            y_classes = classes(y_code)
                            success = sum((
                                max(px[x] * py[y] for x in x_group for y in y_group)
                                for x_group in x_classes for y_group in y_classes
                            ), Fraction(0))
                            exhaustive = max(exhaustive, success)
            actual = optimal_nonuniform_success_probability(
                probabilities_x=px, probabilities_y=py, total_payload_bits=budget,
            )
            self.assertEqual(actual, exhaustive)
            self.assertEqual(
                optimal_nonuniform_success_probability(
                    probabilities_x=(Fraction(1, 4),) * 4,
                    probabilities_y=(Fraction(1, 4),) * 4,
                    total_payload_bits=budget,
                ),
                optimal_success_probability(q=4, total_payload_bits=budget),
            )
        self.assertEqual(
            optimal_nonuniform_success_probability(
                probabilities_x=px, probabilities_y=py, total_payload_bits=10**6,
            ),
            Fraction(1, 1),
        )
        for invalid in ((Fraction(1, 2), Fraction(1, 3)),
                        (Fraction(1, 2), Fraction(1, 2), Fraction(-1, 1)),
                        (0.5, 0.5)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                optimal_nonuniform_success_probability(
                    probabilities_x=invalid, probabilities_y=py, total_payload_bits=1,
                )

    def test_nonuniform_codebook_is_optimal_and_transfer_keeps_decoder_frozen(self):
        training_prior = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8))
        shifted_prior = (Fraction(1, 8), Fraction(1, 8), Fraction(1, 2), Fraction(1, 4))
        codebook = optimal_nonuniform_codebook(probabilities=training_prior, payload_bits=1)

        self.assertEqual(codebook.symbols, (0, 1, 1, 1))
        self.assertEqual(codebook.representatives, (0, 1))
        self.assertEqual(codebook.success_probability(training_prior), Fraction(3, 4))
        self.assertEqual(codebook.success_probability(shifted_prior), Fraction(1, 4))
        self.assertEqual(
            codebook.worst_case_success_probability(training_prior, tv_radius=Fraction(1, 8)),
            Fraction(5, 8),
        )
        nearby_priors = []
        denominator = 8
        for a in range(denominator + 1):
            for b in range(denominator - a + 1):
                for c in range(denominator - a - b + 1):
                    d = denominator - a - b - c
                    candidate = tuple(Fraction(value, denominator) for value in (a, b, c, d))
                    tv = sum((abs(left - right) for left, right in zip(training_prior, candidate)),
                             Fraction(0)) / 2
                    if tv <= Fraction(1, 8):
                        nearby_priors.append(candidate)
        self.assertEqual(
            min(codebook.success_probability(prior) for prior in nearby_priors),
            codebook.worst_case_success_probability(
                training_prior, tv_radius=Fraction(1, 8),
            ),
        )
        shifted_optimum = optimal_nonuniform_codebook(
            probabilities=shifted_prior, payload_bits=1,
        )
        self.assertEqual(shifted_optimum.success_probability(shifted_prior), Fraction(3, 4))
        perfect_codebook = optimal_nonuniform_codebook(
            probabilities=training_prior, payload_bits=2,
        )
        self.assertEqual(
            perfect_codebook.worst_case_success_probability(training_prior, tv_radius=1),
            Fraction(1),
        )
        with self.assertRaises(ValueError):
            codebook.worst_case_success_probability(training_prior, tv_radius=1.1)

        for bits in range(3):
            exhaustive = Fraction(0)
            for encoding in product(range(1 << bits), repeat=len(training_prior)):
                grouped = {}
                for value_index, symbol in enumerate(encoding):
                    grouped.setdefault(symbol, []).append(value_index)
                decoded = {
                    symbol: min(group, key=lambda index: (-training_prior[index], index))
                    for symbol, group in grouped.items()
                }
                score = sum((
                    training_prior[index]
                    for index, symbol in enumerate(encoding)
                    if decoded[symbol] == index
                ), Fraction(0))
                exhaustive = max(exhaustive, score)
            self.assertEqual(
                optimal_nonuniform_codebook(
                    probabilities=training_prior, payload_bits=bits,
                ).success_probability(training_prior),
                exhaustive,
            )

        for value_index in range(4):
            payload = codebook.encode(value_index)
            self.assertEqual(len(payload), 1)
            self.assertEqual(
                codebook.decode(payload),
                codebook.representatives[codebook.symbols[value_index]],
            )
        no_message = optimal_nonuniform_codebook(probabilities=training_prior, payload_bits=0)
        self.assertEqual(no_message.encode(3), "")
        self.assertEqual(no_message.decode(""), 0)
        with self.assertRaises(ValueError):
            codebook.decode("2")
        with self.assertRaises(ValueError):
            codebook.encode(4)

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
        self.assertEqual(len(PROTOCOL_IDS), 6)
        expected = {
            "compact_kv": "x=x0002",
            "decimal_index": "2",
            "short_nl": "x is x0002.",
            "strict_json": '{"x":"x0002"}',
            "fixed_binary": "10",
        }
        for name, valid in expected.items():
            with self.subTest(name=name):
                self.assertEqual(parse_coordinate_message(name, valid, q=4, sender="sender_x"), (True, True, "x0002"))
                self.assertFalse(parse_coordinate_message(name, valid + " ", q=4, sender="sender_x")[0])
                protocol = protocol_by_id(name, 4)
                self.assertIn("pmt3-prompts-5", protocol.protocol_id)
        self.assertEqual(parse_coordinate_message("concise_nl", "The x coordinate is x0002.", q=4, sender="sender_x"), (True, True, "x0002"))
        self.assertEqual(parse_coordinate_message("concise_nl", "x is probably 0002", q=4, sender="sender_x"), (None, None, None))
        self.assertEqual(parse_coordinate_message("short_nl", "x is x0002.", q=4, sender="sender_x"), (True, True, "x0002"))
        self.assertEqual(parse_coordinate_message("short_nl", "x is probably x0002", q=4, sender="sender_x"), (None, None, None))
        for malformed in ("", " 2", "02", "+2", "2.0", "4"):
            self.assertFalse(parse_coordinate_message(
                "decimal_index", malformed, q=4, sender="sender_x",
            )[0])

    def test_registered_encoders_roundtrip_every_value_across_q(self):
        for q in (2, 4, 8, 16, 64):
            for name in PROTOCOL_IDS:
                for sender, coordinate in (("sender_x", "x"), ("sender_y", "y")):
                    for index in range(q):
                        value = f"{coordinate}{index:04d}"
                        message = encode_coordinate_message(name, q=q, sender=sender, value=value)
                        with self.subTest(q=q, name=name, sender=sender, value=value):
                            self.assertEqual(
                                parse_coordinate_message(name, message, q=q, sender=sender),
                                (True, True, value),
                            )
        for args in (
            ("compact_kv", 4, "sender_x", "y0001"),
            ("fixed_binary", 4, "sender_x", "x0004"),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                encode_coordinate_message(args[0], q=args[1], sender=args[2], value=args[3])
        for name in PROTOCOL_IDS:
            message = encode_coordinate_message(
                name, q=16384, sender="sender_x", value="x10000"
            )
            self.assertEqual(
                parse_coordinate_message(name, message, q=16384, sender="sender_x"),
                (True, True, "x10000"),
            )

    def test_model_free_runner_rows_cover_calibration_no_message_and_both_sources(self):
        for condition, planned in (("full_information", 1), ("no_message", 1), ("both_sources", 3)):
            self.assertEqual(planned_model_calls(1, condition), planned)
            protocol = protocol_by_id("compact_kv", 4)
            sender = None if condition in {"full_information", "no_message"} else FrozenFormatSender(protocol.protocol_id)
            row = run_condition(seed=304001, q=4, condition=condition, protocol=protocol,
                sender_model=sender, receiver_model=FrozenFormatReceiver(protocol.protocol_id),
                sender_tokenizer_id="fake-tokenizer", receiver_tokenizer_id="fake-tokenizer",
                model_population_id="fake-population-v1", task_key=TEST_TASK_KEY)
            self.assertEqual(len(row["model_calls"]), planned)
            self.assertEqual(len(row["transmissions"]), 2 if condition == "both_sources" else 0)
            self.assertEqual(row["outcome"]["joint_success"], condition in {"full_information", "both_sources"})
            self.assertEqual(row["diagnostics"]["generation_seed"], 304001)

    def test_representation_arms_share_policy_but_version_code_and_decoder(self):
        rows = [run_condition(seed=304002, q=4, condition="both_sources",
            protocol=protocol_by_id(name, 4), sender_model=FrozenFormatSender(name),
            receiver_model=FrozenFormatReceiver(name), sender_tokenizer_id="fake-tokenizer",
            receiver_tokenizer_id="fake-tokenizer", model_population_id="fake-population-v1",
            task_key=TEST_TASK_KEY)
            for name in PROTOCOL_IDS]
        self.assertEqual({row["protocol"]["policy_id"] for row in rows}, {"fixed-x-then-y-unicast-v1"})
        self.assertEqual(len({row["protocol"]["code_id"] for row in rows}), len(PROTOCOL_IDS))
        self.assertEqual(len({row["protocol"]["decoder_id"] for row in rows}), len(PROTOCOL_IDS))
        paired = paired_report(rows, replicates=100, seed=4)
        self.assertEqual(len(paired["comparisons"]), 15)
        self.assertTrue(all(item["control_alignment"]["policy_matched"] for item in paired["comparisons"]))
        self.assertTrue(all(not item["control_alignment"]["decoder_matched"] for item in paired["comparisons"]))
        self.assertTrue(all(item["control_alignment"]["code_differs"] for item in paired["comparisons"]))
        frontier = frontier_report(rows)
        self.assertEqual(len(frontier["groups"]), 1)
        report = private_match_report(rows, replicates=100, seed=5, task_key=TEST_TASK_KEY)
        self.assertEqual(report["schema_version"], "tlu.private-match-report.v2")
        self.assertEqual(report["analytic_controls"][0]["bayes_accuracy"]["no_message"], "1/16")
        self.assertEqual(report["analytic_controls"][0]["ideal_fixed_width_total_payload_bits"][-1]["success_fraction"], "1/1")
        self.assertEqual(len(report["representation_diagnostics"]), len(PROTOCOL_IDS))
        corrupted_outcome = copy.deepcopy(rows)
        corrupted_outcome[0]["outcome"]["joint_success"] = not corrupted_outcome[0]["outcome"]["joint_success"]
        with self.assertRaisesRegex(RecordError, "exact scorer"):
            private_match_report(corrupted_outcome, replicates=100, seed=5, task_key=TEST_TASK_KEY)
        corrupted_wire = copy.deepcopy(rows)
        corrupted_wire[0]["transmissions"][0]["payload_metadata"]["logical_text_utf8_bytes"] += 1
        with self.assertRaisesRegex(RecordError, "logical text byte count"):
            private_match_report(corrupted_wire, replicates=100, seed=5, task_key=TEST_TASK_KEY)
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "batch.jsonl"
            output = Path(temporary) / "report.json"
            key_file = Path(temporary) / "task.key"
            key_file.write_bytes(TEST_TASK_KEY)
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(report_main([str(ledger), "--replicates", "100",
                                          "--task-key-file", str(key_file),
                                          "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"],
                             "tlu.private-match-report.v2")
        with self.assertRaisesRegex(RecordError, "task parameters"):
            private_match_report(rows, replicates=100, seed=5,
                                 task_key=bytes(reversed(range(32))))

    def test_runner_defaults_to_dry_run_and_capability_ledger_is_seed_disjoint(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(runner_main(["--condition", "both_sources", "--episodes", "4"]), 0)
        self.assertIn('"inference_started": false', output.getvalue())
        rows = [run_condition(seed=303000 + i, q=4, condition="full_information",
            protocol=protocol_by_id("compact_kv", 4), sender_model=None,
            receiver_model=FrozenFormatReceiver(), sender_tokenizer_id=None,
            receiver_tokenizer_id="fake-tokenizer", model_population_id="fake-population-v1",
            task_key=TEST_TASK_KEY)
            for i in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "calibration.jsonl"
            ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            validate_capability_ledger(ledger, episodes=2, seed=304000, q=4,
                receiver_model="fake-receiver-v1", receiver_tokenizer_id="fake-tokenizer",
                model_population_id="fake-population-v1", task_key=TEST_TASK_KEY)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                validate_capability_ledger(ledger, episodes=2, seed=303000, q=4,
                    receiver_model="fake-receiver-v1", receiver_tokenizer_id="fake-tokenizer",
                    model_population_id="fake-population-v1", task_key=TEST_TASK_KEY)
            with self.assertRaisesRegex(ValueError, "task parameters differ"):
                validate_capability_ledger(ledger, episodes=2, seed=304000, q=4,
                    receiver_model="fake-receiver-v1", receiver_tokenizer_id="fake-tokenizer",
                    model_population_id="fake-population-v1",
                    task_key=bytes(reversed(range(32))))

    def test_generation_is_deterministic_and_role_separated(self):
        first = module.generate_episode(episode_id="triad-1", seed=42, q=4, task_key=TEST_TASK_KEY)
        second = module.generate_episode(episode_id="triad-1", seed=42, q=4, task_key=TEST_TASK_KEY)
        other_key = module.generate_episode(episode_id="triad-1", seed=42, q=4,
                                            task_key=bytes(reversed(range(32))))
        self.assertEqual(first, second)
        self.assertNotEqual(first[2]["candidates"], other_key[2]["candidates"])
        self.assertNotEqual(first[3]["target_candidate_id"], other_key[3]["target_candidate_id"])
        sender_x, sender_y, receiver, gold = first
        self.assertTrue(sender_x["private_value"].startswith("x"))
        self.assertNotIn("private_value", receiver)
        self.assertNotIn("target_candidate_id", sender_x)
        self.assertNotIn("target_candidate_id", sender_y)
        self.assertNotIn("target_candidate_id", receiver)
        self.assertIn("target_candidate_id", gold)
        self.assertEqual(sender_x["episode_id"], sender_y["episode_id"])

    def test_designated_target_preserves_uniform_generator_and_table_randomness(self):
        ordinary = module.generate_episode(
            episode_id="preserve-uniform", seed=7001, q=4, task_key=TEST_TASK_KEY,
        )
        sender_x, sender_y = ordinary[:2]
        designated = generate_episode_for_target(
            episode_id="preserve-uniform", seed=7001, q=4, task_key=TEST_TASK_KEY,
            x_index=int(sender_x["private_value"][1:]),
            y_index=int(sender_y["private_value"][1:]),
        )
        self.assertEqual(ordinary, designated)

        other_target = generate_episode_for_target(
            episode_id="other-target", seed=7001, q=4, task_key=TEST_TASK_KEY,
            x_index=3, y_index=2,
        )
        self.assertEqual(ordinary[2]["candidates"], other_target[2]["candidates"])
        self.assertEqual(
            other_target[3]["target_candidate_id"],
            module.oracle_answer(other_target[2], "x0003", "y0002"),
        )

    def test_exact_prior_shift_cohort_matches_frozen_and_adapted_theory(self):
        result = evaluate_prior_shift(
            task_key=TEST_TASK_KEY,
            probabilities_x_train=(Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)),
            probabilities_y_train=(Fraction(1, 4),) * 4,
            probabilities_x_eval=(Fraction(1, 8), Fraction(1, 8), Fraction(1, 2), Fraction(1, 4)),
            probabilities_y_eval=(Fraction(1, 4),) * 4,
            payload_bits_x=1,
            payload_bits_y=2,
            seed=910_000,
        )
        self.assertEqual(result["episode_count"], 32)
        self.assertEqual(result["conditions"]["frozen_training_codebook"]["successes"], 8)
        self.assertEqual(result["conditions"]["frozen_training_codebook"]["success_fraction"], "1/4")
        self.assertEqual(result["conditions"]["evaluation_adapted_codebook"]["successes"], 24)
        self.assertEqual(result["conditions"]["evaluation_adapted_codebook"]["success_fraction"], "3/4")
        with self.assertRaisesRegex(ValueError, "over cap"):
            evaluate_prior_shift(
                task_key=TEST_TASK_KEY,
                probabilities_x_train=(Fraction(1, 4),) * 4,
                probabilities_y_train=(Fraction(1, 4),) * 4,
                probabilities_x_eval=(Fraction(1, 8), Fraction(1, 8), Fraction(1, 2), Fraction(1, 4)),
                probabilities_y_eval=(Fraction(1, 4),) * 4,
                payload_bits_x=1,
                payload_bits_y=1,
                seed=910_000,
                max_episodes=31,
            )

    def test_nl_baseline_selector_uses_only_paired_development_ledgers(self):
        class WrongConciseSender(FrozenFormatSender):
            def complete(self, messages):
                request = json.loads(messages[1]["content"])
                view = json.loads(request["private_context"])
                coordinate = view["coordinate"]
                value = view["private_value"]
                index = int(value[1:])
                wrong_value = f"x{(index + 1) % 4:04d}" if coordinate == "x" else value
                text = f"The {coordinate} coordinate is {wrong_value}."
                return ChatCompletion(text, self.model_name, input_tokens=10, output_tokens=4)

        rows_by_protocol = {"concise_nl": [], "short_nl": []}
        for protocol_id in rows_by_protocol:
            protocol = protocol_by_id(protocol_id, 4)
            sender = (WrongConciseSender(protocol_id) if protocol_id == "concise_nl"
                      else FrozenFormatSender(protocol_id))
            receiver = FrozenFormatReceiver(protocol_id)
            for seed in range(302000, 302008):
                rows_by_protocol[protocol_id].append(run_condition(
                    seed=seed, q=4, condition="both_sources", protocol=protocol,
                    sender_model=sender, receiver_model=receiver,
                    sender_tokenizer_id="fake-tokenizer", receiver_tokenizer_id="fake-tokenizer",
                    model_population_id="fake-population-v1", task_key=TEST_TASK_KEY,
                    split="development",
                ))

        with tempfile.TemporaryDirectory() as temporary:
            paths = {}
            for protocol_id, rows in rows_by_protocol.items():
                path = Path(temporary) / f"{protocol_id}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                paths[protocol_id] = path
            selection = select_natural_language_baseline(
                ledger_paths=paths, task_key=TEST_TASK_KEY,
            )
            evaluation_rows = copy.deepcopy(rows_by_protocol["short_nl"])
            for row in evaluation_rows:
                row["stratum"]["split"] = "evaluation"
                row["diagnostics"]["split"] = "evaluation"
            evaluation_path = Path(temporary) / "evaluation-short-nl.jsonl"
            evaluation_path.write_text(
                "".join(json.dumps(row) + "\n" for row in evaluation_rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "development records only"):
                select_natural_language_baseline(
                    ledger_paths={"concise_nl": paths["concise_nl"], "short_nl": evaluation_path},
                    task_key=TEST_TASK_KEY,
                )
        self.assertEqual(selection["selected_protocol_id"], "short_nl")
        self.assertEqual(selection["candidate_metrics"][0]["joint_successes"], 0)
        self.assertEqual(selection["candidate_metrics"][1]["joint_successes"], 8)
        self.assertEqual(selection["optimizer_setup_cost"]["development_model_calls"], 48)
        self.assertTrue(selection["optimizer_setup_cost"]["account_as_optimizer_setup"])
        self.assertFalse(selection["evaluation_data_used"])

    def test_full_information_cannot_be_labeled_as_development(self):
        with self.assertRaisesRegex(ValueError, "are calibration"):
            run_condition(
                seed=303000, q=4, condition="full_information",
                protocol=protocol_by_id("concise_nl", 4), sender_model=None,
                receiver_model=None, sender_tokenizer_id=None,
                receiver_tokenizer_id="fake-tokenizer",
                model_population_id="fake-population-v1", task_key=TEST_TASK_KEY,
                split="development",
            )

    def test_task_key_file_is_256_bit_private_and_refuses_accidental_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_path = Path(temporary) / ".cache" / "private_match" / "task.key"
            key_id = module.create_task_key(key_path)
            self.assertEqual(len(key_path.read_bytes()), 32)
            self.assertEqual(key_id, module.task_key_id(module.load_task_key(key_path)))
            with self.assertRaises(FileExistsError):
                module.create_task_key(key_path)
            with self.assertRaises(ValueError):
                module._validate_task_key(b"short")

    def test_each_source_alone_leaves_q_candidates_and_both_are_unique(self):
        for q in (2, 4, 8, 16):
            for seed in range(12):
                sender_x, sender_y, receiver, gold = module.generate_episode(
                    episode_id=f"q{q}-seed{seed}", seed=seed, q=q, task_key=TEST_TASK_KEY
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
            episode_id="validate-1", seed=7, q=4, task_key=TEST_TASK_KEY
        )
        leaked_x = dict(sender_x, target_candidate_id=gold["target_candidate_id"])
        with self.assertRaisesRegex(ValueError, "leaks"):
            module.validate_episode(leaked_x, sender_y, receiver, gold)
        missing_row_receiver = dict(receiver, candidates=receiver["candidates"][:-1])
        with self.assertRaisesRegex(ValueError, "exactly q squared"):
            module.validate_episode(sender_x, sender_y, missing_row_receiver, gold)

    def test_three_agent_oracle_uses_real_unicast_channel_and_sealed_submission(self):
        sender_x, sender_y, receiver, gold = module.generate_episode(
            episode_id="loopback-oracle-1", seed=777, q=4, task_key=TEST_TASK_KEY
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
                "sender_x": json.dumps(model_visible_view(sender_x)),
                "sender_y": json.dumps(model_visible_view(sender_y)),
                "receiver": json.dumps(model_visible_view(receiver)),
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
            manifest = module.generate_dataset(output, episodes=5, seed=3000, q=4,
                                                task_key=TEST_TASK_KEY)
            self.assertEqual(manifest["task_key_id"], module.task_key_id(TEST_TASK_KEY))
            self.assertEqual(manifest["generator_version"], "0.3.3")
            self.assertNotIn(TEST_TASK_KEY.hex(), json.dumps(manifest))
            self.assertEqual(manifest["agent_count"], 3)
            self.assertEqual(manifest["candidate_count"], 16)
            for name in module.ROLE_FILES:
                path = output / name
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    manifest["files_sha256"][name],
                )
                self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 5)
            with (output / "sender_x.jsonl").open(encoding="utf-8") as stream:
                shard_sender_x = json.loads(next(stream))
            runner_sender_x = module.generate_episode(
                episode_id=episode_id_for_seed(3000), seed=3000, q=4,
                task_key=TEST_TASK_KEY,
            )[0]
            self.assertEqual(shard_sender_x["episode_id"], runner_sender_x["episode_id"])
            self.assertEqual(shard_sender_x, runner_sender_x)
            with self.assertRaises(FileExistsError):
                module.generate_dataset(output, episodes=5, seed=3000, q=4,
                                        task_key=TEST_TASK_KEY)
            replaced = module.generate_dataset(output, episodes=2, seed=4000, q=2,
                                               task_key=TEST_TASK_KEY, force=True)
            self.assertEqual(replaced["episodes"], 2)
            with (output / "receiver.jsonl").open(encoding="utf-8") as stream:
                receiver = json.loads(next(stream))
            self.assertEqual(len(receiver["candidates"]), 4)


if __name__ == "__main__":
    unittest.main()
