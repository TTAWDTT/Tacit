import json
import re
import unittest

from examples.multiparty_private_sum import MESSAGE_FORMATS, _decode_sender_message, run_sum_episode
from tacit import ChatCompletion, LocalTCPMessageChannel


def encode_fake_value(message_format, value):
    if message_format == "decimal":
        return str(value)
    if message_format == "json":
        return json.dumps({"value": value}, separators=(",", ":"))
    if message_format == "labeled":
        return f"v={value}"
    if message_format == "binary":
        return f"{value:02b}"
    if message_format == "sentence":
        return f"My private integer is {value}."
    raise AssertionError(message_format)


class FakeSumAgent:
    def __init__(self, name, private_value=None, raw_output=None, message_format="decimal"):
        self.name = name
        self.private_value = private_value
        self.raw_output = raw_output
        self.message_format = message_format
        self.calls = []
        self.system_instructions = []

    def complete(self, messages):
        payload = json.loads(messages[-1]["content"])
        self.system_instructions.append(messages[0]["content"])
        self.calls.append(payload)
        if self.private_value is not None:
            output = (
                encode_fake_value(self.message_format, self.private_value)
                if self.raw_output is None else self.raw_output
            )
            return ChatCompletion(output, f"fake-{self.name}")
        transcript = payload["visible_transcript"]
        if "All private integers, in sender order: " in payload["private_context"]:
            values = payload["private_context"].rsplit(": ", 1)[1]
            answer = sum(int(value) for value in values.split(", "))
        else:
            match = re.search(r"sender_count=(\d+)", payload["private_context"])
            sender_count = 0 if match is None else int(match.group(1))
            parsed_values = [
                _decode_sender_message(self.message_format, entry["message"])
                for entry in transcript
            ]
            known_values = [value for value in parsed_values if value is not None]
            missing_count = max(0, sender_count - len(known_values))
            answer = sum(known_values) + (3 * missing_count) // 2
        return ChatCompletion(str(answer), f"fake-{self.name}")


class MultipartyPrivateSumExampleTests(unittest.TestCase):
    def run_with_fakes(self, values, condition="communicate", message_format="decimal"):
        senders = [
            FakeSumAgent(f"S{i + 1}", value, message_format=message_format)
            for i, value in enumerate(values)
        ]
        receiver = FakeSumAgent("R", message_format=message_format)
        result = run_sum_episode(
            values,
            sender_clients=senders,
            receiver_client=receiver,
            condition=condition,
            message_format=message_format,
        )
        return result, senders, receiver

    def test_communicating_senders_keep_inputs_private_and_receiver_gets_only_messages(self):
        result, senders, receiver = self.run_with_fakes([1, 2, 3])
        self.assertTrue(result["exact_success"])
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(len(result["message_transmissions"]), 3)
        for value, sender in zip([1, 2, 3], senders):
            self.assertEqual(
                sender.calls[0]["private_context"],
                f"Public metadata: sender_count=3. Your private integer is {value}.",
            )
            self.assertEqual(sender.calls[0]["visible_transcript"], [])
        self.assertEqual(receiver.calls[0]["private_context"], "Public metadata: sender_count=3.")
        self.assertIn("public/private context and received sender messages", receiver.system_instructions[0])
        self.assertEqual(
            [entry["message"] for entry in receiver.calls[0]["visible_transcript"]],
            ["1", "2", "3"],
        )
        self.assertTrue(result["all_sender_messages_syntax_valid"])
        self.assertTrue(result["all_sender_values_faithful"])

    def test_final_sum_success_does_not_hide_sender_value_errors(self):
        senders = [
            FakeSumAgent("S1", 1, raw_output="0"),
            FakeSumAgent("S2", 2, raw_output="3"),
        ]
        receiver = FakeSumAgent("R")
        result = run_sum_episode([1, 2], sender_clients=senders, receiver_client=receiver)
        self.assertTrue(result["exact_success"])  # errors cancel: 0 + 3 == 1 + 2
        self.assertTrue(result["all_sender_messages_syntax_valid"])
        self.assertFalse(result["all_sender_values_faithful"])
        self.assertEqual(result["sender_value_faithful_count"], 0)

    def test_task_prompt_is_constant_as_sender_count_grows(self):
        observed_tasks = []
        for values in ([0, 1], [0, 1, 2, 3, 0]):
            result, senders, receiver = self.run_with_fakes(list(values))
            self.assertTrue(result["exact_success"])
            observed_tasks.extend(call["task"] for sender in senders for call in sender.calls)
            observed_tasks.extend(call["task"] for call in receiver.calls)
        self.assertEqual(len(set(observed_tasks)), 1)
        self.assertNotIn("S1, S2", observed_tasks[0])

    def test_frozen_message_formats_decode_and_score_sender_fidelity(self):
        protocol_ids = set()
        for message_format in MESSAGE_FORMATS:
            result, _senders, _receiver = self.run_with_fakes(
                [0, 1, 2, 3], message_format=message_format,
            )
            self.assertTrue(result["exact_success"], message_format)
            self.assertTrue(result["all_sender_messages_syntax_valid"], message_format)
            self.assertTrue(result["all_sender_values_faithful"], message_format)
            protocol_ids.add(result["protocol_id"])
        self.assertEqual(len(protocol_ids), len(MESSAGE_FORMATS))

    def test_duplicate_json_field_fails_syntax_even_when_bayes_guess_saves_answer(self):
        senders = [
            FakeSumAgent("S1", 1, raw_output='{"value":1,"value":1}', message_format="json"),
            FakeSumAgent("S2", 2, message_format="json"),
        ]
        receiver = FakeSumAgent("R", message_format="json")
        result = run_sum_episode(
            [1, 2], sender_clients=senders, receiver_client=receiver, message_format="json",
        )
        self.assertTrue(result["exact_success"])  # the optimal missing-value guess happens to match this input
        self.assertEqual(result["sender_syntax_valid_count"], 1)
        self.assertEqual(result["sender_value_faithful_count"], 1)

    def test_budget_rejection_separates_generated_from_delivered_message_scores(self):
        values = [1, 2, 3]
        senders = [FakeSumAgent(f"S{i + 1}", value) for i, value in enumerate(values)]
        receiver = FakeSumAgent("R")
        protocol_id = "private-sum-decimal-v0"
        channel = LocalTCPMessageChannel(lambda _envelope: None)
        first_cost = channel.measure(
            "1", protocol_id=protocol_id, round_number=1, sender="S1", recipient="R",
        ).total_application_bytes
        second_minimum = channel.measure(
            "", protocol_id=protocol_id, round_number=2, sender="S2", recipient="R",
        ).total_application_bytes

        result = run_sum_episode(
            values,
            sender_clients=senders,
            receiver_client=receiver,
            wire_budget_bytes=first_cost + second_minimum,
        )

        self.assertEqual(result["stop_reason"], "wire_budget_exhausted")
        self.assertEqual(result["model_calls"], 3)  # S1, rejected S2 output, sealed receiver answer
        self.assertEqual(result["wire_bytes"], first_cost)
        self.assertEqual(result["sender_generated_message_count"], 2)
        self.assertEqual(result["sender_delivered_message_count"], 1)
        self.assertEqual(result["sender_value_faithful_count"], 2)
        self.assertEqual(result["sender_delivered_value_faithful_count"], 1)
        self.assertEqual([row["delivered"] for row in result["sender_messages"]], [True, False])
        self.assertEqual(result["prediction"], 4)  # delivered 1 plus the mode for two missing inputs
        self.assertEqual(
            [row["message"] for row in receiver.calls[0]["visible_transcript"]],
            ["1"],
        )
        self.assertFalse(result["exact_success"])
        self.assertEqual(senders[2].calls, [])

    def test_no_message_and_full_information_are_distinct_controls(self):
        no_message, senders, receiver = self.run_with_fakes([0, 0], "no_message")
        self.assertEqual(no_message["model_calls"], 1)
        self.assertEqual(no_message["message_transmissions"], [])
        self.assertEqual([len(sender.calls) for sender in senders], [0, 0])
        self.assertFalse(no_message["exact_success"])
        self.assertEqual(no_message["prediction"], 3)  # Bayes-optimal mode for two missing uniform inputs
        self.assertEqual(receiver.calls[0]["private_context"], "Public metadata: sender_count=2.")
        self.assertIsNone(no_message["all_sender_messages_syntax_valid"])
        self.assertIsNone(no_message["all_sender_values_faithful"])

        full_information, senders, receiver = self.run_with_fakes([1, 2], "full_information")
        self.assertTrue(full_information["exact_success"])
        self.assertEqual(full_information["model_calls"], 1)
        self.assertEqual([len(sender.calls) for sender in senders], [0, 0])
        self.assertIn("sender_count=2", receiver.calls[0]["private_context"])
        self.assertIn("1, 2", receiver.calls[0]["private_context"])
        self.assertIn("public/private context and received sender messages", receiver.system_instructions[0])
        self.assertIsNone(full_information["all_sender_values_faithful"])

    def test_rejects_invalid_inputs_before_any_client_call(self):
        senders = [FakeSumAgent("S1", 4), FakeSumAgent("S2", 0)]
        receiver = FakeSumAgent("R")
        with self.assertRaises(ValueError):
            run_sum_episode([4, 0], sender_clients=senders, receiver_client=receiver)
        self.assertEqual(senders[0].calls, [])
        self.assertEqual(receiver.calls, [])

    def test_request_cap_rejects_oversized_batch_before_any_client_call(self):
        senders = [FakeSumAgent(f"S{i + 1}", 0) for i in range(3)]
        receiver = FakeSumAgent("R")
        with self.assertRaisesRegex(ValueError, "exceeding request cap"):
            run_sum_episode(
                [0, 0, 0],
                sender_clients=senders,
                receiver_client=receiver,
                request_cap=3,
            )
        self.assertTrue(all(sender.calls == [] for sender in senders))
        self.assertEqual(receiver.calls, [])

    def test_frozen_ceilings_cannot_be_raised(self):
        senders = [FakeSumAgent("S1", 1), FakeSumAgent("S2", 2)]
        receiver = FakeSumAgent("R")
        with self.assertRaisesRegex(ValueError, "frozen 12-call ceiling"):
            run_sum_episode(
                [1, 2], sender_clients=senders, receiver_client=receiver, request_cap=13,
            )
        with self.assertRaisesRegex(ValueError, "between zero and 4096"):
            run_sum_episode(
                [1, 2],
                sender_clients=senders,
                receiver_client=receiver,
                wire_budget_bytes=4097,
            )
        self.assertTrue(all(sender.calls == [] for sender in senders))
        self.assertEqual(receiver.calls, [])


if __name__ == "__main__":
    unittest.main()
