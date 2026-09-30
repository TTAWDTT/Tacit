import json
import unittest

from examples.multiparty_private_sum import run_sum_episode
from tacit import ChatCompletion


class FakeSumAgent:
    def __init__(self, name, private_value=None):
        self.name = name
        self.private_value = private_value
        self.calls = []

    def complete(self, messages):
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload)
        if self.private_value is not None:
            return ChatCompletion(str(self.private_value), f"fake-{self.name}")
        transcript = payload["visible_transcript"]
        if transcript:
            answer = sum(int(entry["message"]) for entry in transcript)
        elif payload["private_context"].startswith("All private integers, in sender order: "):
            values = payload["private_context"].rsplit(": ", 1)[1]
            answer = sum(int(value) for value in values.split(", "))
        else:
            answer = 0
        return ChatCompletion(str(answer), f"fake-{self.name}")


class MultipartyPrivateSumExampleTests(unittest.TestCase):
    def run_with_fakes(self, values, condition="communicate"):
        senders = [FakeSumAgent(f"S{i + 1}", value) for i, value in enumerate(values)]
        receiver = FakeSumAgent("R")
        result = run_sum_episode(
            values,
            sender_clients=senders,
            receiver_client=receiver,
            condition=condition,
        )
        return result, senders, receiver

    def test_communicating_senders_keep_inputs_private_and_receiver_gets_only_messages(self):
        result, senders, receiver = self.run_with_fakes([1, 2, 3])
        self.assertTrue(result["exact_success"])
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(len(result["message_transmissions"]), 3)
        for value, sender in zip([1, 2, 3], senders):
            self.assertEqual(sender.calls[0]["private_context"], f"Your private integer is {value}.")
            self.assertEqual(sender.calls[0]["visible_transcript"], [])
        self.assertEqual(receiver.calls[0]["private_context"], "")
        self.assertEqual(
            [entry["message"] for entry in receiver.calls[0]["visible_transcript"]],
            ["1", "2", "3"],
        )

    def test_no_message_and_full_information_are_distinct_controls(self):
        no_message, senders, _receiver = self.run_with_fakes([1, 2], "no_message")
        self.assertEqual(no_message["model_calls"], 1)
        self.assertEqual(no_message["message_transmissions"], [])
        self.assertEqual([len(sender.calls) for sender in senders], [0, 0])
        self.assertFalse(no_message["exact_success"])

        full_information, senders, receiver = self.run_with_fakes([1, 2], "full_information")
        self.assertTrue(full_information["exact_success"])
        self.assertEqual(full_information["model_calls"], 1)
        self.assertEqual([len(sender.calls) for sender in senders], [0, 0])
        self.assertIn("1, 2", receiver.calls[0]["private_context"])

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
